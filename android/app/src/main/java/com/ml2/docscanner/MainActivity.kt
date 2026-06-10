package com.ml2.docscanner

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.ml2.docscanner.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService

    @Volatile private var model: YoloSegModel? = null
    @Volatile private var modelKind: ModelKind = ModelKind.DOC
    @Volatile private var busy = false

    // bitmap upright mới nhất từ analyzer, dùng cho lúc chụp
    private val latestLock = Any()
    private var latestBitmap: Bitmap? = null

    private enum class ModelKind(val asset: String, val labels: List<String>) {
        PAGE_V2("yolo_page_v2.tflite", listOf("page")),
        DOC("yolo_doc.tflite", listOf("document")),
        SPINE("yolo_spine.tflite", listOf("left_page", "right_page"))
    }

    private val requestPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
        else { Toast.makeText(this, "Cần quyền camera", Toast.LENGTH_LONG).show(); finish() }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        cameraExecutor = Executors.newSingleThreadExecutor()

        binding.modelToggle.check(binding.btnPageV2.id)
        binding.modelToggle.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (!isChecked) return@addOnButtonCheckedListener
            val kind = when (checkedId) {
                binding.btnSpine.id -> ModelKind.SPINE
                binding.btnDoc.id -> ModelKind.DOC
                else -> ModelKind.PAGE_V2
            }
            switchModel(kind)
        }

        binding.frameModeButton.addOnCheckedChangeListener { _, isChecked ->
            binding.overlay.setFrameMode(isChecked)
        }

        binding.captureButton.setOnClickListener { capture() }
        binding.closeResult.setOnClickListener {
            binding.resultContainer.visibility = android.view.View.GONE
        }

        // nạp model mặc định
        cameraExecutor.execute { loadModel(ModelKind.PAGE_V2) }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) startCamera() else requestPermission.launch(Manifest.permission.CAMERA)
    }

    private fun switchModel(kind: ModelKind) {
        if (kind == modelKind && model != null) return
        cameraExecutor.execute { loadModel(kind) }
    }

    private fun loadModel(kind: ModelKind) {
        try {
            model?.close()
            model = YoloSegModel(this, kind.asset, kind.labels)
            modelKind = kind
            runOnUiThread { binding.overlay.clear() }
        } catch (e: Exception) {
            Log.e(TAG, "Load model lỗi", e)
            runOnUiThread { Toast.makeText(this, "Lỗi nạp model: ${e.message}", Toast.LENGTH_LONG).show() }
        }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
            }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(cameraExecutor, ::analyze) }

            provider.unbindAll()
            provider.bindToLifecycle(
                this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(image: ImageProxy) {
        val m = model
        if (m == null || busy) { image.close(); return }
        busy = true
        try {
            val upright = image.toUprightBitmap()
            synchronized(latestLock) {
                latestBitmap?.recycle()
                latestBitmap = upright
            }
            val result = m.detect(upright)
            runOnUiThread {
                binding.overlay.setResult(result)
                binding.infoText.text =
                    "Model: ${modelKind.name}\n${result.detections.size} obj • ${result.inferenceMs} ms"
            }
        } catch (e: Exception) {
            Log.e(TAG, "analyze lỗi", e)
        } finally {
            busy = false
            image.close()
        }
    }

    private fun capture() {
        val src: Bitmap? = synchronized(latestLock) { latestBitmap?.copy(Bitmap.Config.ARGB_8888, false) }
        if (src == null) { Toast.makeText(this, "Chưa có khung hình", Toast.LENGTH_SHORT).show(); return }
        val m = model ?: return
        cameraExecutor.execute {
            try {
                val result = m.detect(src)
                if (result.detections.isEmpty()) {
                    runOnUiThread { Toast.makeText(this, "Không phát hiện trang giấy", Toast.LENGTH_SHORT).show() }
                    src.recycle(); return@execute
                }
                val cut = CutoutUtils.cutout(src, result)
                src.recycle()
                runOnUiThread {
                    binding.resultImage.setImageBitmap(cut)
                    binding.resultContainer.visibility = android.view.View.VISIBLE
                }
            } catch (e: Exception) {
                Log.e(TAG, "capture lỗi", e)
                runOnUiThread { Toast.makeText(this, "Lỗi cắt nền: ${e.message}", Toast.LENGTH_LONG).show() }
            }
        }
    }

    /** Chuyển ImageProxy (RGBA_8888) thành Bitmap đã xoay đứng. */
    private fun ImageProxy.toUprightBitmap(): Bitmap {
        val plane = planes[0]
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val bmp = Bitmap.createBitmap(
            rowStride / plane.pixelStride, height, Bitmap.Config.ARGB_8888
        )
        buffer.rewind()
        bmp.copyPixelsFromBuffer(buffer)
        // cắt bỏ padding rowStride nếu có
        val cropped = if (bmp.width != width) {
            Bitmap.createBitmap(bmp, 0, 0, width, height).also { if (it != bmp) bmp.recycle() }
        } else bmp

        val rot = imageInfo.rotationDegrees
        return if (rot == 0) cropped else {
            val matrix = Matrix().apply { postRotate(rot.toFloat()) }
            Bitmap.createBitmap(cropped, 0, 0, cropped.width, cropped.height, matrix, true)
                .also { if (it != cropped) cropped.recycle() }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        model?.close()
        synchronized(latestLock) { latestBitmap?.recycle(); latestBitmap = null }
    }

    companion object { private const val TAG = "DocScanner" }
}
