package com.ml2.docscanner

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.os.SystemClock
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Bộ suy luận YOLOv11n-seg trên TFLite.
 *
 * Layout tensor (đã xác minh từ model export):
 *   input  "images"     : [1, 640, 640, 3]  float32, RGB, chuẩn hoá 0..1
 *   output "Identity"   : [1, F, 8400]       float32, F = 4(box) + nc(class) + 32(mask coeff)
 *   output "Identity_1" : [1, 160, 160, 32]  float32, mask prototypes (NHWC)
 * Toạ độ box (xywh) đã chuẩn hoá 0..1 theo input; điểm class đã qua sigmoid.
 */
class YoloSegModel(
    context: Context,
    modelAsset: String,
    val labels: List<String>,
    private val confThreshold: Float = 0.30f,
    private val iouThreshold: Float = 0.45f,
    numThreads: Int = 4
) {
    private val interpreter: Interpreter

    private val inputSize = 640
    private val protoSize = 160
    private val maskCoeffs = 32
    private val numBoxes: Int
    private val numFeatures: Int
    private val nc: Int = labels.size

    // buffer tái sử dụng
    private val inputBuffer: ByteBuffer
    private val pixels = IntArray(inputSize * inputSize)
    private val detOut: Array<Array<FloatArray>>      // [1, F, 8400]
    private val protoOut: Array<Array<Array<FloatArray>>> // [1, 160, 160, 32]

    // màu theo lớp
    private val classColors = intArrayOf(
        Color.argb(110, 0, 230, 80),    // lớp 0 / left_page -> xanh lá
        Color.argb(110, 30, 130, 255)   // lớp 1 / right_page -> xanh dương
    )

    init {
        val opts = Interpreter.Options().apply {
            setNumThreads(numThreads)
            setUseXNNPACK(true)
        }
        interpreter = Interpreter(loadModelFile(context, modelAsset), opts)

        val outShape = interpreter.getOutputTensor(0).shape() // [1, F, 8400]
        numFeatures = outShape[1]
        numBoxes = outShape[2]

        inputBuffer = ByteBuffer.allocateDirect(inputSize * inputSize * 3 * 4)
            .order(ByteOrder.nativeOrder())
        detOut = Array(1) { Array(numFeatures) { FloatArray(numBoxes) } }
        protoOut = Array(1) { Array(protoSize) { Array(protoSize) { FloatArray(maskCoeffs) } } }
    }

    private fun loadModelFile(context: Context, asset: String): ByteBuffer {
        context.assets.openFd(asset).use { fd ->
            java.io.FileInputStream(fd.fileDescriptor).use { input ->
                return input.channel.map(
                    FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength
                )
            }
        }
    }

    /** Suy luận trên 1 bitmap (đã xoay đứng). Trả về box chuẩn hoá + mask. */
    fun detect(bitmap: Bitmap): InferenceResult {
        val t0 = SystemClock.elapsedRealtime()
        val imgW = bitmap.width
        val imgH = bitmap.height
        val lb = preprocess(bitmap, imgW, imgH)

        val outputs = mapOf(0 to detOut, 1 to protoOut)
        interpreter.runForMultipleInputsOutputs(arrayOf<Any>(inputBuffer), outputs)

        val candidates = ArrayList<Cand>()
        val d = detOut[0]
        for (b in 0 until numBoxes) {
            var bestScore = 0f
            var bestCls = -1
            for (c in 0 until nc) {
                val s = d[4 + c][b]
                if (s > bestScore) { bestScore = s; bestCls = c }
            }
            if (bestScore < confThreshold) continue
            // xywh chuẩn hoá 0..1 -> pixel trong không gian 640 (letterboxed)
            val cx = d[0][b] * inputSize
            val cy = d[1][b] * inputSize
            val w = d[2][b] * inputSize
            val h = d[3][b] * inputSize
            val x1 = cx - w / 2f
            val y1 = cy - h / 2f
            val x2 = cx + w / 2f
            val y2 = cy + h / 2f
            val coeffs = FloatArray(maskCoeffs) { d[4 + nc + it][b] }
            candidates.add(Cand(x1, y1, x2, y2, bestScore, bestCls, coeffs))
        }

        val kept = nms(candidates)

        val detections = kept.map { c ->
            val mask = buildMask(c)
            // map box 640-letterbox -> ảnh gốc -> chuẩn hoá 0..1
            val bx1 = ((c.x1 - lb.padX) / lb.scale) / imgW
            val by1 = ((c.y1 - lb.padY) / lb.scale) / imgH
            val bx2 = ((c.x2 - lb.padX) / lb.scale) / imgW
            val by2 = ((c.y2 - lb.padY) / lb.scale) / imgH
            Detection(
                classId = c.cls,
                label = labels.getOrElse(c.cls) { "obj" },
                score = c.score,
                box = RectF(
                    bx1.coerceIn(0f, 1f), by1.coerceIn(0f, 1f),
                    bx2.coerceIn(0f, 1f), by2.coerceIn(0f, 1f)
                ),
                mask = mask
            )
        }
        return InferenceResult(detections, lb, imgW, imgH, SystemClock.elapsedRealtime() - t0)
    }

    /** Letterbox bitmap về 640x640, nạp vào inputBuffer (RGB /255). */
    private fun preprocess(bitmap: Bitmap, imgW: Int, imgH: Int): LetterboxInfo {
        val scale = min(inputSize / imgW.toFloat(), inputSize / imgH.toFloat())
        val newW = Math.round(imgW * scale)
        val newH = Math.round(imgH * scale)
        val padX = (inputSize - newW) / 2f
        val padY = (inputSize - newH) / 2f

        val canvasBmp = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(canvasBmp)
        canvas.drawColor(Color.rgb(114, 114, 114))
        val m = Matrix().apply {
            postScale(scale, scale)
            postTranslate(padX, padY)
        }
        canvas.drawBitmap(bitmap, m, Paint(Paint.FILTER_BITMAP_FLAG))

        canvasBmp.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        inputBuffer.rewind()
        for (p in pixels) {
            inputBuffer.putFloat(((p shr 16) and 0xFF) / 255f) // R
            inputBuffer.putFloat(((p shr 8) and 0xFF) / 255f)  // G
            inputBuffer.putFloat((p and 0xFF) / 255f)          // B
        }
        canvasBmp.recycle()
        return LetterboxInfo(scale, padX, padY)
    }

    /** Tạo mask 160x160 ARGB cho 1 detection (chỉ tính trong vùng bbox). */
    private fun buildMask(c: Cand): Bitmap {
        val proto = protoOut[0]
        val color = classColors[c.cls % classColors.size]
        val out = IntArray(protoSize * protoSize)

        // bbox trong không gian 160 (640 / 4)
        val mx1 = (c.x1 / 4f).toInt().coerceIn(0, protoSize - 1)
        val my1 = (c.y1 / 4f).toInt().coerceIn(0, protoSize - 1)
        val mx2 = (c.x2 / 4f).toInt().coerceIn(0, protoSize - 1)
        val my2 = (c.y2 / 4f).toInt().coerceIn(0, protoSize - 1)
        val coeffs = c.coeffs
        for (y in my1..my2) {
            val row = proto[y]
            val base = y * protoSize
            for (x in mx1..mx2) {
                val pc = row[x]
                var s = 0f
                for (k in 0 until maskCoeffs) s += coeffs[k] * pc[k]
                if (1f / (1f + exp(-s)) >= 0.5f) out[base + x] = color
            }
        }
        return Bitmap.createBitmap(out, protoSize, protoSize, Bitmap.Config.ARGB_8888)
    }

    private fun nms(cands: ArrayList<Cand>): List<Cand> {
        cands.sortByDescending { it.score }
        val kept = ArrayList<Cand>()
        val removed = BooleanArray(cands.size)
        for (i in cands.indices) {
            if (removed[i]) continue
            val a = cands[i]
            kept.add(a)
            for (j in i + 1 until cands.size) {
                if (removed[j]) continue
                if (iou(a, cands[j]) > iouThreshold) removed[j] = true
            }
        }
        return kept
    }

    private fun iou(a: Cand, b: Cand): Float {
        val ix1 = max(a.x1, b.x1); val iy1 = max(a.y1, b.y1)
        val ix2 = min(a.x2, b.x2); val iy2 = min(a.y2, b.y2)
        val iw = max(0f, ix2 - ix1); val ih = max(0f, iy2 - iy1)
        val inter = iw * ih
        val areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
        val areaB = (b.x2 - b.x1) * (b.y2 - b.y1)
        val union = areaA + areaB - inter
        return if (union <= 0f) 0f else inter / union
    }

    fun close() = interpreter.close()

    private class Cand(
        val x1: Float, val y1: Float, val x2: Float, val y2: Float,
        val score: Float, val cls: Int, val coeffs: FloatArray
    )
}
