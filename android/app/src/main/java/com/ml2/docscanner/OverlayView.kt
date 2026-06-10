package com.ml2.docscanner

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.abs
import kotlin.math.max

/**
 * Lớp phủ trong suốt trên PreviewView.
 *
 * Hai chế độ hiển thị:
 *  - MASK (mặc định): vẽ mask tô màu + bbox + góc scanner cho MỌI detection.
 *  - FRAME (frameMode=true): chỉ vẽ ĐƯỜNG BAO XẤP XỈ (tứ giác 4 góc) của PAGE CHÍNH
 *    (detection lớn nhất), kèm hiệu ứng co vào / nở ra mượt (easing) khi camera đổi góc.
 *
 * Toạ độ box/góc chuẩn hoá 0..1 theo ảnh phân tích; mask ở không gian 160 (=640 letterbox / 4).
 * Ánh xạ CENTER_CROP để khớp PreviewView (FILL_CENTER).
 */
class OverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private var result: InferenceResult? = null
    var frameMode: Boolean = false
        private set

    // --- trạng thái animation cho FRAME mode ---
    private val displayCorners = FloatArray(8)   // góc đang hiển thị (chuẩn hoá 0..1)
    private var displayValid = false
    private val lerp = 0.28f                      // tốc độ co/nở (0..1)
    private val eps = 0.0012f

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeWidth = 6f; color = Color.argb(255, 0, 230, 80)
    }
    private val cornerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeWidth = 12f; strokeCap = Paint.Cap.ROUND
    }
    private val textBg = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.argb(180, 0, 0, 0) }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; textSize = 38f; isFakeBoldText = true
    }
    private val maskPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val maskMatrix = Matrix()

    // paint riêng cho FRAME mode
    private val framePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeWidth = 8f; strokeJoin = Paint.Join.ROUND
        color = Color.rgb(0, 230, 120)
    }
    private val frameFill = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL; color = Color.argb(60, 0, 230, 120)
    }
    private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL; color = Color.WHITE
    }
    private val framePath = Path()

    private val classBorder = intArrayOf(Color.rgb(0, 230, 80), Color.rgb(40, 150, 255))

    fun setFrameMode(on: Boolean) {
        frameMode = on
        displayValid = false
        postInvalidate()
    }

    fun setResult(r: InferenceResult?) {
        result = r
        postInvalidate()
    }

    fun clear() {
        result = null
        displayValid = false
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val r = result ?: return
        if (r.imageW == 0 || r.imageH == 0) return

        val vScale = max(width / r.imageW.toFloat(), height / r.imageH.toFloat())
        val offX = (width - r.imageW * vScale) / 2f
        val offY = (height - r.imageH * vScale) / 2f

        if (frameMode) drawFrameMode(canvas, r, vScale, offX, offY)
        else drawMaskMode(canvas, r, vScale, offX, offY)
    }

    // ---------------- FRAME MODE (page chính + co/nở) ----------------
    private fun drawFrameMode(canvas: Canvas, r: InferenceResult, vScale: Float, offX: Float, offY: Float) {
        val main = mainPage(r)
        val target = main?.corners
        if (target == null) {
            displayValid = false            // mất page -> ẩn khung
            return
        }
        if (!displayValid) {
            // xuất hiện lần đầu: khởi tạo tại tâm để có hiệu ứng "nở ra"
            var cx = 0f; var cy = 0f
            for (i in 0 until 4) { cx += target[i * 2]; cy += target[i * 2 + 1] }
            cx /= 4f; cy /= 4f
            for (i in 0 until 4) { displayCorners[i * 2] = cx; displayCorners[i * 2 + 1] = cy }
            displayValid = true
        }
        // easing: co/nở mượt về target
        var moving = false
        for (i in 0 until 8) {
            val diff = target[i] - displayCorners[i]
            if (abs(diff) > eps) moving = true
            displayCorners[i] += diff * lerp
        }

        // vẽ tứ giác (chuẩn hoá -> view)
        framePath.reset()
        for (i in 0 until 4) {
            val vx = offX + displayCorners[i * 2] * r.imageW * vScale
            val vy = offY + displayCorners[i * 2 + 1] * r.imageH * vScale
            if (i == 0) framePath.moveTo(vx, vy) else framePath.lineTo(vx, vy)
        }
        framePath.close()
        canvas.drawPath(framePath, frameFill)
        canvas.drawPath(framePath, framePaint)
        // chấm 4 góc
        for (i in 0 until 4) {
            val vx = offX + displayCorners[i * 2] * r.imageW * vScale
            val vy = offY + displayCorners[i * 2 + 1] * r.imageH * vScale
            canvas.drawCircle(vx, vy, 12f, dotPaint)
        }
        // nhãn
        val label = "${main.label} ${(main.score * 100).toInt()}%"
        val tx0 = offX + displayCorners[0] * r.imageW * vScale
        val ty0 = offY + displayCorners[1] * r.imageH * vScale
        val tw = textPaint.measureText(label)
        canvas.drawRect(tx0, ty0 - 46f, tx0 + tw + 16f, ty0, textBg)
        canvas.drawText(label, tx0 + 8f, ty0 - 12f, textPaint)

        if (moving) postInvalidateOnAnimation()   // tiếp tục animate tới khi hội tụ
    }

    /** Page chính = detection có diện tích bbox lớn nhất và có góc xấp xỉ. */
    private fun mainPage(r: InferenceResult): Detection? =
        r.detections.filter { it.corners != null }
            .maxByOrNull { it.box.width() * it.box.height() }

    // ---------------- MASK MODE (mặc định) ----------------
    private fun drawMaskMode(canvas: Canvas, r: InferenceResult, vScale: Float, offX: Float, offY: Float) {
        val sx = 4f * vScale / r.lb.scale
        val sy = 4f * vScale / r.lb.scale
        val tx = -r.lb.padX / r.lb.scale * vScale + offX
        val ty = -r.lb.padY / r.lb.scale * vScale + offY

        for (det in r.detections) {
            det.mask?.let { m ->
                maskMatrix.reset()
                maskMatrix.setScale(sx, sy)
                maskMatrix.postTranslate(tx, ty)
                canvas.drawBitmap(m, maskMatrix, maskPaint)
            }
            val left = offX + det.box.left * r.imageW * vScale
            val top = offY + det.box.top * r.imageH * vScale
            val right = offX + det.box.right * r.imageW * vScale
            val bottom = offY + det.box.bottom * r.imageH * vScale
            val rect = RectF(left, top, right, bottom)

            val col = classBorder[det.classId % classBorder.size]
            boxPaint.color = col
            cornerPaint.color = col
            canvas.drawRect(rect, boxPaint)
            drawCorners(canvas, rect)

            val label = "${det.label} ${(det.score * 100).toInt()}%"
            val tw = textPaint.measureText(label)
            canvas.drawRect(left, top - 46f, left + tw + 16f, top, textBg)
            canvas.drawText(label, left + 8f, top - 12f, textPaint)
        }
    }

    private fun drawCorners(canvas: Canvas, r: RectF) {
        val len = (minOf(r.width(), r.height()) * 0.12f).coerceAtMost(60f)
        canvas.drawLine(r.left, r.top, r.left + len, r.top, cornerPaint)
        canvas.drawLine(r.left, r.top, r.left, r.top + len, cornerPaint)
        canvas.drawLine(r.right, r.top, r.right - len, r.top, cornerPaint)
        canvas.drawLine(r.right, r.top, r.right, r.top + len, cornerPaint)
        canvas.drawLine(r.left, r.bottom, r.left + len, r.bottom, cornerPaint)
        canvas.drawLine(r.left, r.bottom, r.left, r.bottom - len, cornerPaint)
        canvas.drawLine(r.right, r.bottom, r.right - len, r.bottom, cornerPaint)
        canvas.drawLine(r.right, r.bottom, r.right, r.bottom - len, cornerPaint)
    }
}
