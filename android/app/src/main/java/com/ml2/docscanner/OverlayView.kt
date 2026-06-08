package com.ml2.docscanner

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.max

/**
 * Lớp phủ trong suốt trên PreviewView để vẽ khung trang giấy được nhận diện.
 * Toạ độ box là chuẩn hoá 0..1 theo ảnh phân tích; mask ở không gian 160 (=640 letterbox / 4).
 * Ánh xạ theo CENTER_CROP để khớp với PreviewView (ScaleType FILL_CENTER mặc định).
 */
class OverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private var result: InferenceResult? = null

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        color = Color.argb(255, 0, 230, 80)
    }
    private val cornerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 12f
        strokeCap = Paint.Cap.ROUND
    }
    private val textBg = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.argb(180, 0, 0, 0) }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 38f
        isFakeBoldText = true
    }
    private val maskPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val maskMatrix = Matrix()

    private val classBorder = intArrayOf(
        Color.rgb(0, 230, 80),
        Color.rgb(40, 150, 255)
    )

    fun setResult(r: InferenceResult?) {
        result = r
        postInvalidate()
    }

    fun clear() {
        result = null
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val r = result ?: return
        if (r.imageW == 0 || r.imageH == 0) return

        // CENTER_CROP: ánh xạ ảnh -> view
        val vScale = max(width / r.imageW.toFloat(), height / r.imageH.toFloat())
        val offX = (width - r.imageW * vScale) / 2f
        val offY = (height - r.imageH * vScale) / 2f

        // ma trận mask: pixel 160 -> 640 (x4) -> ảnh (bỏ pad / scale) -> view (center-crop)
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

            // box -> view
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

    /** Góc kiểu khung scan cho dễ nhìn. */
    private fun drawCorners(canvas: Canvas, r: RectF) {
        val len = (minOf(r.width(), r.height()) * 0.12f).coerceAtMost(60f)
        // 4 góc
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
