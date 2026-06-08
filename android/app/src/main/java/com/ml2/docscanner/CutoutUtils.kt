package com.ml2.docscanner

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint

/**
 * Cắt nền: dùng mask của model làm kênh alpha, xuất bitmap nền trong suốt.
 * Gộp mọi detection (vd. trang trái + phải) vào 1 ảnh.
 */
object CutoutUtils {

    fun cutout(original: Bitmap, r: InferenceResult): Bitmap {
        val w = original.width
        val h = original.height

        // 1) Dựng mask đầy đủ ở kích thước ảnh gốc từ các mask 160.
        //    maskPx160 -> 640 (x4) -> ảnh gốc: imagePx = (px*4 - pad) / scale
        val maskFull = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val mc = Canvas(maskFull)
        val sx = 4f / r.lb.scale
        val tx = -r.lb.padX / r.lb.scale
        val ty = -r.lb.padY / r.lb.scale
        val p = Paint(Paint.FILTER_BITMAP_FLAG) // bilinear -> viền mượt
        val m = Matrix()
        for (det in r.detections) {
            det.mask?.let {
                m.reset()
                m.setScale(sx, sx)
                m.postTranslate(tx, ty)
                mc.drawBitmap(it, m, p)
            }
        }

        // 2) Kết hợp: giữ pixel gốc nơi có mask, trong suốt nơi còn lại.
        val maskPx = IntArray(w * h)
        maskFull.getPixels(maskPx, 0, w, 0, 0, w, h)
        maskFull.recycle()
        val srcPx = IntArray(w * h)
        original.getPixels(srcPx, 0, w, 0, 0, w, h)

        val outPx = IntArray(w * h)
        for (i in outPx.indices) {
            val a = (maskPx[i] ushr 24) and 0xFF
            outPx[i] = if (a > 20) (srcPx[i] or 0xFF000000.toInt()) else 0
        }
        return Bitmap.createBitmap(outPx, w, h, Bitmap.Config.ARGB_8888)
    }
}
