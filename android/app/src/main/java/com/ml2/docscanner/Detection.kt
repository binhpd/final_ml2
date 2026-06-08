package com.ml2.docscanner

import android.graphics.Bitmap
import android.graphics.RectF

/**
 * Một đối tượng được phát hiện (1 trang giấy / trang trái / trang phải).
 *
 * @param box        Bounding box ở toạ độ CHUẨN HOÁ 0..1 theo ảnh phân tích (đã xoay đứng).
 * @param mask       Bitmap mask 160x160 (ARGB) nằm trong không gian 640-letterbox / 4.
 *                   Pixel thuộc vật thể được tô màu lớp, ngoài vật thể trong suốt.
 */
data class Detection(
    val classId: Int,
    val label: String,
    val score: Float,
    val box: RectF,
    val mask: Bitmap?
)

/**
 * Thông tin letterbox khi resize ảnh gốc về input 640x640 của model.
 * scale: tỉ lệ thu nhỏ; padX/padY: lề (pixel) ở không gian 640.
 */
data class LetterboxInfo(
    val scale: Float,
    val padX: Float,
    val padY: Float
)

/**
 * Kết quả 1 lần suy luận.
 * @param imageW/imageH kích thước ảnh phân tích (đã xoay đứng) mà box được chuẩn hoá theo.
 */
data class InferenceResult(
    val detections: List<Detection>,
    val lb: LetterboxInfo,
    val imageW: Int,
    val imageH: Int,
    val inferenceMs: Long
)
