package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import org.tensorflow.lite.task.vision.detector.Detection
import java.util.LinkedList
import kotlin.math.max

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results: List<Detection> = LinkedList()
    private var boxPaint = Paint()
    private var textPaint = Paint()
    private var textBackgroundPaint = Paint()
    private var linePaint = Paint()
    private var bounds = Rect()
    private var scaleFactor: Float = 1f

    init {
        initPaints()
    }

    private fun initPaints() {
        boxPaint.color = ContextCompat.getColor(context!!, R.color.bounding_box_color)
        boxPaint.strokeWidth = 8F
        boxPaint.style = Paint.Style.STROKE

        textPaint.color = Color.WHITE
        textPaint.textSize = 50f
        textPaint.style = Paint.Style.FILL

        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL

        linePaint.color = Color.WHITE
        linePaint.strokeWidth = 4f
        linePaint.style = Paint.Style.STROKE
        linePaint.pathEffect = DashPathEffect(floatArrayOf(10f, 10f), 0f)
    }

    fun setResults(
        detectionResults: MutableList<Detection>,
        imageHeight: Int,
        imageWidth: Int,
    ) {
        results = detectionResults
        scaleFactor = max(width * 1f / imageWidth, height * 1f / imageHeight)
        invalidate()
    }

    fun clear() {
        results = emptyList()
        invalidate()
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        // Draw vertical division lines for left/center/right regions
        val third = width / 3f
        canvas.drawLine(third, 0f, third, height.toFloat(), linePaint)
        canvas.drawLine(2 * third, 0f, 2 * third, height.toFloat(), linePaint)

        for (result in results) {
            val box = result.boundingBox
            val left = box.left * scaleFactor
            val top = box.top * scaleFactor
            val right = box.right * scaleFactor
            val bottom = box.bottom * scaleFactor

            val rectF = RectF(left, top, right, bottom)
            canvas.drawRect(rectF, boxPaint)

            val label = result.categories.firstOrNull()?.label ?: "?"
            val score = result.categories.firstOrNull()?.score ?: 0f
            val labelText = "$label ${"%.2f".format(score)}"

            textPaint.getTextBounds(labelText, 0, labelText.length, bounds)
            val textWidth = bounds.width().toFloat()
            val textHeight = bounds.height().toFloat()

            canvas.drawRect(
                left,
                top,
                left + textWidth + BOUNDING_RECT_TEXT_PADDING,
                top + textHeight + BOUNDING_RECT_TEXT_PADDING,
                textBackgroundPaint
            )

            canvas.drawText(labelText, left, top + textHeight, textPaint)
        }
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}
