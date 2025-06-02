package org.tensorflow.lite.examples.objectdetection
import java.io.ByteArrayOutputStream

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Build
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import org.tensorflow.lite.examples.objectdetection.databinding.ActivityMainBinding
import org.tensorflow.lite.task.vision.detector.Detection
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.*

class MainActivity : AppCompatActivity(), ObjectDetectorHelper.DetectorListener, TextToSpeech.OnInitListener {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var objectDetectorHelper: ObjectDetectorHelper
    private lateinit var textToSpeech: TextToSpeech

    private var lastAnnouncedLabel: String? = null
    private var lastVoiceAlertTime = 0L
    private val voiceAlertInterval = 3000L
    private var dynamicThreshold: Float = 0.5f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        textToSpeech = TextToSpeech(this, this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        objectDetectorHelper = ObjectDetectorHelper(
            threshold = dynamicThreshold,
            numThreads = 4,
            maxResults = 5,
            currentDelegate = ObjectDetectorHelper.DELEGATE_CPU,
            currentModel = ObjectDetectorHelper.MODEL_EFFICIENTDETV0,
            context = this,
            objectDetectorListener = this
        )

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
        else {
            Toast.makeText(this, "Camera permission is required", Toast.LENGTH_SHORT).show()
            finish()
        }
    }

    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(
        this, Manifest.permission.CAMERA
    ) == PackageManager.PERMISSION_GRANTED

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }

            val analyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, this::processImageProxy)
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analyzer)
            } catch (e: Exception) {
                Log.e("MainActivity", "Camera binding failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processImageProxy(imageProxy: ImageProxy) {
        val mediaImage = imageProxy.image ?: return
        val bitmap = imageProxy.toBitmap() ?: return
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees

        objectDetectorHelper.threshold = dynamicThreshold
        objectDetectorHelper.detect(bitmap, rotationDegrees)

        imageProxy.close()
    }

    private fun ImageProxy.toBitmap(): Bitmap? {
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
        return BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())
    }

    override fun onResults(results: MutableList<Detection>?, inferenceTime: Long, imageHeight: Int, imageWidth: Int) {
        runOnUiThread {
            if (results == null) return@runOnUiThread

            // Draw bounding boxes
            binding.overlayView.setResults(results, imageHeight, imageWidth)
            binding.overlayView.invalidate()

            val currentTime = System.currentTimeMillis()

            // Region-based alert
            var leftCount = 0
            var centerCount = 0
            var rightCount = 0

            val thirdWidth = imageWidth / 3.0f

            results.forEach {
                val label = it.categories.firstOrNull()?.label ?: return@forEach
                if (!isVehicleLabel(label)) return@forEach

                val centerX = (it.boundingBox.left + it.boundingBox.right) / 2.0f

                when {
                    centerX < thirdWidth -> leftCount++
                    centerX < 2 * thirdWidth -> centerCount++
                    else -> rightCount++
                }
            }

            if (currentTime - lastVoiceAlertTime > voiceAlertInterval) {
                val spokenParts = mutableListOf<String>()
                if (leftCount > 0) spokenParts.add("$leftCount on left")
                if (centerCount > 0) spokenParts.add("$centerCount in center")
                if (rightCount > 0) spokenParts.add("$rightCount on right")

                if (spokenParts.isNotEmpty()) {
                    speakOut("Vehicle ${spokenParts.joinToString(", ")}")
                    lastVoiceAlertTime = currentTime
                }
            }

            // Adjust dynamic threshold based on object spread
            val xCenters = results.map { (it.boundingBox.left + it.boundingBox.right) / 2.0f }
            if (xCenters.isNotEmpty()) {
                val range = max(1f, (xCenters.maxOrNull() ?: 0f) - (xCenters.minOrNull() ?: 0f))
                val norm = range / imageWidth
                dynamicThreshold = (0.3f + 0.4f * (1f - norm)).coerceIn(0.1f, 0.9f)
            }
        }
    }

    private fun isVehicleLabel(label: String): Boolean {
        return listOf("car", "truck", "bus", "motorcycle").contains(label.lowercase(Locale.getDefault()))
    }

    private fun speakOut(text: String) {
        textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, "")
    }

    override fun onError(error: String) {
        runOnUiThread { Toast.makeText(this, "Error: $error", Toast.LENGTH_SHORT).show() }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            textToSpeech.language = Locale.US
        } else {
            Toast.makeText(this, "TTS initialization failed", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onDestroy() {
        cameraExecutor.shutdown()
        textToSpeech.shutdown()
        objectDetectorHelper.clearObjectDetector()
        super.onDestroy()
    }
}
