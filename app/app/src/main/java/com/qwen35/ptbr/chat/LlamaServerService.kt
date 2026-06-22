package com.qwen35.ptbr.chat

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat

class LlamaServerService : Service() {

    companion object {
        const val TAG = "LlamaServerService"
        const val CHANNEL_ID = "llama_server_channel"
        const val NOTIFICATION_ID = 1

        const val EXTRA_MODEL_PATH = "model_path"
        const val EXTRA_PORT = "port"

        const val DEFAULT_PORT = 8080
        const val DEFAULT_THREADS = 3
        const val DEFAULT_BATCH_SIZE = 128
        const val DEFAULT_CTX_SIZE = 2048
    }

    private val lock = Any()
    @Volatile private var serverProcess: Process? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val modelPath = intent?.getStringExtra(EXTRA_MODEL_PATH)
            ?: run {
                Log.e(TAG, "No model path provided")
                return START_NOT_STICKY
            }

        val port = intent.getIntExtra(EXTRA_PORT, DEFAULT_PORT)

        val notification = buildNotification()
        startForeground(NOTIFICATION_ID, notification)

        Thread({
            startServer(modelPath, port)
        }, "llama-server-thread").start()

        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        stopServer()
        super.onDestroy()
    }

    private fun startServer(modelPath: String, port: Int) {
        try {
            val workspaceDir = prepareWorkspace()
            if (workspaceDir == null) {
                Log.e(TAG, "Failed to prepare workspace")
                return
            }

            val serverBinary = java.io.File(workspaceDir, "llama-server")
            if (!serverBinary.exists()) {
                Log.e(TAG, "llama-server binary not found at ${serverBinary.absolutePath}")
                return
            }

            serverBinary.setExecutable(true, false)

            val nativeLibDir = applicationInfo.nativeLibraryDir
            val ldPath = listOfNotNull(
                nativeLibDir,
                workspaceDir.absolutePath
            ).joinToString(":")

            val cmd = arrayOf(
                serverBinary.absolutePath,
                "-m", modelPath,
                "-t", DEFAULT_THREADS.toString(),
                "-b", DEFAULT_BATCH_SIZE.toString(),
                "-c", DEFAULT_CTX_SIZE.toString(),
                "-ctk", "q8_0",
                "-ctv", "q8_0",
                "--mlock",
                "--host", "127.0.0.1",
                "--port", port.toString()
            )

            Log.i(TAG, "Starting server: ${cmd.joinToString(" ")}")
            Log.i(TAG, "LD_LIBRARY_PATH=$ldPath")

            val processBuilder = ProcessBuilder(*cmd)
            processBuilder.directory(workspaceDir)
            processBuilder.environment()["LD_LIBRARY_PATH"] = ldPath
            processBuilder.redirectErrorStream(true)

            synchronized(lock) {
                serverProcess = processBuilder.start()
            }

            val reader = serverProcess!!.inputStream.bufferedReader()
            var line: String?
            while (reader.readLine().also { line = it } != null) {
                Log.d(TAG, "server: $line")
            }

            val exitCode = try {
                serverProcess?.waitFor() ?: -1
            } catch (e: InterruptedException) {
                -1
            }
            Log.i(TAG, "Server exited with code $exitCode")

        } catch (e: Exception) {
            Log.e(TAG, "Error starting server", e)
        }
    }

    private fun prepareWorkspace(): java.io.File? {
        return try {
            val workspaceDir = java.io.File(filesDir, "workspace")
            if (!workspaceDir.exists()) {
                workspaceDir.mkdirs()
            }

            val targetFile = java.io.File(workspaceDir, "llama-server")
            if (!targetFile.exists()) {
                try {
                    assets.open("llama-server").use { input ->
                        targetFile.outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }
                    targetFile.setExecutable(true, false)
                    Log.d(TAG, "Extracted llama-server to ${targetFile.absolutePath}")
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to extract llama-server: ${e.message}")
                    return null
                }
            }

            targetFile.setExecutable(true, false)
            workspaceDir
        } catch (e: Exception) {
            Log.e(TAG, "Error preparing workspace", e)
            null
        }
    }

    private fun stopServer() {
        synchronized(lock) {
            serverProcess?.let {
                it.destroy()
                serverProcess = null
            }
        }
    }

    private fun buildNotification(): Notification {
        val activityIntent = Intent(this, MainActivity::class.java)
        val activityPendingIntent = PendingIntent.getActivity(
            this, 0, activityIntent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.notification_running))
            .setContentText("porta 8080")
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setOngoing(true)
            .setContentIntent(activityPendingIntent)
            .build()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                getString(R.string.notification_channel_name),
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = getString(R.string.notification_channel_desc)
            }

            val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            manager.createNotificationChannel(channel)
        }
    }
}
