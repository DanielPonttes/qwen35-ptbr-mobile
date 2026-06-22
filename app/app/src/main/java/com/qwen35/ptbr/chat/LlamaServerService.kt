package com.qwen35.ptbr.chat

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import java.io.File

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

        const val ACTION_SERVER_READY = "com.qwen35.ptbr.chat.SERVER_READY"
        const val ACTION_SERVER_ERROR = "com.qwen35.ptbr.chat.SERVER_ERROR"
        const val EXTRA_STATUS = "status"
        const val STATUS_LOADING = "loading"
        const val STATUS_READY = "ready"
        const val STATUS_ERROR = "error"

        private const val MAX_RETRY_MS = 30_000L
        private const val BASE_RETRY_MS = 1_000L
    }

    private val lock = Any()
    @Volatile private var serverProcess: Process? = null
    @Volatile private var serverReady = false
    private var wakelock: PowerManager.WakeLock? = null
    private var retryDelayMs = BASE_RETRY_MS

    private val watchdogReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action == ACTION_SERVER_READY) return
            ensureServerRunning()
        }
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        val filter = IntentFilter(ACTION_SERVER_READY)
        registerReceiver(watchdogReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val modelPath = intent?.getStringExtra(EXTRA_MODEL_PATH)
        if (modelPath == null) {
            Log.e(TAG, "No model path provided")
            broadcastStatus(STATUS_ERROR)
            return START_NOT_STICKY
        }

        val port = intent.getIntExtra(EXTRA_PORT, DEFAULT_PORT)

        acquireWakelock()
        startForeground(NOTIFICATION_ID, buildNotification(STATUS_LOADING))

        if (serverReady) {
            Log.d(TAG, "Server already running")
            broadcastStatus(STATUS_READY)
            return START_STICKY
        }

        Thread({
            startServer(modelPath, port)
        }, "llama-server-thread").start()

        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        releaseWakelock()
        try {
            unregisterReceiver(watchdogReceiver)
        } catch (_: Exception) {}
        stopServer()
        super.onDestroy()
    }

    private fun startServer(modelPath: String, port: Int) {
        try {
            retryDelayMs = BASE_RETRY_MS

            val workspaceDir = prepareWorkspace()
            if (workspaceDir == null) {
                Log.e(TAG, "Failed to prepare workspace")
                broadcastStatus(STATUS_ERROR)
                return
            }

            val serverBinary = File(workspaceDir, "llama-server")
            if (!serverBinary.exists()) {
                Log.e(TAG, "llama-server binary not found at ${serverBinary.absolutePath}")
                broadcastStatus(STATUS_ERROR)
                return
            }
            serverBinary.setExecutable(true, false)

            val nativeLibDir = applicationInfo.nativeLibraryDir
            val ldPath = listOfNotNull(nativeLibDir, workspaceDir.absolutePath).joinToString(":")

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

            val processBuilder = ProcessBuilder(*cmd)
            processBuilder.directory(workspaceDir)
            processBuilder.environment()["LD_LIBRARY_PATH"] = ldPath
            processBuilder.redirectErrorStream(true)

            synchronized(lock) {
                if (serverProcess != null) {
                    Log.w(TAG, "Server already running, killing previous instance")
                    serverProcess?.destroy()
                }
                serverProcess = processBuilder.start()
                serverReady = false
            }

            // Non-blocking log reader in separate thread
            Thread({
                try {
                    val reader = serverProcess!!.inputStream.bufferedReader()
                    var line: String?
                    while (reader.readLine().also { line = it } != null) {
                        Log.d(TAG, "server: $line")
                        if (line!!.contains("HTTP server listening") || line!!.contains("starting the main loop")) {
                            serverReady = true
                            broadcastStatus(STATUS_READY)
                        }
                    }
                } catch (_: Exception) {}
            }, "llama-log-reader").start()

            val exitCode = try {
                serverProcess?.waitFor() ?: -1
            } catch (e: InterruptedException) {
                -1
            }
            Log.i(TAG, "Server exited with code $exitCode")
            serverReady = false

            // Auto-restart with exponential backoff
            if (exitCode != 0 && !Thread.currentThread().isInterrupted) {
                Log.i(TAG, "Restarting in ${retryDelayMs}ms")
                broadcastStatus(STATUS_LOADING)
                updateNotification(STATUS_LOADING)
                Thread.sleep(retryDelayMs)
                retryDelayMs = (retryDelayMs * 2).coerceAtMost(MAX_RETRY_MS)
                startServer(modelPath, port)
            } else {
                broadcastStatus(STATUS_ERROR)
            }

        } catch (e: InterruptedException) {
            Log.i(TAG, "Server thread interrupted")
        } catch (e: Exception) {
            Log.e(TAG, "Error starting server", e)
            broadcastStatus(STATUS_ERROR)
        }
    }

    private fun ensureServerRunning() {
        synchronized(lock) {
            if (serverProcess != null && serverProcess!!.isAlive) return
        }
        Log.w(TAG, "Watchdog detected dead server, restarting")
        // The service will be restarted via START_STICKY
    }

    private fun prepareWorkspace(): File? {
        return try {
            val workspaceDir = File(filesDir, "workspace")
            if (!workspaceDir.exists()) workspaceDir.mkdirs()

            val targetFile = File(workspaceDir, "llama-server")
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
        serverReady = false
    }

    private fun acquireWakelock() {
        if (wakelock == null) {
            val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
            wakelock = powerManager.newWakeLock(
                PowerManager.PARTIAL_WAKE_LOCK,
                "qwen35-ptbr:llama-server"
            )
            wakelock?.acquire(24 * 60 * 60 * 1000L)
        }
    }

    private fun releaseWakelock() {
        wakelock?.let {
            if (it.isHeld) it.release()
            wakelock = null
        }
    }

    private fun broadcastStatus(status: String) {
        val intent = Intent(ACTION_SERVER_READY).apply {
            putExtra(EXTRA_STATUS, status)
            setPackage(packageName)
        }
        sendBroadcast(intent)
        updateNotification(status)
    }

    private fun updateNotification(status: String) {
        val text = when (status) {
            STATUS_LOADING -> "Carregando modelo..."
            STATUS_READY -> "Pronto na porta $DEFAULT_PORT"
            STATUS_ERROR -> "Erro ao iniciar"
            else -> "Iniciando..."
        }
        val notification = buildNotification(status)
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        manager.notify(NOTIFICATION_ID, notification)
    }

    private fun buildNotification(status: String): Notification {
        val activityIntent = Intent(this, MainActivity::class.java)
        val activityPendingIntent = PendingIntent.getActivity(
            this, 0, activityIntent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        val title = when (status) {
            STATUS_READY -> getString(R.string.notification_running)
            else -> getString(R.string.server_starting)
        }
        val text = when (status) {
            STATUS_LOADING -> "Carregando modelo..."
            STATUS_READY -> "porta $DEFAULT_PORT"
            else -> "Aguardando..."
        }

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(title)
            .setContentText(text)
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
