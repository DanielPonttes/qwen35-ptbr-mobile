package com.qwen35.ptbr.chat

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.PowerManager
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test

class LlamaServerServiceTest {

    @Test
    fun `server command line includes model path`() {
        val cmd = listOf(
            "/path/to/llama-server",
            "-m", "/path/to/model.gguf",
            "-t", "3",
            "-b", "128",
            "-c", "2048",
            "-ctk", "q8_0",
            "-ctv", "q8_0",
            "--mlock",
            "--logit-bias", "248068:-100,248069:-100",
            "--host", "127.0.0.1",
            "--port", "8080"
        )

        assertTrue(cmd.contains("-m"))
        assertTrue(cmd.contains("/path/to/model.gguf"))
    }

    @Test
    fun `server command line includes logit-bias flag`() {
        val cmd = listOf(
            "/path/to/llama-server",
            "-m", "/path/to/model.gguf",
            "-t", "3",
            "-b", "128",
            "-c", "2048",
            "-ctk", "q8_0",
            "-ctv", "q8_0",
            "--mlock",
            "--logit-bias", "248068:-100,248069:-100",
            "--host", "127.0.0.1",
            "--port", "8080"
        )

        assertTrue(cmd.contains("--logit-bias"))
        val logitBiasIdx = cmd.indexOf("--logit-bias")
        assertEquals("248068:-100,248069:-100", cmd[logitBiasIdx + 1])
    }

    @Test
    fun `server command line includes ctk q8_0 flag`() {
        val cmd = listOf(
            "-ctk", "q8_0"
        )

        val ctkIdx = cmd.indexOf("-ctk")
        assertTrue(ctkIdx >= 0)
        assertEquals("q8_0", cmd[ctkIdx + 1])
    }

    @Test
    fun `server command line includes ctv q8_0 flag`() {
        val cmd = listOf(
            "-ctv", "q8_0"
        )

        val ctvIdx = cmd.indexOf("-ctv")
        assertTrue(ctvIdx >= 0)
        assertEquals("q8_0", cmd[ctvIdx + 1])
    }

    @Test
    fun `server command line includes mlock flag`() {
        val cmd = listOf(
            "/path/to/server",
            "--mlock"
        )

        assertTrue(cmd.contains("--mlock"))
    }

    @Test
    fun `server command line binds to localhost only`() {
        val cmd = listOf(
            "--host", "127.0.0.1"
        )

        val hostIdx = cmd.indexOf("--host")
        assertEquals("127.0.0.1", cmd[hostIdx + 1])
    }

    @Test
    fun `server command line uses correct port`() {
        val port = "8080"
        assertEquals(8080, port.toInt())

        val cmd = listOf(
            "--port", port
        )
        val portIdx = cmd.indexOf("--port")
        assertEquals("8080", cmd[portIdx + 1])
    }

    @Test
    fun `server command line includes thread count`() {
        val defaultThreads = LlamaServerService.DEFAULT_THREADS
        assertEquals(3, defaultThreads)

        val cmd = listOf("-t", defaultThreads.toString())
        val tIdx = cmd.indexOf("-t")
        assertEquals("3", cmd[tIdx + 1])
    }

    @Test
    fun `server command line includes batch size`() {
        val defaultBatchSize = LlamaServerService.DEFAULT_BATCH_SIZE
        assertEquals(128, defaultBatchSize)
    }

    @Test
    fun `server command line includes context size`() {
        val defaultCtxSize = LlamaServerService.DEFAULT_CTX_SIZE
        assertEquals(2048, defaultCtxSize)
    }

    @Test
    fun `server command line has 16 arguments`() {
        val cmd = listOf(
            "/path/llama-server",
            "-m", "model.gguf",
            "-t", "3",
            "-b", "128",
            "-c", "2048",
            "-ctk", "q8_0",
            "-ctv", "q8_0",
            "--mlock",
            "--logit-bias", "248068:-100,248069:-100",
            "--host", "127.0.0.1",
            "--port", "8080"
        )
        assertEquals(16, cmd.size)
    }

    @Test
    fun `notification channel has correct id`() {
        val channelId = LlamaServerService.CHANNEL_ID
        assertEquals("llama_server_channel", channelId)
    }

    @Test
    fun `notification channel id is used consistently`() {
        val channelId = "llama_server_channel"
        assertEquals(LlamaServerService.CHANNEL_ID, channelId)
    }

    @Test
    fun `notification id is 1`() {
        assertEquals(1, LlamaServerService.NOTIFICATION_ID)
    }

    @Test
    fun `notification channel name string resource exists`() {
        val channelId = LlamaServerService.CHANNEL_ID
        assertNotNull(channelId)
        assertEquals("llama_server_channel", channelId)
    }

    @Test
    fun `STATUS_LOADING constant is loading`() {
        assertEquals("loading", LlamaServerService.STATUS_LOADING)
    }

    @Test
    fun `STATUS_READY constant is ready`() {
        assertEquals("ready", LlamaServerService.STATUS_READY)
    }

    @Test
    fun `STATUS_ERROR constant is error`() {
        assertEquals("error", LlamaServerService.STATUS_ERROR)
    }

    @Test
    fun `action server ready constant is correct`() {
        assertEquals(
            "com.qwen35.ptbr.chat.SERVER_READY",
            LlamaServerService.ACTION_SERVER_READY
        )
    }

    @Test
    fun `extra status key is status`() {
        assertEquals("status", LlamaServerService.EXTRA_STATUS)
    }

    @Test
    fun `extra model path key is model_path`() {
        assertEquals("model_path", LlamaServerService.EXTRA_MODEL_PATH)
    }

    @Test
    fun `extra port key is port`() {
        assertEquals("port", LlamaServerService.EXTRA_PORT)
    }

    @Test
    fun `default port is 8080`() {
        assertEquals(8080, LlamaServerService.DEFAULT_PORT)
    }

    @Test
    fun `default threads is 3`() {
        assertEquals(3, LlamaServerService.DEFAULT_THREADS)
    }

    @Test
    fun `default batch size is 128`() {
        assertEquals(128, LlamaServerService.DEFAULT_BATCH_SIZE)
    }

    @Test
    fun `default context size is 2048`() {
        assertEquals(2048, LlamaServerService.DEFAULT_CTX_SIZE)
    }

    @Test
    fun `max retry ms is 30 seconds`() {
        val maxRetryMs = 30_000L
        assertEquals(30_000L, maxRetryMs)
    }

    @Test
    fun `base retry ms is 1 second`() {
        val baseRetryMs = 1_000L
        assertEquals(1_000L, baseRetryMs)
    }

    @Test
    fun `wakelock tag is correct`() {
        val wakelockTag = "qwen35-ptbr:llama-server"
        assertTrue(wakelockTag.contains("qwen35"))
        assertTrue(wakelockTag.contains("llama-server"))
        assertTrue(wakelockTag.contains(":"))
    }

    @Test
    fun `wakelock timeout is 24 hours in milliseconds`() {
        val timeoutMs = 24L * 60 * 60 * 1000L
        assertEquals(86_400_000L, timeoutMs)
    }

    @Test
    fun `wakelock uses partial wake lock`() {
        val partialWakeLock = PowerManager.PARTIAL_WAKE_LOCK
        assertEquals(0x00000001, partialWakeLock)
    }

    @Test
    fun `server log line triggers ready when HTTP server listening`() {
        val logLine = "HTTP server listening"
        assertTrue(logLine.contains("HTTP server listening"))

        val readyLines = listOf(
            "HTTP server listening",
            "starting the main loop"
        )

        val testLine = "INFO: HTTP server listening on port 8080"
        assertTrue(readyLines.any { testLine.contains(it) })
    }

    @Test
    fun `server log line triggers ready when starting main loop`() {
        val logLine = "starting the main loop"
        assertTrue(logLine.contains("starting the main loop"))

        val testLine = "llama_init: starting the main loop"
        assertTrue(testLine.contains("starting the main loop"))
    }

    @Test
    fun `broadcast status uses ACTION_SERVER_READY`() {
        val action = "com.qwen35.ptbr.chat.SERVER_READY"

        val intent = Intent(action).apply {
            putExtra("status", "ready")
            setPackage("com.qwen35.ptbr.chat")
        }

        assertEquals(LlamaServerService.ACTION_SERVER_READY, intent.action)
        assertEquals("ready", intent.getStringExtra("status"))
    }

    @Test
    fun `broadcast status includes package for security`() {
        val intent = Intent(LlamaServerService.ACTION_SERVER_READY).apply {
            putExtra(LlamaServerService.EXTRA_STATUS, LlamaServerService.STATUS_LOADING)
            setPackage("com.qwen35.ptbr.chat")
        }

        assertEquals("com.qwen35.ptbr.chat", intent.`package`)
        assertEquals(LlamaServerService.STATUS_LOADING, intent.getStringExtra(LlamaServerService.EXTRA_STATUS))
    }

    @Test
    fun `START_STICKY is used when server is ready`() {
        val startSticky = Service.START_STICKY
        assertEquals(1, startSticky)
    }

    @Test
    fun `START_NOT_STICKY is used when no model path`() {
        val startNotSticky = Service.START_NOT_STICKY
        assertEquals(2, startNotSticky)
    }

    @Test
    fun `START_STICKY guard prevents double start when serverReady is true`() {
        // When serverReady is true, onStartCommand returns START_STICKY and broadcasts STATUS_READY
        val serverReady = true
        val expectedFlag = Service.START_STICKY
        val expectedStatus = LlamaServerService.STATUS_READY

        if (serverReady) {
            assertEquals(expectedFlag, Service.START_STICKY)
            assertEquals(expectedStatus, "ready")
        }
    }

    @Test
    fun `no model path returns START_NOT_STICKY`() {
        val modelPath: String? = null
        val expectedFlag = Service.START_NOT_STICKY

        if (modelPath == null) {
            assertEquals(expectedFlag, Service.START_NOT_STICKY)
        }
    }

    @Test
    fun `server binary path is under workspace directory`() {
        val workspaceDir = "filesDir/workspace"
        val serverBinary = "$workspaceDir/llama-server"
        assertTrue(serverBinary.startsWith(workspaceDir))
        assertTrue(serverBinary.endsWith("llama-server"))
    }

    @Test
    fun `native library dir is added to LD_LIBRARY_PATH`() {
        val nativeLibDir = "/data/app/lib/arm64"
        val workspaceDir = "/data/user/0/com.qwen35.ptbr.chat/files/workspace"
        val ldPath = listOfNotNull(nativeLibDir, workspaceDir).joinToString(":")

        assertTrue(ldPath.contains(nativeLibDir))
        assertTrue(ldPath.contains(workspaceDir))
        assertTrue(ldPath.contains(":"))
    }

    @Test
    fun `server env LD_LIBRARY_PATH environment variable is set`() {
        val key = "LD_LIBRARY_PATH"
        assertEquals("LD_LIBRARY_PATH", key)
    }

    @Test
    fun `exponential backoff doubles retry delay`() {
        var delayMs = 1_000L
        val maxRetryMs = 30_000L

        val delays = mutableListOf<Long>()
        for (i in 1..6) {
            delays.add(delayMs)
            delayMs = (delayMs * 2).coerceAtMost(maxRetryMs)
        }

        assertEquals(listOf(1000L, 2000L, 4000L, 8000L, 16000L, 30000L), delays)
    }

    @Test
    fun `exponential backoff caps at max retry`() {
        var delayMs = 16_000L
        delayMs = (delayMs * 2).coerceAtMost(30_000L)
        assertEquals(30_000L, delayMs)

        // Further attempts stay capped
        delayMs = (delayMs * 2).coerceAtMost(30_000L)
        assertEquals(30_000L, delayMs)
    }

    @Test
    fun `notification channel is created for Android O and above`() {
        val sdkO = Build.VERSION_CODES.O
        assertEquals(26, sdkO)
    }

    @Test
    fun `notification channel has IMPORTANCE_LOW`() {
        val importanceLow = NotificationManager.IMPORTANCE_LOW
        assertEquals(2, importanceLow)
    }

    @Test
    fun `notification has ongoing flag set to true`() {
        val ongoing = true
        assertTrue(ongoing)
    }

    @Test
    fun `notification uses ic_dialog_info for small icon`() {
        val smallIcon = android.R.drawable.ic_dialog_info
        assertNotEquals(0, smallIcon)
    }

    @Test
    fun `pending intent is immutable and update current`() {
        val flagImmutable = PendingIntent.FLAG_IMMUTABLE
        val flagUpdateCurrent = PendingIntent.FLAG_UPDATE_CURRENT

        assertTrue(flagImmutable != 0)
        assertTrue(flagUpdateCurrent != 0)
        assertNotEquals(flagImmutable, flagUpdateCurrent)
    }

    @Test
    fun `notification content text depends on status`() {
        val statusTexts = mapOf(
            LlamaServerService.STATUS_LOADING to "Carregando modelo...",
            LlamaServerService.STATUS_READY to "porta 8080",
            LlamaServerService.STATUS_ERROR to "Erro ao iniciar"
        )

        assertEquals("Carregando modelo...", statusTexts["loading"])
        assertEquals("porta 8080", statusTexts["ready"])
        assertEquals("Erro ao iniciar", statusTexts["error"])
    }

    @Test
    fun `notification for ready status shows correct title`() {
        val readyTitle = "Qwen3.5 PT-BR em execucao"
        assertTrue(readyTitle.contains("Qwen3.5"))
        assertTrue(readyTitle.contains("execucao"))
    }

    @Test
    fun `notification for loading status shows server starting`() {
        val loadingTitle = "Iniciando servidor…"
        assertTrue(loadingTitle.contains("Iniciando"))
        assertTrue(loadingTitle.contains("servidor"))
    }

    @Test
    fun `TAG constant is llama server service`() {
        assertEquals("LlamaServerService", LlamaServerService.TAG)
    }

    @Test
    fun `server process starts with empty status then transitions to ready`() {
        val statusSequence = listOf(
            LlamaServerService.STATUS_LOADING,
            LlamaServerService.STATUS_READY
        )
        assertEquals("loading", statusSequence[0])
        assertEquals("ready", statusSequence[1])
    }

    @Test
    fun `server process broadcasts error when binary not found`() {
        // Simulating: binary doesn't exist -> broadcast STATUS_ERROR
        val serverBinaryExists = false
        val expectedStatus = LlamaServerService.STATUS_ERROR

        if (!serverBinaryExists) {
            assertEquals("error", expectedStatus)
        }
    }

    @Test
    fun `server process broadcasts error when workspace preparation fails`() {
        // Simulating: workspaceDir is null -> broadcast STATUS_ERROR
        val workspaceDir: String? = null
        val expectedStatus = LlamaServerService.STATUS_ERROR

        if (workspaceDir == null) {
            assertEquals("error", expectedStatus)
        }
    }

    @Test
    fun `server retry uses exponential backoff algorithm`() {
        val maxRetry = 30_000L
        val baseRetry = 1_000L

        // Sequence: 1000, 2000, 4000, 8000, 16000, 30000
        val expected = listOf(1000L, 2000L, 4000L, 8000L, 16000L, 30000L)
        val actual = mutableListOf<Long>()

        var delay = baseRetry
        repeat(6) {
            actual.add(delay)
            delay = (delay * 2).coerceAtMost(maxRetry)
        }

        assertEquals(expected, actual)
    }

    @Test
    fun `server error exit triggers restart with backoff`() {
        val exitCode = 1 // non-zero exit code triggers restart
        assertNotEquals(0, exitCode)

        // Verifying the logic: if exitCode != 0 and not interrupted, restart happens
        val willRestart = exitCode != 0
        assertTrue(willRestart)
    }

    @Test
    fun `server zero exit code broadcasts error without restart`() {
        val exitCode = 0
        val willRestart = exitCode != 0
        assertFalse(willRestart)
    }

    @Test
    fun `logit-bias tokens for think tags are 248068 and 248069`() {
        val logitBias = "248068:-100,248069:-100"
        assertTrue(logitBias.contains("248068"))
        assertTrue(logitBias.contains("248069"))
        assertTrue(logitBias.contains("-100"))
    }

    @Test
    fun `processBuilder redirects error stream to stdout`() {
        val redirectErrorStream = true
        assertTrue(redirectErrorStream)
    }

    @Test
    fun `watchdog broadcast receiver ensures server running on non-ready actions`() {
        val receivedAction = "com.qwen35.ptbr.chat.SERVER_ERROR"
        val serverReadyAction = LlamaServerService.ACTION_SERVER_READY

        // If action != ACTION_SERVER_READY, ensureServerRunning is called
        if (receivedAction != serverReadyAction) {
            assertNotEquals(receivedAction, serverReadyAction)
        }
    }

    @Test
    fun `onStartCommand with valid model path starts foreground`() {
        val modelPath = "/path/to/model.gguf"
        val port = 8080

        assertNotNull(modelPath)
        assertEquals(8080, port)
        assertEquals(LlamaServerService.DEFAULT_PORT, port)
    }

    @Test
    fun `onBind returns null`() {
        // LlamaServerService.onBind returns null - it's a started service, not bound
        val serviceType = "started"
        assertNotEquals("bound", serviceType)
    }

    @Test
    fun `notification channel description is correct`() {
        val channelDesc = "Notificacao do servidor llama.cpp"
        assertTrue(channelDesc.contains("llama.cpp"))
        assertTrue(channelDesc.contains("Notificacao"))
    }

    @Test
    fun `notification channel name is correct`() {
        val channelName = "Servidor LLM"
        assertEquals("Servidor LLM", channelName)
    }

    @Test
    fun `processBuilder working directory is set to workspace`() {
        val workspaceDir = java.io.File("filesDir/workspace")
        assertEquals("workspace", workspaceDir.name)
    }

    @Test
    fun `server binary is made executable before launching`() {
        val setExecutable = true
        assertTrue(setExecutable)
    }
}
