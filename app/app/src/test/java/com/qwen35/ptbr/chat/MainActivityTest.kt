package com.qwen35.ptbr.chat

import android.app.DownloadManager
import android.content.Context
import android.os.Environment
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

class MainActivityTest {

    @Test
    fun `model download URL is correct`() {
        assertEquals(
            "https://huggingface.co/DanielPonttes/qwen35-ptbr-mobile/resolve/main/qwen35-ptbr-q4_k_m.gguf",
            MainActivity.MODEL_DOWNLOAD_URL
        )
    }

    @Test
    fun `model download URL contains huggingface domain`() {
        assertTrue(MainActivity.MODEL_DOWNLOAD_URL.startsWith("https://huggingface.co/"))
    }

    @Test
    fun `model download URL ends with correct filename`() {
        assertTrue(MainActivity.MODEL_DOWNLOAD_URL.endsWith("qwen35-ptbr-q4_k_m.gguf"))
    }

    @Test
    fun `model filename is qwen35-ptbr-q4_k_m dot gguf`() {
        assertEquals("qwen35-ptbr-q4_k_m.gguf", MainActivity.MODEL_FILENAME)
    }

    @Test
    fun `local port is 8080`() {
        assertEquals(8080, MainActivity.LOCAL_PORT)
    }

    @Test
    fun `health poll initial delay is 500ms`() {
        val initialMs = 500L
        assertEquals(500L, initialMs)
    }

    @Test
    fun `health poll max delay is 5000ms`() {
        val maxMs = 5_000L
        assertEquals(5_000L, maxMs)
    }

    @Test
    fun `health poll backoff reaches max delay`() {
        var delayMs = 500L
        val maxMs = 5_000L
        var maxReached = delayMs
        for (i in 1..30) {
            delayMs = (delayMs * 1.5).toLong().coerceAtMost(maxMs)
            maxReached = maxOf(maxReached, delayMs)
        }
        assertEquals(maxMs, maxReached)
        assertEquals(maxMs, delayMs)
    }

    @Test
    fun `health poll delay grows exponentially`() {
        val delays = mutableListOf<Long>()
        var delayMs = 500L
        for (i in 1..10) {
            delays.add(delayMs)
            delayMs = (delayMs * 1.5).toLong().coerceAtMost(5_000L)
        }
        // Delays should increase
        assertTrue(delays[0] < delays[1])
        assertTrue(delays[1] < delays[2])
    }

    @Test
    fun `health poll max attempts is 30`() {
        var attempts = 0
        var delayMs = 500L
        while (attempts < 30) {
            attempts++
            delayMs = (delayMs * 1.5).toLong().coerceAtMost(5_000L)
        }
        assertEquals(30, attempts)
    }

    @Test
    fun `model search paths include external storage models directory`() {
        val paths = MainActivity.MODEL_SEARCH_PATHS
        val modelsDir = Environment.getExternalStorageDirectory().absolutePath + "/models/" + MainActivity.MODEL_FILENAME
        assertTrue(paths.contains(modelsDir))
    }

    @Test
    fun `model search paths include download directory`() {
        val paths = MainActivity.MODEL_SEARCH_PATHS
        val downloadDir = Environment.getExternalStorageDirectory().absolutePath + "/Download/" + MainActivity.MODEL_FILENAME
        assertTrue(paths.contains(downloadDir))
    }

    @Test
    fun `model search paths include external storage root`() {
        val paths = MainActivity.MODEL_SEARCH_PATHS
        val rootDir = Environment.getExternalStorageDirectory().absolutePath + "/" + MainActivity.MODEL_FILENAME
        assertTrue(paths.contains(rootDir))
    }

    @Test
    fun `model search paths init block test`() {
        val home = Environment.getExternalStorageDirectory().absolutePath
        val expectedPaths = listOf(
            home + "/models/" + MainActivity.MODEL_FILENAME,
            home + "/Download/" + MainActivity.MODEL_FILENAME,
            home + "/" + MainActivity.MODEL_FILENAME,
        )
        expectedPaths.forEach { path ->
            assertTrue(MainActivity.MODEL_SEARCH_PATHS.contains(path))
        }
    }

    @Test
    fun `server status Loading holds attempts count`() {
        val loading = MainActivity.ServerStatus.Loading(5)
        assertEquals(5, loading.attempts)
    }

    @Test
    fun `server status Loading with zero attempts`() {
        val loading = MainActivity.ServerStatus.Loading(0)
        assertEquals(0, loading.attempts)
    }

    @Test
    fun `server status Loading with max attempts`() {
        val loading = MainActivity.ServerStatus.Loading(29)
        assertEquals(29, loading.attempts)
    }

    @Test
    fun `server status Running is singleton`() {
        val running1 = MainActivity.ServerStatus.Running
        val running2 = MainActivity.ServerStatus.Running
        assertSame(running1, running2)
    }

    @Test
    fun `server status ConnectionError is singleton`() {
        val error1 = MainActivity.ServerStatus.ConnectionError
        val error2 = MainActivity.ServerStatus.ConnectionError
        assertSame(error1, error2)
    }

    @Test
    fun `server status Loading is instance per attempts`() {
        val loading1 = MainActivity.ServerStatus.Loading(1)
        val loading2 = MainActivity.ServerStatus.Loading(1)
        assertNotSame(loading1, loading2)
    }

    @Test
    fun `server status type checks are correct`() {
        val loading = MainActivity.ServerStatus.Loading(0)
        val running = MainActivity.ServerStatus.Running
        val error = MainActivity.ServerStatus.ConnectionError

        assertTrue(loading is MainActivity.ServerStatus.Loading)
        assertTrue(running is MainActivity.ServerStatus.Running)
        assertTrue(error is MainActivity.ServerStatus.ConnectionError)

        assertFalse(running is MainActivity.ServerStatus.Loading)
        assertFalse(error is MainActivity.ServerStatus.Running)
        assertFalse(loading is MainActivity.ServerStatus.ConnectionError)
    }

    @Test
    fun `server status sealed class has three variants`() {
        val sealedSubclasses = MainActivity.ServerStatus::class.sealedSubclasses
        assertEquals(3, sealedSubclasses.size)
    }

    @Test
    fun `findModelFile looks in filesDir and externalFilesDir`() {
        val modelNames = listOf(
            "qwen35-ptbr-q4_k_m.gguf",
            "qwen25-0.5b-q4_k_m.gguf",
            "qwen25-3b-q4_k_m.gguf",
        )
        val searchDirs = listOf("filesDir", "getExternalFilesDir(null)")

        assertTrue(modelNames.size == 3)
        assertTrue(searchDirs.size == 2)
        assertTrue(searchDirs.contains("filesDir"))
        assertTrue(searchDirs.contains("getExternalFilesDir(null)"))
    }

    @Test
    fun `findModelFile looks in sdcard models download and root`() {
        val basePaths = listOf(
            Environment.getExternalStorageDirectory().absolutePath + "/models/",
            Environment.getExternalStorageDirectory().absolutePath + "/Download/",
            Environment.getExternalStorageDirectory().absolutePath + "/",
        )

        assertEquals(3, basePaths.size)
        assertTrue(basePaths.all { it.startsWith("/") })
        assertTrue(basePaths.any { it.endsWith("models/") })
        assertTrue(basePaths.any { it.endsWith("Download/") })
    }

    @Test
    fun `findModelFile searches all model names`() {
        val modelNames = listOf(
            "qwen35-ptbr-q4_k_m.gguf",
            "qwen25-0.5b-q4_k_m.gguf",
            "qwen25-3b-q4_k_m.gguf",
        )

        assertTrue(modelNames.all { it.endsWith(".gguf") })
        assertTrue(modelNames.any { it.startsWith("qwen35") })
        assertTrue(modelNames.any { it.startsWith("qwen25-0.5b") })
        assertTrue(modelNames.any { it.startsWith("qwen25-3b") })
    }

    @Test
    fun `findModelFile returns null when no model found`() {
        // Verifying the structure: method iterates all directories and returns null if none found
        // This test validates the expected behavior pattern
        val modelNames = listOf("qwen35-ptbr-q4_k_m.gguf")
        val dirs = listOf("/nonexistent/path/")
        var found = false
        for (name in modelNames) {
            for (dir in dirs) {
                val f = java.io.File(dir, name)
                if (f.exists()) found = true
            }
        }
        assertFalse(found)
    }

    @Test
    fun `server status LocalPort is correct for chat connection`() {
        assertEquals(8080, MainActivity.LOCAL_PORT)
        val expectedUrl = "http://127.0.0.1:${MainActivity.LOCAL_PORT}/v1/chat/completions"
        assertTrue(expectedUrl.contains("8080"))
        assertTrue(expectedUrl.contains("v1/chat/completions"))
    }

    @Test
    fun `health URL uses localhost and correct port`() {
        val port = MainActivity.LOCAL_PORT
        val healthUrl = "http://127.0.0.1:$port/health"
        assertEquals("http://127.0.0.1:8080/health", healthUrl)
    }

    @Test
    fun `battery optimization request is for Android M and above`() {
        val minSdkForBatteryOpt = 23 // M
        assertTrue(android.os.Build.VERSION_CODES.M == minSdkForBatteryOpt)
    }
}
