package com.qwen35.ptbr.chat

import android.Manifest
import android.app.DownloadManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.PowerManager
import android.provider.Settings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.qwen35.ptbr.chat.ui.chat.ChatScreen
import kotlinx.coroutines.delay
import java.io.File

class MainActivity : ComponentActivity() {

    companion object {
        const val MODEL_FILENAME = "qwen35-ptbr-q4_k_m.gguf"
        const val LOCAL_PORT = 8080
        const val MODEL_DOWNLOAD_URL = "https://huggingface.co/DanielPonttes/qwen35-ptbr-mobile/resolve/main/qwen35-ptbr-q4_k_m.gguf"
        val MODEL_SEARCH_PATHS = listOf(
            Environment.getExternalStorageDirectory().absolutePath + "/models/" + MODEL_FILENAME,
            Environment.getExternalStorageDirectory().absolutePath + "/Download/" + MODEL_FILENAME,
            Environment.getExternalStorageDirectory().absolutePath + "/" + MODEL_FILENAME,
        )
        private const val HEALTH_POLL_INITIAL_MS = 500L
        private const val HEALTH_POLL_MAX_MS = 5_000L
    }

    private val notificationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { _ -> }

    private var downloadId: Long = -1L
    private var serverStatus by mutableStateOf<ServerStatus>(ServerStatus.Loading(0))
    private var modelPath by mutableStateOf<String?>(null)

    private val serverReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val status = intent?.getStringExtra(LlamaServerService.EXTRA_STATUS)
            serverStatus = when (status) {
                LlamaServerService.STATUS_READY -> ServerStatus.Running
                LlamaServerService.STATUS_ERROR -> ServerStatus.ConnectionError
                else -> ServerStatus.Loading(0)
            }
        }
    }

    private val batteryOptimizationLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED
            ) {
                notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
            }
        }

        requestBatteryOptimizationExemption()

        val filter = IntentFilter(LlamaServerService.ACTION_SERVER_READY)
        registerReceiver(serverReceiver, filter, Context.RECEIVER_NOT_EXPORTED)

        modelPath = findModelFile()

        setContent {
            MaterialTheme(
                colorScheme = lightColorScheme(
                    primary = Color(0xFF1A73E8),
                    secondary = Color(0xFF03DAC6),
                    background = Color.White,
                    surface = Color(0xFFF5F5F5),
                )
            ) {
                when {
                    modelPath == null -> ModelNotFoundScreen()
                    else -> MainChatScreen(modelPath!!)
                }
            }
        }
    }

    override fun onDestroy() {
        try {
            unregisterReceiver(serverReceiver)
        } catch (_: Exception) {}
        super.onDestroy()
    }

    override fun onStop() {
        super.onStop()
        // Do NOT stop the service here -- let it run in background
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        modelPath = findModelFile()
    }

    private fun findModelFile(): String? {
        for (dir in listOf(filesDir, getExternalFilesDir(null))) {
            if (dir == null) continue
            val f = File(dir, MODEL_FILENAME)
            if (f.exists()) return f.absolutePath
        }
        for (path in MODEL_SEARCH_PATHS) {
            val f = File(path)
            if (f.exists()) return f.absolutePath
        }
        return null
    }

    private fun requestBatteryOptimizationExemption() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
            if (!pm.isIgnoringBatteryOptimizations(packageName)) {
                val intent = Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS).apply {
                    data = Uri.parse("package:$packageName")
                }
                batteryOptimizationLauncher.launch(intent)
            }
        }
    }

    private fun startModelDownload() {
        val request = DownloadManager.Request(Uri.parse(MODEL_DOWNLOAD_URL)).apply {
            setTitle("Baixando modelo Qwen3.5 PT-BR")
            setDescription("505 MB — necessario para o chat funcionar offline")
            setDestinationInExternalFilesDir(
                this@MainActivity,
                null,
                MODEL_FILENAME
            )
            setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
            setAllowedOverMetered(true)
            setAllowedOverRoaming(false)
        }

        val dm = getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
        downloadId = dm.enqueue(request)
    }

    @Composable
    fun ModelNotFoundScreen() {
        var isDownloading by remember { mutableStateOf(false) }

        Surface(
            modifier = Modifier.fillMaxSize(),
            color = MaterialTheme.colorScheme.background
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(32.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                Icon(
                    imageVector = Icons.Default.Info,
                    contentDescription = null,
                    modifier = Modifier.size(64.dp),
                    tint = MaterialTheme.colorScheme.primary
                )
                Spacer(modifier = Modifier.height(24.dp))
                Text(
                    text = "Modelo nao encontrado",
                    style = MaterialTheme.typography.headlineMedium,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = "O modelo GGUF (505 MB) precisa ser baixado uma vez para usar o chat offline.\n\n" +
                           "Sera salvo em: ${getExternalFilesDir(null)}/$MODEL_FILENAME",
                    style = MaterialTheme.typography.bodyMedium,
                    textAlign = TextAlign.Center,
                    color = MaterialTheme.colorScheme.onSurface,
                    lineHeight = 24.sp
                )
                Spacer(modifier = Modifier.height(32.dp))
                Button(
                    onClick = {
                        startModelDownload()
                        isDownloading = true
                    },
                    enabled = !isDownloading
                ) {
                    Icon(
                        imageVector = Icons.Default.Download,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(if (isDownloading) "Baixando..." else "Baixar modelo (505 MB)")
                }
                if (isDownloading) {
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(
                        text = "O download esta em andamento. Feche e reabra o app quando terminar.",
                        style = MaterialTheme.typography.bodySmall,
                        textAlign = TextAlign.Center,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                    )
                }
            }
        }
    }

    @Composable
    fun MainChatScreen(modelPath: String) {
        var useWebView by remember { mutableStateOf(true) }
        var showStopDialog by remember { mutableStateOf(false) }

        LaunchedEffect(modelPath) {
            val intent = Intent(this@MainActivity, LlamaServerService::class.java).apply {
                putExtra(LlamaServerService.EXTRA_MODEL_PATH, modelPath)
                putExtra(LlamaServerService.EXTRA_PORT, LOCAL_PORT)
            }
            ContextCompat.startForegroundService(this@MainActivity, intent)

            // Poll health endpoint with exponential backoff
            var delayMs = HEALTH_POLL_INITIAL_MS
            var attempts = 0
            while (serverStatus !is ServerStatus.Running && attempts < 30) {
                delay(delayMs)
                try {
                    val url = java.net.URL("http://127.0.0.1:$LOCAL_PORT/health")
                    val conn = url.openConnection() as java.net.HttpURLConnection
                    conn.connectTimeout = 1000
                    conn.readTimeout = 1000
                    if (conn.responseCode == 200) {
                        serverStatus = ServerStatus.Running
                        break
                    }
                } catch (_: Exception) {}
                attempts++
                serverStatus = ServerStatus.Loading(attempts)
                delayMs = (delayMs * 1.5).toLong().coerceAtMost(HEALTH_POLL_MAX_MS)
            }

            if (serverStatus !is ServerStatus.Running) {
                serverStatus = ServerStatus.ConnectionError
            }
        }

        if (showStopDialog) {
            AlertDialog(
                onDismissRequest = { showStopDialog = false },
                title = { Text("Parar servidor?") },
                text = { Text("O servidor sera parado e o chat ficara indisponivel.") },
                confirmButton = {
                    TextButton(onClick = {
                        showStopDialog = false
                        val intent = Intent(this@MainActivity, LlamaServerService::class.java)
                        stopService(intent)
                        finish()
                    }) { Text("Parar") }
                },
                dismissButton = {
                    TextButton(onClick = { showStopDialog = false }) { Text("Cancelar") }
                }
            )
        }

        Column(modifier = Modifier.fillMaxSize()) {
            when (val state = serverStatus) {
                is ServerStatus.Loading -> {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            CircularProgressIndicator()
                            Spacer(modifier = Modifier.height(16.dp))
                            Text(
                                text = "Iniciando servidor... (${state.attempts}/30)",
                                style = MaterialTheme.typography.bodyLarge
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = "Carregando modelo de 505 MB",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f)
                            )
                        }
                    }
                }
                is ServerStatus.ConnectionError -> {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Text(
                                text = "Servidor nao iniciou",
                                style = MaterialTheme.typography.headlineSmall,
                                color = MaterialTheme.colorScheme.error
                            )
                            Spacer(modifier = Modifier.height(16.dp))
                            Button(onClick = {
                                serverStatus = ServerStatus.Loading(0)
                                val intent = Intent(this@MainActivity, LlamaServerService::class.java).apply {
                                    putExtra(LlamaServerService.EXTRA_MODEL_PATH, modelPath)
                                    putExtra(LlamaServerService.EXTRA_PORT, LOCAL_PORT)
                                }
                                ContextCompat.startForegroundService(this@MainActivity, intent)
                            }) { Text("Tentar novamente") }
                        }
                    }
                }
                is ServerStatus.Running -> {
                    if (useWebView) {
                        AndroidView(
                            factory = { ctx ->
                                WebView(ctx).apply {
                                    settings.javaScriptEnabled = true
                                    settings.domStorageEnabled = true
                                    webViewClient = WebViewClient()
                                    loadUrl("http://127.0.0.1:$LOCAL_PORT")
                                }
                            },
                            modifier = Modifier.weight(1f)
                        )
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(8.dp),
                            horizontalArrangement = Arrangement.SpaceEvenly
                        ) {
                            TextButton(onClick = { useWebView = false }) {
                                Text("Chat nativo")
                            }
                            TextButton(onClick = { showStopDialog = true }) {
                                Text("Parar servidor", color = MaterialTheme.colorScheme.error)
                            }
                        }
                    } else {
                        ChatScreen(
                            port = LOCAL_PORT,
                            modifier = Modifier.weight(1f)
                        )
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(8.dp),
                            horizontalArrangement = Arrangement.SpaceEvenly
                        ) {
                            TextButton(onClick = { useWebView = true }) {
                                Text("WebView")
                            }
                            TextButton(onClick = { showStopDialog = true }) {
                                Text("Parar servidor", color = MaterialTheme.colorScheme.error)
                            }
                        }
                    }
                }
            }
        }
    }

    sealed class ServerStatus {
        data class Loading(val attempts: Int) : ServerStatus()
        data object Running : ServerStatus()
        data object ConnectionError : ServerStatus()
    }
}
