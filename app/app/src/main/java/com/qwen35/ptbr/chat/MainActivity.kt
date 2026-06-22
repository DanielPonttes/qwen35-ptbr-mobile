package com.qwen35.ptbr.chat

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
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
import java.io.File

class MainActivity : ComponentActivity() {

    companion object {
        const val MODEL_FILENAME = "qwen35-ptbr-q4_k_m.gguf"
        const val LOCAL_PORT = 8080
        val MODEL_SEARCH_PATHS = listOf(
            Environment.getExternalStorageDirectory().absolutePath + "/models/" + MODEL_FILENAME,
            Environment.getExternalStorageDirectory().absolutePath + "/Download/" + MODEL_FILENAME,
            Environment.getExternalStorageDirectory().absolutePath + "/" + MODEL_FILENAME,
        )
    }

    private val notificationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { _ -> }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED
            ) {
                notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
            }
        }

        val modelPath = findModelFile()

        setContent {
            MaterialTheme(
                colorScheme = lightColorScheme(
                    primary = Color(0xFF1A73E8),
                    secondary = Color(0xFF03DAC6),
                    background = Color.White,
                    surface = Color(0xFFF5F5F5),
                )
            ) {
                if (modelPath == null) {
                    ModelNotFoundScreen()
                } else {
                    MainChatScreen(modelPath)
                }
            }
        }
    }

    private fun findModelFile(): String? {
        val internalFile = File(filesDir, MODEL_FILENAME)
        if (internalFile.exists()) return internalFile.absolutePath

        val externalFile = File(getExternalFilesDir(null), MODEL_FILENAME)
        if (externalFile.exists()) return externalFile.absolutePath

        for (path in MODEL_SEARCH_PATHS) {
            val f = File(path)
            if (f.exists()) return f.absolutePath
        }

        return null
    }

    @Composable
    fun ModelNotFoundScreen() {
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
                    text = getString(R.string.model_not_found_title),
                    style = MaterialTheme.typography.headlineMedium,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = getString(
                        R.string.model_not_found_message,
                        "${getExternalFilesDir(null)}/$MODEL_FILENAME"
                    ),
                    style = MaterialTheme.typography.bodyMedium,
                    textAlign = TextAlign.Center,
                    color = MaterialTheme.colorScheme.onSurface,
                    lineHeight = 24.sp
                )
                Spacer(modifier = Modifier.height(32.dp))
                Button(
                    onClick = {
                        val browserIntent = Intent(
                            Intent.ACTION_VIEW,
                            Uri.parse("https://huggingface.co/models?search=qwen35-ptbr")
                        )
                        startActivity(browserIntent)
                    }
                ) {
                    Text(getString(R.string.download_model))
                }
            }
        }
    }

    @Composable
    fun MainChatScreen(modelPath: String) {
        var serverState by remember { mutableStateOf<ServerState>(ServerState.Starting) }
        var useWebView by remember { mutableStateOf(true) }

        LaunchedEffect(modelPath) {
            val intent = Intent(this@MainActivity, LlamaServerService::class.java).apply {
                putExtra(LlamaServerService.EXTRA_MODEL_PATH, modelPath)
                putExtra(LlamaServerService.EXTRA_PORT, LOCAL_PORT)
            }
            ContextCompat.startForegroundService(this@MainActivity, intent)
            serverState = ServerState.Running
        }

        Column(modifier = Modifier.fillMaxSize()) {
            when {
                serverState == ServerState.Starting -> {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            CircularProgressIndicator()
                            Spacer(modifier = Modifier.height(16.dp))
                            Text(
                                text = getString(R.string.server_starting),
                                style = MaterialTheme.typography.bodyLarge
                            )
                        }
                    }
                }
                useWebView -> {
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
                        horizontalArrangement = Arrangement.Center
                    ) {
                        TextButton(onClick = { useWebView = false }) {
                            Text("Mudar para Chat UI")
                        }
                    }
                }
                else -> {
                    ChatScreen(
                        port = LOCAL_PORT,
                        modifier = Modifier.weight(1f)
                    )
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(8.dp),
                        horizontalArrangement = Arrangement.Center
                    ) {
                        TextButton(onClick = { useWebView = true }) {
                            Text("Mudar para WebView")
                        }
                    }
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        val intent = Intent(this, LlamaServerService::class.java)
        stopService(intent)
    }

    enum class ServerState {
        Starting, Running, Error
    }
}
