package com.qwen35.ptbr.chat.ui.chat

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.sse.EventSource
import okhttp3.sse.EventSourceListener
import okhttp3.sse.EventSources
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

data class ChatMessage(
    val id: String = java.util.UUID.randomUUID().toString(),
    val role: String,
    val content: String,
    val isStreaming: Boolean = false
)

// Approximate token count: ~4 chars per token for PT-BR
private const val CHARS_PER_TOKEN = 4
private const val MAX_CONTEXT_TOKENS = 1500

@Composable
fun ChatScreen(
    port: Int,
    modifier: Modifier = Modifier
) {
    val messages = remember { mutableStateListOf<ChatMessage>() }
    val inputText = remember { mutableStateOf("") }
    val isLoading = remember { mutableStateOf(false) }
    val errorMessage = remember { mutableStateOf<String?>(null) }
    val listState = rememberLazyListState()
    val focusManager = LocalFocusManager.current
    val scope = rememberCoroutineScope()
    val sessionId = remember { java.util.UUID.randomUUID().toString().take(8) }

    val client = remember {
        OkHttpClient.Builder()
            .connectTimeout(5, TimeUnit.SECONDS)
            .readTimeout(120, TimeUnit.SECONDS)
            .build()
    }

    LaunchedEffect(messages.size) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(messages.size - 1)
        }
    }

    fun sendMessage() {
        val text = inputText.value.trim()
        if (text.isEmpty() || isLoading.value) return

        inputText.value = ""
        errorMessage.value = null
        focusManager.clearFocus()

        // Build truncated history -- keep only last ~MAX_CONTEXT_TOKENS of context
        val history = prepareContext(messages.toList())

        messages.add(ChatMessage(role = "user", content = text))
        val assistantId = java.util.UUID.randomUUID().toString()
        val assistantMsg = ChatMessage(id = assistantId, role = "assistant", content = "", isStreaming = true)
        messages.add(assistantMsg)
        isLoading.value = true

        scope.launch {
            try {
                val result = callChatApi(client, port, history, text, sessionId)
                val idx = messages.indexOfFirst { it.id == assistantId }
                if (idx >= 0) {
                    if (result.isNotEmpty()) {
                        messages[idx] = messages[idx].copy(content = result, isStreaming = false)
                    } else {
                        messages.removeAt(idx)
                        errorMessage.value = "Resposta vazia do servidor"
                    }
                }
                isLoading.value = false
            } catch (e: java.net.ConnectException) {
                val msg = "Servidor nao disponivel na porta $port"
                val idx = messages.indexOfFirst { it.id == assistantId }
                if (idx >= 0) {
                    messages[idx] = messages[idx].copy(content = msg, isStreaming = false)
                }
                errorMessage.value = msg
                isLoading.value = false
            } catch (e: Exception) {
                val msg = e.message ?: "Erro desconhecido"
                val idx = messages.indexOfFirst { it.id == assistantId }
                if (idx >= 0) {
                    messages[idx] = messages[idx].copy(content = "Erro: $msg", isStreaming = false)
                }
                errorMessage.value = msg
                isLoading.value = false
            }
        }
    }

    Column(modifier = modifier.fillMaxSize()) {
        Surface(
            modifier = Modifier.fillMaxWidth(),
            color = MaterialTheme.colorScheme.primary
        ) {
            Text(
                text = "Qwen3.5 PT-BR Chat",
                modifier = Modifier.padding(16.dp),
                color = Color.White,
                fontWeight = FontWeight.Bold,
                fontSize = 18.sp
            )
        }

        if (messages.isEmpty()) {
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = "Ola! Eu sou o Qwen3.5 finetuned para PT-BR.",
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Como posso ajudar?",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                    )
                }
            }
        } else {
            LazyColumn(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp),
                state = listState,
                verticalArrangement = Arrangement.spacedBy(8.dp),
                contentPadding = PaddingValues(vertical = 12.dp)
            ) {
                items(
                    items = messages.toList(),
                    key = { it.id }
                ) { message ->
                    ChatBubble(message = message)
                }
            }
        }

        AnimatedVisibility(
            visible = errorMessage.value != null,
            enter = fadeIn()
        ) {
            Surface(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp, vertical = 4.dp),
                color = MaterialTheme.colorScheme.errorContainer,
                shape = RoundedCornerShape(8.dp)
            ) {
                Text(
                    text = errorMessage.value ?: "",
                    modifier = Modifier.padding(12.dp),
                    color = MaterialTheme.colorScheme.onErrorContainer,
                    style = MaterialTheme.typography.bodySmall
                )
            }
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 8.dp),
            verticalAlignment = Alignment.Bottom
        ) {
            OutlinedTextField(
                value = inputText.value,
                onValueChange = { inputText.value = it },
                modifier = Modifier.weight(1f),
                placeholder = { Text("Digite sua mensagem...") },
                maxLines = 4,
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                keyboardActions = KeyboardActions(onSend = { sendMessage() }),
                shape = RoundedCornerShape(24.dp)
            )
            Spacer(modifier = Modifier.width(8.dp))
            IconButton(
                onClick = { sendMessage() },
                enabled = inputText.value.isNotBlank() && !isLoading.value,
                modifier = Modifier
                    .size(48.dp)
                    .background(
                        if (inputText.value.isNotBlank() && !isLoading.value)
                            MaterialTheme.colorScheme.primary
                        else
                            MaterialTheme.colorScheme.surfaceVariant,
                        shape = RoundedCornerShape(50)
                    )
            ) {
                if (isLoading.value) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(24.dp),
                        strokeWidth = 2.dp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                } else {
                    Icon(
                        imageVector = Icons.Default.Send,
                        contentDescription = "Enviar",
                        tint = if (inputText.value.isNotBlank()) Color.White
                        else MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}

// Truncate message history to fit within ~MAX_CONTEXT_TOKENS
private fun prepareContext(allMessages: List<ChatMessage>): List<ChatMessage> {
    val result = mutableListOf<ChatMessage>()
    var totalChars = 0
    // Walk backwards, keep most recent messages that fit
    for (msg in allMessages.asReversed()) {
        val chars = msg.content.length
        if (totalChars + chars > MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN && result.isNotEmpty()) {
            break
        }
        result.add(0, msg)
        totalChars += chars
    }
    return result
}

@Composable
fun ChatBubble(message: ChatMessage) {
    val isUser = message.role == "user"
    val alignment = if (isUser) Alignment.End else Alignment.Start
    val bgColor = if (isUser) Color(0xFF1A73E8) else Color(0xFFE8E8E8)
    val textColor = if (isUser) Color.White else Color(0xFF1C1B1F)
    val shape = RoundedCornerShape(
        topStart = 16.dp,
        topEnd = 16.dp,
        bottomStart = if (isUser) 16.dp else 4.dp,
        bottomEnd = if (isUser) 4.dp else 16.dp
    )

    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = alignment
    ) {
        Box(
            modifier = Modifier
                .widthIn(max = 300.dp)
                .clip(shape)
                .background(bgColor)
                .padding(12.dp)
        ) {
            Text(
                text = message.content,
                color = textColor,
                style = MaterialTheme.typography.bodyMedium,
                lineHeight = 20.sp
            )
        }
        if (message.isStreaming) {
            Text(
                text = "|",
                modifier = Modifier.padding(start = 8.dp),
                color = textColor.copy(alpha = 0.6f),
                fontSize = 12.sp
            )
        }
    }
}

private suspend fun callChatApi(
    client: OkHttpClient,
    port: Int,
    history: List<ChatMessage>,
    userMessage: String,
    sessionId: String
): String = withContext(Dispatchers.IO) {
    suspendCancellableCoroutine { continuation ->
        val responseContent = StringBuilder()
        var eventSource: EventSource? = null

        try {
            val requestJson = buildRequestJson(history, userMessage, sessionId)
            val requestBody = requestJson.toString()
                .toRequestBody("application/json".toMediaType())

            val request = Request.Builder()
                .url("http://127.0.0.1:$port/v1/chat/completions")
                .post(requestBody)
                .header("Content-Type", "application/json")
                .build()

            eventSource = EventSources.createFactory(client)
                .newEventSource(request, object : EventSourceListener() {
                    override fun onEvent(
                        eventSource: EventSource,
                        id: String?,
                        type: String?,
                        data: String
                    ) {
                        if (data == "[DONE]") return
                        try {
                            val json = JSONObject(data)
                            val choices = json.getJSONArray("choices")
                            if (choices.length() > 0) {
                                val delta = choices.getJSONObject(0)
                                    .getJSONObject("delta")
                                if (delta.has("content")) {
                                    val token = delta.getString("content")
                                    responseContent.append(token)
                                    // Strip think tags: if we collected a full tag, remove it
                                    val cleaned = responseContent.toString()
                                        .replace(Regex("<think>[\\s\\S]*?</think>"), "")
                                    responseContent.clear()
                                    responseContent.append(cleaned)
                                }
                            }
                        } catch (_: Exception) {}
                    }

                    override fun onClosed(eventSource: EventSource) {
                        if (continuation.isActive) {
                            continuation.resume(responseContent.toString())
                        }
                    }

                    override fun onFailure(
                        eventSource: EventSource,
                        t: Throwable?,
                        response: Response?
                    ) {
                        if (continuation.isActive) {
                            if (responseContent.isNotEmpty()) {
                                continuation.resume(responseContent.toString())
                            } else {
                                continuation.resumeWithException(
                                    t ?: java.io.IOException("Conexao falhou")
                                )
                            }
                        }
                    }
                })

            continuation.invokeOnCancellation {
                try { eventSource?.cancel() } catch (_: Exception) {}
            }
        } catch (e: Exception) {
            if (continuation.isActive) {
                continuation.resumeWithException(e)
            }
        }
    }
}

private fun buildRequestJson(
    history: List<ChatMessage>,
    userMessage: String,
    sessionId: String
): JSONObject {
    // Extract user context from conversation history for memory injection
    val memoryHints = buildMemoryHints(history)

    return JSONObject().apply {
        put("messages", JSONArray().apply {
            put(JSONObject().apply {
                put("role", "system")
                put("content", "Voce e um assistente util e amigavel que fala portugues brasileiro. " +
                    "Responda sempre em portugues de forma clara, direta e natural. " +
                    "Preste atencao ao historico da conversa e use as informacoes anteriores quando relevante. " +
                    memoryHints)
            })
            for (msg in history) {
                put(JSONObject().apply {
                    put("role", msg.role)
                    put("content", msg.content)
                })
            }
            put(JSONObject().apply {
                put("role", "user")
                put("content", userMessage)
            })
        })
        put("max_tokens", 512)
        put("temperature", 0.4)
        put("repeat_penalty", 1.05)
        put("stream", true)
        put("id_session", sessionId)
    }
}

// Extract user name/location from conversation for memory injection
private fun buildMemoryHints(history: List<ChatMessage>): String {
    val userMessages = history.filter { it.role == "user" }.takeLast(5)
    val hints = mutableListOf<String>()

    for (msg in userMessages) {
        val c = msg.content.lowercase()
        // Extract name patterns
        val nameMatch = Regex("(?:meu nome [ée]|me chamo|sou (?:o|a))\\s+(\\w+)").find(c)
        if (nameMatch != null) {
            hints.add("O usuario se chama ${nameMatch.groupValues[1].replaceFirstChar { it.uppercase() }}.")
        }
        // Extract location
        if (Regex("moro em|sou de|vivo em").containsMatchIn(c)) {
            val locMatch = Regex("(?:moro em|sou de|vivo em)\\s+([^.]+)").find(c)
            if (locMatch != null) {
                hints.add("O usuario mora em ${locMatch.groupValues[1].trim()}.")
            }
        }
    }

    return if (hints.isNotEmpty()) "Contexto: ${hints.joinToString(" ")}" else ""
}
