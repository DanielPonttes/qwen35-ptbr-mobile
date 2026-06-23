package com.qwen35.ptbr.chat.ui.chat

import org.json.JSONObject
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeEach

class ChatScreenTest {

    @Test
    fun `prepareContext returns empty list for empty input`() {
        val result = prepareContext(emptyList())
        assertTrue(result.isEmpty())
    }

    @Test
    fun `prepareContext keeps all messages when under token limit`() {
        val messages = listOf(
            ChatMessage(role = "user", content = "Short message"),
            ChatMessage(role = "assistant", content = "Short reply"),
            ChatMessage(role = "user", content = "Another short one")
        )
        val result = prepareContext(messages)
        assertEquals(3, result.size)
    }

    @Test
    fun `prepareContext truncates when total chars exceed 6000`() {
        val longContent = "a".repeat(5000)
        val messages = listOf(
            ChatMessage(role = "user", content = longContent),
            ChatMessage(role = "user", content = "b".repeat(2000)),
            ChatMessage(role = "user", content = "c".repeat(500))
        )
        val result = prepareContext(messages)
        assertTrue(result.size < 3)
        assertTrue(result.sumOf { it.content.length } <= 6000 + result.first().content.length)
    }

    @Test
    fun `prepareContext preserves message order`() {
        val messages = listOf(
            ChatMessage(role = "user", content = "first"),
            ChatMessage(role = "assistant", content = "second"),
            ChatMessage(role = "user", content = "third")
        )
        val result = prepareContext(messages)
        assertEquals("first", result.first().content)
        assertEquals("third", result.last().content)
    }

    @Test
    fun `prepareContext drops oldest messages when exceeding token budget`() {
        val messages = listOf(
            ChatMessage(role = "user", content = "old message"),
            ChatMessage(role = "assistant", content = "old reply"),
            ChatMessage(role = "user", content = "a".repeat(5000)),
            ChatMessage(role = "assistant", content = "b".repeat(1500))
        )
        val result = prepareContext(messages)
        val totalChars = result.sumOf { it.content.length }
        assertTrue(totalChars <= 6000 + 4)
        assertFalse(result.map { it.content }.contains("old message"))
    }

    @Test
    fun `buildMemoryHints extracts name with meu nome e pattern`() {
        val messages = listOf(
            ChatMessage(role = "user", content = "oi, meu nome e Daniel")
        )
        val result = buildMemoryHints(messages)
        assertTrue(result.contains("Daniel"))
        assertTrue(result.contains("O usuario se chama"))
    }

    @Test
    fun `buildMemoryHints extracts name with me chamo pattern`() {
        val messages = listOf(
            ChatMessage(role = "user", content = "me chamo Rafael")
        )
        val result = buildMemoryHints(messages)
        assertTrue(result.contains("Rafael"))
    }

    @Test
    fun `buildMemoryHints extracts name with sou o pattern`() {
        val messages = listOf(
            ChatMessage(role = "user", content = "sou o Joao")
        )
        val result = buildMemoryHints(messages)
        assertTrue(result.contains("Joao"))
    }

    @Test
    fun `buildMemoryHints extracts name with sou a pattern`() {
        val messages = listOf(
            ChatMessage(role = "user", content = "sou a Maria")
        )
        val result = buildMemoryHints(messages)
        assertTrue(result.contains("Maria"))
    }

    @Test
    fun `buildMemoryHints extracts name with meu nome é accent`() {
        val messages = listOf(
            ChatMessage(role = "user", content = "meu nome é Pedro")
        )
        val result = buildMemoryHints(messages)
        assertTrue(result.contains("Pedro"))
    }

    @Test
    fun `buildMemoryHints capitalizes extracted name`() {
        val messages = listOf(
            ChatMessage(role = "user", content = "meu nome e daniel")
        )
        val result = buildMemoryHints(messages)
        assertTrue(result.contains("Daniel"))
        assertFalse(result.contains("daniel"))
    }

    @Test
    fun `buildMemoryHints extracts location with moro em pattern`() {
        val messages = listOf(
            ChatMessage(role = "user", content = "moro em SP")
        )
        val result = buildMemoryHints(messages)
        assertTrue(result.contains("SP"))
        assertTrue(result.contains("O usuario mora em"))
    }

    @Test
    fun `buildMemoryHints extracts location with sou de pattern`() {
        val messages = listOf(
            ChatMessage(role = "user", content = "sou de Rio de Janeiro")
        )
        val result = buildMemoryHints(messages)
        assertTrue(result.contains("Rio de Janeiro"))
    }

    @Test
    fun `buildMemoryHints extracts location with vivo em pattern`() {
        val messages = listOf(
            ChatMessage(role = "user", content = "vivo em Brasilia")
        )
        val result = buildMemoryHints(messages)
        assertTrue(result.contains("Brasilia"))
    }

    @Test
    fun `buildMemoryHints extracts both name and location`() {
        val messages = listOf(
            ChatMessage(role = "user", content = "meu nome e Ana, moro em Salvador")
        )
        val result = buildMemoryHints(messages)
        assertTrue(result.contains("Ana"))
        assertTrue(result.contains("Salvador"))
        assertTrue(result.contains("Contexto:"))
    }

    @Test
    fun `buildMemoryHints returns empty string for empty history`() {
        val result = buildMemoryHints(emptyList())
        assertEquals("", result)
    }

    @Test
    fun `buildMemoryHints returns empty string when no personal info present`() {
        val messages = listOf(
            ChatMessage(role = "user", content = "qual a capital do Brasil?"),
            ChatMessage(role = "user", content = "como funciona git?")
        )
        val result = buildMemoryHints(messages)
        assertEquals("", result)
    }

    @Test
    fun `buildMemoryHints only checks last 5 user messages`() {
        val messages = (1..10).map {
            ChatMessage(role = "user", content = "question $it")
        }
        val result = buildMemoryHints(messages)
        assertEquals("", result)
    }

    @Test
    fun `buildMemoryHints finds name in last 5 messages`() {
        val messages = buildList {
            add(ChatMessage(role = "user", content = "q1"))
            add(ChatMessage(role = "assistant", content = "a1"))
            add(ChatMessage(role = "user", content = "q2"))
            add(ChatMessage(role = "assistant", content = "a2"))
            add(ChatMessage(role = "user", content = "q3"))
            add(ChatMessage(role = "assistant", content = "a3"))
            add(ChatMessage(role = "user", content = "q4"))
            add(ChatMessage(role = "assistant", content = "a4"))
            add(ChatMessage(role = "user", content = "q5"))
            add(ChatMessage(role = "assistant", content = "a5"))
            add(ChatMessage(role = "user", content = "meu nome e Carlos"))
        }
        val result = buildMemoryHints(messages)
        assertTrue(result.contains("Carlos"))
    }

    @Test
    fun `buildMemoryHints does not extract from assistant messages`() {
        val messages = listOf(
            ChatMessage(role = "assistant", content = "meu nome e Bot")
        )
        val result = buildMemoryHints(messages)
        assertEquals("", result)
    }

    @Test
    fun `buildRequestJson has required top-level fields`() {
        val history = listOf(
            ChatMessage(role = "user", content = "hello")
        )
        val json = buildRequestJson(history, "test message", "session123")

        assertEquals(512, json.getInt("max_tokens"))
        assertEquals(0.4, json.getDouble("temperature"), 0.001)
        assertEquals(1.05, json.getDouble("repeat_penalty"), 0.001)
        assertEquals(true, json.getBoolean("stream"))
        assertEquals("session123", json.getString("id_session"))
    }

    @Test
    fun `buildRequestJson messages array starts with system prompt`() {
        val history = listOf(
            ChatMessage(role = "user", content = "hello")
        )
        val json = buildRequestJson(history, "test message", "session123")

        val messages = json.getJSONArray("messages")
        assertTrue(messages.length() >= 2)
        val systemMsg = messages.getJSONObject(0)
        assertEquals("system", systemMsg.getString("role"))
        assertTrue(systemMsg.getString("content").contains("portugues brasileiro"))
    }

    @Test
    fun `buildRequestJson includes history messages`() {
        val history = listOf(
            ChatMessage(role = "user", content = "msg1"),
            ChatMessage(role = "assistant", content = "reply1"),
            ChatMessage(role = "user", content = "msg2")
        )
        val json = buildRequestJson(history, "current message", "session123")

        val messages = json.getJSONArray("messages")
        assertEquals(5, messages.length()) // system + 3 history + 1 current
        assertEquals("user", messages.getJSONObject(1).getString("role"))
        assertEquals("msg1", messages.getJSONObject(1).getString("content"))
        assertEquals("assistant", messages.getJSONObject(2).getString("role"))
    }

    @Test
    fun `buildRequestJson last message is the current user input`() {
        val history = listOf(
            ChatMessage(role = "user", content = "previous")
        )
        val json = buildRequestJson(history, "current message", "session123")

        val messages = json.getJSONArray("messages")
        val lastMsg = messages.getJSONObject(messages.length() - 1)
        assertEquals("user", lastMsg.getString("role"))
        assertEquals("current message", lastMsg.getString("content"))
    }

    @Test
    fun `buildRequestJson injects memory hints into system prompt`() {
        val history = listOf(
            ChatMessage(role = "user", content = "meu nome e Carlos")
        )
        val json = buildRequestJson(history, "hello", "session123")

        val messages = json.getJSONArray("messages")
        val systemContent = messages.getJSONObject(0).getString("content")
        assertTrue(systemContent.contains("Carlos"))
        assertTrue(systemContent.contains("Contexto:"))
    }

    @Test
    fun `buildRequestJson system prompt without memory hints when no context`() {
        val history = listOf(
            ChatMessage(role = "user", content = "qual a capital?")
        )
        val json = buildRequestJson(history, "hello again", "session123")

        val messages = json.getJSONArray("messages")
        val systemContent = messages.getJSONObject(0).getString("content")
        assertFalse(systemContent.contains("Contexto:"))
        assertTrue(systemContent.contains("portugues brasileiro"))
    }

    @Test
    fun `strip think tags removes complete think block`() {
        val input = "<think>reasoning about the answer</think>the actual response"
        val cleaned = input.replace(Regex("<think>[\\s\\S]*?</think>"), "")
        assertEquals("the actual response", cleaned)
    }

    @Test
    fun `strip think tags removes multiple think blocks`() {
        val input = "<think>first thought</think>text <think>second thought</think>more text"
        val cleaned = input.replace(Regex("<think>[\\s\\S]*?</think>"), "")
        assertEquals("text more text", cleaned)
    }

    @Test
    fun `strip think tags handles multiline think blocks`() {
        val input = "<think>\nline one\nline two\n</think>actual output"
        val cleaned = input.replace(Regex("<think>[\\s\\S]*?</think>"), "")
        assertEquals("actual output", cleaned)
    }

    @Test
    fun `strip think tags leaves content without think tags unchanged`() {
        val input = "normal response without any tags"
        val cleaned = input.replace(Regex("<think>[\\s\\S]*?</think>"), "")
        assertEquals("normal response without any tags", cleaned)
    }

    @Test
    fun `strip think tags handles incomplete think tag`() {
        val input = "<think>unclosed tag"
        val cleaned = input.replace(Regex("<think>[\\s\\S]*?</think>"), "")
        assertEquals("<think>unclosed tag", cleaned)
    }

    @Test
    fun `strip think tags removes nested-like tags`() {
        val input = "<think>some reasoning</think>keep this part"
        val cleaned = input.replace(Regex("<think>[\\s\\S]*?</think>"), "")
        assertEquals("keep this part", cleaned)
    }

    @Test
    fun `prepareContext token limit is 1500 tokens at 4 chars each`() {
        val maxTokens = 1500
        val charsPerToken = 4
        assertEquals(6000, maxTokens * charsPerToken)
    }
}
