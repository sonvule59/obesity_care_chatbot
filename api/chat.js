/**
 * PRODUCTION LLM — POST /api/chat (new server route; not wired yet)
 *
 * Deploy as:
 *   • Vercel: api/chat.js (this file) or api/chat/route.js
 *   • Netlify: netlify/functions/chat.js
 *   • Express: app.post("/api/chat", handler)
 *
 * ─── Environment (server only — never VITE_*) ───────────────────────────────
 *   ANTHROPIC_API_KEY=sk-ant-...
 *   LLM_PROVIDER=anthropic          # anthropic | groq (staging/dev on server)
 *   LLM_MODEL=claude-haiku-4-5-20251001
 *
 * ─── Request body (from frontend callLLMOnce) ─────────────────────────────────
 *   {
 *     messages: [{ role: "user"|"assistant", content: string }],  // last N turns
 *     systemPrompt: string,
 *     maxTokens?: number,          // default 350
 *     moduleId?: string,           // chat | checkin | education | judge
 *     conditionId?: string,        // A | B | C — for tools / caching keys
 *     stream?: boolean             // default true
 *   }
 *
 * ─── Anthropic Messages API mapping ─────────────────────────────────────────
 *   POST https://api.anthropic.com/v1/messages
 *   Headers:
 *     x-api-key: process.env.ANTHROPIC_API_KEY
 *     anthropic-version: 2023-06-01
 *     content-type: application/json
 *   Body:
 *     model: "claude-haiku-4-5-20251001"
 *     max_tokens: maxTokens
 *     temperature: 0.3
 *     system: systemPrompt          // NOT in messages array
 *     messages: [...]              // user/assistant only
 *     stream: true
 *
 *   Streaming SSE (Anthropic native):
 *     event: content_block_delta → data.delta.text
 *     event: message_stop → end stream
 *   Normalize to OpenAI-style for the React client OR update parser in App.jsx callGroqOnce.
 *
 * ─── Phase 4: prompt caching ────────────────────────────────────────────────
 *   For long system prompts (Condition B/C chatTrained), add cache_control on system block:
 *     system: [{ type: "text", text: systemPrompt, cache_control: { type: "ephemeral" } }]
 *
 * ─── Phase 5: tool loop (reference library) ─────────────────────────────────
 *   1. First call with tools: [search_references, get_reference_by_id]
 *   2. If stop_reason === "tool_use", execute tools server-side (reuse retrieveChunks logic)
 *   3. Append tool_result messages; call again until stop_reason === "end_turn"
 *   4. Stream final assistant text to client
 *
 * ─── Security / trial requirements ────────────────────────────────────────────
 *   • Never log PHI; de-identify before audit logs
 *   • Rate limit per IP or session
 *   • Optional: require trial participant token in Authorization header
 *   • Return generic 500 to client; log details server-side only
 *
 * ─── Response headers (optional metadata for frontend) ──────────────────────
 *   X-LLM-Model: claude-haiku-4-5-20251001
 *   X-LLM-Provider: anthropic
 *
 * Implementation stub — export handler when ready:
 *
 * export default async function handler(req, res) {
 *   if (req.method !== "POST") return res.status(405).end();
 *   // ... validate body, call Anthropic, pipe stream ...
 * }
 */
