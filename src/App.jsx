import { useState, useRef, useEffect, useCallback, useMemo } from "react";

// ─── PRODUCTION LLM MIGRATION (Claude Haiku 4.5) — implementation map ─────────
// Phase 1 — Backend proxy (required for production; Anthropic key must NOT live in VITE_*)
//   • Add server route: POST /api/chat (Vercel serverless, Netlify Function, or Express).
//   • Server env only: ANTHROPIC_API_KEY, LLM_MODEL=claude-haiku-4-5-20251001, LLM_PROVIDER=anthropic.
//   • Frontend env only: VITE_API_BASE_URL=https://your-host (no LLM API key in browser).
//   • vite.config.js: optional dev proxy /api → local backend (see comment there).
//
// Phase 2 — Provider abstraction (replace direct Groq coupling below)
//   • Rename callGroq / callGroqOnce → callLLM / callLLMOnce (or keep names, delegate inside).
//   • callLLMOnce: if VITE_API_BASE_URL set → fetch POST /api/chat with { messages, systemPrompt, maxTokens };
//     else dev fallback → existing Groq OpenAI-compatible call.
//   • Server maps request → Anthropic Messages API (api.anthropic.com/v1/messages):
//     - system prompt → top-level `system` field (NOT role:"system" in messages)
//     - messages → only user/assistant roles
//     - headers: x-api-key, anthropic-version: 2023-06-01
//     - stream: true → SSE events content_block_delta (different from OpenAI choices[0].delta.content)
//   • Server streams tokens back in one normalized format the client already parses.
//
// Phase 3 — Production cutover
//   • Deploy backend with Haiku; point VITE_API_BASE_URL at it.
//   • Keep Groq path for local dev (no VITE_API_BASE_URL) or set LLM_PROVIDER=groq on server for staging.
//   • Tune max_tokens / temperature per module (chat 350, assessments 280, judge 200).
//
// Phase 4 — Prompt caching (server-side, optional)
//   • Long system prompts (buildSystems chatTrained, Condition C instruments) → Anthropic prompt caching
//     to cut cost/latency on repeated turns.
//
// Phase 5 — Reference library + tools (server-side, optional)
//   • Move retrieveChunks / KNOWLEDGE_CHUNKS search into server tool executor.
//   • Anthropic tools: search_references, get_reference_by_id; agent loop on server before final stream.
//   • Disable always-on RAG injection for Condition A when tools are live (see send() + retrieveChunks).
//
// Touchpoints in this file: env constants (below), callGroqOnce/callGroq, send(), scoreResponseWithJudge(),
//   buildResearchExportReport model field, Progress UI model label, error messages in send() catch.
// ─────────────────────────────────────────────────────────────────────────────

const GROQ_API_KEY = import.meta.env.VITE_GROQ_API_SECRET_KEY;
// PRODUCTION: remove hard requirement on Groq key when using proxy — require VITE_API_BASE_URL OR Groq key.
if (!GROQ_API_KEY) {
  throw new Error("VITE_GROQ_API_SECRET_KEY is not set in .env");
}
const GROQ_MODEL = import.meta.env.VITE_GROQ_MODEL || "llama-3.1-8b-instant";
// PRODUCTION: replace with LLM_MODEL_DISPLAY from /api/chat health response, or
//   const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";
//   const LLM_MODEL = import.meta.env.VITE_LLM_MODEL_DISPLAY || GROQ_MODEL;  // server reports actual model

// ─── FUNCTION CALLING (Groq/OpenAI-compatible tools; use with a tool-capable
// model like llama-3.3-70b-versatile) ─────────────────────────────────────────
// CURRENTLY OFF BY DEFAULT — we are running RAG (always-on keyword injection) for
// now. The function-calling code path is left intact (send() + callGroqWithTools)
// so we can flip back to on-demand tool retrieval later without a rewrite.
// When ON, the PA Coach chat exposes a `search_references` tool and the model
// decides WHEN to retrieve — instead of the always-on RAG injection.
//   Re-enable function calling:  VITE_FUNCTION_CALLING=true  (turns RAG off automatically)
//   Show tool debug in chat (dev only):  VITE_TOOL_DEBUG=true
const FUNCTION_CALLING_ENABLED = (import.meta.env.VITE_FUNCTION_CALLING ?? "false") === "true";
const SHOW_TOOL_DEBUG = import.meta.env.VITE_TOOL_DEBUG === "true";
const TOOL_CALLS_LOG_KEY = "confidentMoves_tool_calls";

/** Log reference-library tool usage for research (not shown in participant chat). */
function recordReferenceLookups(meta = {}) {
  try {
    const raw = localStorage.getItem(TOOL_CALLS_LOG_KEY);
    const prev = raw ? JSON.parse(raw) : [];
    const entry = { recordedAt: new Date().toISOString(), ...meta };
    prev.unshift(entry);
    localStorage.setItem(TOOL_CALLS_LOG_KEY, JSON.stringify(prev.slice(0, 200)));
    return entry;
  } catch { return null; }
}

function formatReferenceLookups(lookups) {
  if (!lookups?.length) return "";
  return lookups
    .map(l => `search("${l.query ?? ""}") → [${(l.ids ?? []).join(", ") || "none"}]`)
    .join("; ");
}

/** Skip tool round-trip for short confirmations (e.g. "Tuesday and Thursday evenings"). */
function messageLikelyNeedsReferences(text) {
  const t = String(text ?? "").toLowerCase().trim();
  if (!t) return false;
  if (/^(thanks|thank you|ok|okay|yes|sure|got it|sounds good|that works|perfect|great)\b/i.test(t)) return false;
  if (/^(hi|hello|hey|good morning|good evening)\b/i.test(t)) return false;

  const referenceSignals = [
    "pain", "hurt", "nausea", "tired", "fatigue", "embarrass", "ashamed", "stressed",
    "weather", "rain", "busy", "no time", "semaglutide", "glp", "injection", "medication",
    "barrier", "confidence", "gym", "guideline", "how many", "muscle", "side effect",
    "alone", "lonely", "self-conscious", "overwhelmed", "anxious",
  ];
  if (referenceSignals.some(s => t.includes(s))) return true;

  // Agreeing to a schedule/plan — coach from context; no library lookup
  if (t.length < 100 && /\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|evening|morning|weekend)\b/.test(t)) {
    return false;
  }

  return t.length >= 45;
}

const API_METRICS_KEY = "confidentMoves_api_metrics";
const MIN_GROQ_INTERVAL_MS = 2500;
const ASSESSMENT_GROQ_INTERVAL_MS = 3500;
const CHAT_MAX_TOKENS = 450;        // was 350 — 70B follows brevity rules tightly; +100 headroom
const ASSESSMENT_MAX_TOKENS = 300;  // was 280
// const JUDGE_MAX_TOKENS = 200;    // LLM-as-judge disabled — restore when benchmarking resumes
// PRODUCTION: rename to MIN_LLM_INTERVAL_MS; rate limiting can move entirely to server (per session/user).

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function readApiMetrics() {
  try {
    const raw = localStorage.getItem(API_METRICS_KEY);
    if (!raw) return { totalCalls: 0, successCalls: 0, rateLimitHits: 0, errors: 0, lastCallAt: null, last429At: null };
    return { totalCalls: 0, successCalls: 0, rateLimitHits: 0, errors: 0, lastCallAt: null, last429At: null, ...JSON.parse(raw) };
  } catch {
    return { totalCalls: 0, successCalls: 0, rateLimitHits: 0, errors: 0, lastCallAt: null, last429At: null };
  }
}

function recordApiMetric(event) {
  try {
    const m = readApiMetrics();
    const now = new Date().toISOString();
    if (event === "call") { m.totalCalls += 1; m.lastCallAt = now; }
    if (event === "success") { m.successCalls += 1; }
    if (event === "429") { m.rateLimitHits += 1; m.last429At = now; }
    if (event === "error") { m.errors += 1; }
    localStorage.setItem(API_METRICS_KEY, JSON.stringify(m));
  } catch {}
}

let lastGroqCallFinishedAt = 0;
// PRODUCTION: rename lastGroqCallFinishedAt → lastLlmCallFinishedAt

async function callGroqOnce(messages, systemPrompt, onChunk, maxTokens = CHAT_MAX_TOKENS) {
  // PRODUCTION Phase 2 — branch here:
  //   if (API_BASE_URL) return callProxyOnce(API_BASE_URL + "/api/chat", { messages, systemPrompt, maxTokens }, onChunk);
  //   else return callGroqDirect(...)  // current implementation below (dev only)
  //
  // Server /api/chat handler (new file, e.g. api/chat.js or server/routes/chat.ts):
  //   1. Validate body; strip/limit message history; optional auth header.
  //   2. POST https://api.anthropic.com/v1/messages with model claude-haiku-4-5-20251001.
  //   3. If tools requested later: run tool loop (non-streaming) then stream final answer.
  //   4. Pipe Anthropic SSE → client as simple text deltas (or teach client new parser below).
  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${GROQ_API_KEY}`,
    },
    body: JSON.stringify({
      model: GROQ_MODEL,
      max_tokens: maxTokens,
      temperature: 0.3,
      stream: true,
      messages: [
        { role: "system", content: systemPrompt },
        ...messages.map(m => ({ role: m.role, content: m.content || "..." })),
      ],
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Groq API error ${res.status}: ${err}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let full = "";

  // PRODUCTION: if proxy normalizes to same OpenAI-style SSE, this loop can stay unchanged.
  // If parsing Anthropic SSE in the client instead, handle event types:
  //   content_block_delta → delta.text; message_stop → done; error → throw.
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const lines = decoder.decode(value).split("\n");
    for (const line of lines) {
      if (line.startsWith("data: ") && !line.includes("[DONE]")) {
        try {
          const d = JSON.parse(line.slice(6));
          const text = d.choices?.[0]?.delta?.content;
          if (text) { full += text; onChunk(full); }
        } catch {}
      }
    }
  }
  return full;
}

async function callGroq(messages, systemPrompt, onChunk, options = {}) {
  // PRODUCTION: rename to callLLM — same signature so send() and scoreResponseWithJudge() need minimal edits.
  // Retries on 429 and recordApiMetric() can stay here or move to server.
  const minInterval = options.minIntervalMs ?? MIN_GROQ_INTERVAL_MS;
  const maxTokens = options.maxTokens ?? CHAT_MAX_TOKENS;
  const maxRetries = options.maxRetries ?? 4;

  const wait = lastGroqCallFinishedAt + minInterval - Date.now();
  if (wait > 0) await sleep(wait);

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    recordApiMetric("call");
    try {
      const full = await callGroqOnce(messages, systemPrompt, onChunk, maxTokens);
      recordApiMetric("success");
      lastGroqCallFinishedAt = Date.now();
      return full;
    } catch (e) {
      const errMsg = e?.message || String(e);
      const is429 = errMsg.includes("429");
      if (is429) recordApiMetric("429");
      else recordApiMetric("error");

      if (is429 && attempt < maxRetries) {
        const backoffMs = 6000 * (attempt + 1);
        await sleep(backoffMs);
        continue;
      }
      lastGroqCallFinishedAt = Date.now();
      throw e;
    }
  }
}

// ─── FUNCTION CALLING: reference-library tool (OpenAI-compatible tool schema) ──
// The model calls search_references(query, limit) when it needs grounded evidence.
// The executor reuses the existing keyword retriever over KNOWLEDGE_CHUNKS, so no
// new knowledge base is required to test the mechanism on Groq's 70B model.
const REFERENCE_TOOLS = [
  {
    type: "function",
    function: {
      name: "search_references",
      description:
        "Search the ObesityCare clinical reference library for evidence to ground coaching: physical-activity guidelines; Bandura self-efficacy coaching rules by domain (job/transport/domestic/leisure); SERPA barrier strategies (schedule, self-consciousness, pain/discomfort, weather, social support, stress, no encouragement); GLP-1 + exercise facts; and behavior-change technique rules (action planning, confidence check, problem solving). Call this whenever the participant mentions a barrier or asks something where grounded clinical guidance improves the reply. Do NOT call it for greetings or small talk.",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description:
              "Concise natural-language description of what to look up, e.g. 'embarrassed exercising in public' or 'nausea from semaglutide and exercise'.",
          },
          limit: {
            type: "integer",
            description: "How many reference snippets to return (1-5). Default 3.",
          },
        },
        required: ["query"],
      },
    },
  },
];

/** Execute a tool call requested by the model. Returns a JSON string (tool result). */
function executeReferenceTool(name, args) {
  if (name === "search_references") {
    const query = String(args?.query ?? "").trim();
    const limit = Math.max(1, Math.min(5, Number(args?.limit) || 3));
    const chunks = retrieveChunks(query, "", {}, limit);
    if (!chunks.length) {
      return JSON.stringify({
        results: [],
        note: "No matching references found. Answer from general knowledge and stay within safe coaching scope.",
      });
    }
    return JSON.stringify({ results: chunks.map(c => ({ id: c.id, text: c.text })) });
  }
  return JSON.stringify({ error: `Unknown tool: ${name}` });
}

/** Short nudge appended to the system prompt so the model knows the tool exists. */
function buildToolInstructionBlock() {
  return `\n\nREFERENCE TOOL: You have a function \`search_references\` for barrier-specific or factual clinical questions. Call it BEFORE giving barrier-specific or factual guidance. Do NOT call it for greetings, thanks, short confirmations, or when the participant is agreeing to a specific plan (days/times). Synthesize results naturally — never paste snippets verbatim.`;
}

/**
 * Tool decision (1 non-streaming call) → optional tool execution → streaming final answer.
 * Faster UX than multiple non-streaming completions; confirmations skip tools upstream.
 */
async function callGroqOnceWithTools(messages, systemPrompt, onChunk, maxTokens, onToolEvent) {
  const apiMessages = messages.map(m => ({ role: m.role, content: m.content || "..." }));

  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${GROQ_API_KEY}`,
    },
    body: JSON.stringify({
      model: GROQ_MODEL,
      max_tokens: 150,
      temperature: 0.1,
      stream: false,
      tools: REFERENCE_TOOLS,
      tool_choice: "auto",
      messages: [{ role: "system", content: systemPrompt }, ...apiMessages],
    }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Groq API error ${res.status}: ${err}`);
  }

  const data = await res.json();
  const msg = data.choices?.[0]?.message ?? {};
  const toolCalls = msg.tool_calls ?? [];
  let activePrompt = systemPrompt;

  if (toolCalls.length > 0) {
    const collectedChunks = [];
    for (const tc of toolCalls) {
      let toolArgs = {};
      try { toolArgs = JSON.parse(tc.function?.arguments || "{}"); } catch {}
      const result = executeReferenceTool(tc.function?.name, toolArgs);
      if (onToolEvent) onToolEvent({ name: tc.function?.name, args: toolArgs, result });
      try {
        const parsed = JSON.parse(result);
        for (const r of parsed.results ?? []) {
          collectedChunks.push({ id: r.id, text: r.text });
        }
      } catch {}
    }
    if (collectedChunks.length) {
      activePrompt += buildRagBlock(collectedChunks);
    }
  } else if (msg.content?.trim()) {
    onChunk(msg.content);
    return msg.content;
  }

  return callGroqOnce(apiMessages, activePrompt, onChunk, maxTokens);
}

/** Retry/rate-limit wrapper around callGroqOnceWithTools (mirrors callGroq).
 *  On tool_use_failed (400), falls back to a plain streaming call without tools. */
async function callGroqWithTools(messages, systemPrompt, onChunk, options = {}) {
  const minInterval = options.minIntervalMs ?? MIN_GROQ_INTERVAL_MS;
  const maxTokens = options.maxTokens ?? CHAT_MAX_TOKENS;
  const maxRetries = options.maxRetries ?? 4;
  const onToolEvent = options.onToolEvent;

  const wait = lastGroqCallFinishedAt + minInterval - Date.now();
  if (wait > 0) await sleep(wait);

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    recordApiMetric("call");
    try {
      const full = await callGroqOnceWithTools(messages, systemPrompt, onChunk, maxTokens, onToolEvent);
      recordApiMetric("success");
      lastGroqCallFinishedAt = Date.now();
      return full;
    } catch (e) {
      const errMsg = e?.message || String(e);
      const is429 = errMsg.includes("429");
      const isTool400 = errMsg.includes("400") && (errMsg.includes("tool_use_failed") || errMsg.includes("Failed to call a function"));
      if (is429) recordApiMetric("429");
      else recordApiMetric("error");

      // Groq sometimes rejects malformed tool output — coach without tools rather than failing the turn.
      if (isTool400) {
        console.warn("Tool call failed (400) — falling back to plain chat:", errMsg.slice(0, 300));
        recordApiMetric("call");
        const full = await callGroqOnce(messages, systemPrompt, onChunk, maxTokens);
        recordApiMetric("success");
        lastGroqCallFinishedAt = Date.now();
        return full;
      }

      if (is429 && attempt < maxRetries) {
        const backoffMs = 6000 * (attempt + 1);
        await sleep(backoffMs);
        continue;
      }
      lastGroqCallFinishedAt = Date.now();
      throw e;
    }
  }
}

const INSTRUMENT_LOG_KEY = "confidentMoves_instrument_log";
const GOOGLE_SHEETS_WEBAPP_URL = import.meta.env.VITE_GOOGLE_SHEETS_WEBAPP_URL || "";

function readInstrumentLog() {
  try {
    const raw = localStorage.getItem(INSTRUMENT_LOG_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function triggerDownload(filename, mime, body) {
  const blob = new Blob([body], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function csvEscapeCell(v) {
  if (v === null || v === undefined) return "";
  const s = String(v);
  if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

// ─── De-identification helper ─────────────────────────────────
// All exports use participant CODE (PT-0001) not real names.
function deidentify(record) {
  return record.participantId ?? record.profileId ?? "UNKNOWN";
}

// ─── CSV exports (Excel-friendly, flat, de-identified) ────────

function instrumentLogToCsv(log) {
  // One flat row per submission. Numeric scores in separate columns — open in Excel directly.
  const paseKeys = ["item1","item2","item3","item4","item5","item6"];
  const serpaExtra = ["item7","item8","item9","item10","item11","item12","item13"];
  const headers = [
    "participantCode","condition","conditionLabel","instrumentKey","instrumentLabel","submittedAt",
    "item1","item2","item3","item4","item5","item6",
    "item7","item8","item9","item10","item11","item12","item13",
    "moderate_total","vigorous_total","total_score",
    "gender","income","ethnicity","race","education","marital_status","employment",
  ];
  const lines = [headers.join(",")];
  for (const row of log) {
    const r = row.responses ?? {};
    const modItems = r.moderate_items ?? {};
    const vigItems = r.vigorous_items ?? {};
    lines.push([
      csvEscapeCell(deidentify(row)),
      csvEscapeCell(row.condition ?? ""),
      csvEscapeCell(row.conditionLabel ?? ""),
      csvEscapeCell(row.instrumentKey ?? ""),
      csvEscapeCell(row.instrumentLabel ?? ""),
      csvEscapeCell(row.submittedAt ?? ""),
      ...paseKeys.map(k => csvEscapeCell(r[k] ?? modItems[k] ?? "")),
      ...serpaExtra.map((k,i) => csvEscapeCell(r[k] ?? vigItems[`item${i+1}`] ?? "")),
      csvEscapeCell(r.moderate_total ?? ""),
      csvEscapeCell(r.vigorous_total ?? ""),
      csvEscapeCell(r.total_score ?? ""),
      csvEscapeCell(r.gender ?? ""),
      csvEscapeCell(r.income ?? ""),
      csvEscapeCell(r.ethnicity ?? ""),
      csvEscapeCell(r.race ?? ""),
      csvEscapeCell(r.education ?? ""),
      csvEscapeCell(r.marital_status ?? ""),
      csvEscapeCell(r.employment ?? ""),
    ].join(","));
  }
  return lines.join("\r\n");
}

/** Parse JSON object after [INSTRUMENT_DATA: … ] in assistant text (balanced braces). */
function extractInstrumentJson(text) {
  const marker = "[INSTRUMENT_DATA:";
  const i = text.indexOf(marker);
  if (i === -1) return null;
  const start = text.indexOf("{", i);
  if (start === -1) return null;
  let depth = 0;
  for (let j = start; j < text.length; j++) {
    if (text[j] === "{") depth++;
    else if (text[j] === "}") {
      depth--;
      if (depth === 0) {
        try {
          return JSON.parse(text.slice(start, j + 1));
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

/** Parse a [QUIZ: {...}] marker from an AI response (balanced-brace JSON).
 *  Returns { quiz, cleanText } — quiz is null if no marker found. */
function parseQuizFromMessage(text) {
  const marker = "[QUIZ:";
  const i = text.indexOf(marker);
  if (i === -1) return { quiz: null, cleanText: text };
  const start = text.indexOf("{", i);
  if (start === -1) return { quiz: null, cleanText: text };
  let depth = 0;
  for (let j = start; j < text.length; j++) {
    if (text[j] === "{") depth++;
    else if (text[j] === "}") {
      depth--;
      if (depth === 0) {
        const closeBracket = text.indexOf("]", j);
        if (closeBracket === -1) return { quiz: null, cleanText: text };
        try {
          const quiz = JSON.parse(text.slice(start, j + 1));
          const tagEnd = closeBracket + 1;
          const cleanText = (text.slice(0, i) + text.slice(tagEnd)).replace(/\n{3,}/g, "\n\n").trim();
          return { quiz, cleanText };
        } catch {
          return { quiz: null, cleanText: text };
        }
      }
    }
  }
  return { quiz: null, cleanText: text };
}

const DEFAULT_SCALE_QUIZ = {
  type: "scale",
  min: 0,
  max: 4,
  labels: ["No Confidence", "Low", "Moderate", "High", "Complete Confidence"],
};

/** If the model forgets [QUIZ:], infer the widget from instrument + question text (assessment mode only). */
function inferAssessmentQuiz(text, instrumentKey) {
  if (!instrumentKey || !text?.trim()) return null;
  const t = text.trim();
  const lower = t.toLowerCase();
  if (lower.includes("[instrument_data:") || extractInstrumentJson(t)) return null;

  if (instrumentKey === "inst1") {
    if (lower.includes("annual income") || lower.includes("household's current")) return null;
    if (/\bgender\b/.test(lower)) return { type: "choice", options: ["Female", "Male"] };
    if (/\bethnicity\b/.test(lower)) return { type: "choice", options: ["Hispanic or Latino", "Not Hispanic or Latino"] };
    if (/\brace\b/.test(lower)) return { type: "choice", options: ["American Indian or Alaska Native", "Asian", "Black or African American", "Native Hawaiian or Other Pacific Islander", "White", "More than one race"] };
    if (lower.includes("education")) return { type: "choice", options: ["Grammar School", "High School or equivalent", "Vocational/Technical School (2 year)", "Some College", "College Graduate (4 year)", "Master's Degree (MS)", "Doctoral Degree (PhD)", "Professional Degree (MD, JD, etc.)", "Other"] };
    if (lower.includes("marital")) return { type: "choice", options: ["Divorced", "Living with partner", "Married", "Separated", "Single", "Widowed"] };
    if (lower.includes("employment")) return { type: "choice", options: ["Full Time", "Part Time", "Retired", "Unemployed"] };
    return null;
  }

  if (["inst2", "inst3", "inst4", "inst5", "inst6"].includes(instrumentKey)) {
    const isConfidenceItem =
      lower.includes("how confident") ||
      (/\?\s*$/.test(t) && (lower.includes("next week") || lower.includes("minutes") || lower.includes("when ")));
    if (isConfidenceItem) return DEFAULT_SCALE_QUIZ;
  }
  return null;
}

async function persistInstrumentSubmission({ instrumentKey, instrumentLabel, responses, patient, condition, conditionLabel }) {
  const record = {
    submittedAt: new Date().toISOString(),
    participantId: patient.id,
    participantName: patient.name,
    condition: condition ?? "",
    conditionLabel: conditionLabel ?? "",
    instrumentKey,
    instrumentLabel,
    responses,
  };
  try {
    const prev = readInstrumentLog();
    prev.push(record);
    localStorage.setItem(INSTRUMENT_LOG_KEY, JSON.stringify(prev));
  } catch (e) {
    console.warn("Instrument localStorage log failed:", e);
  }
  // Also update per-profile score cache so PA Coach can inject scores into prompts.
  writeProfileScore(patient.id, instrumentKey, responses);
  if (!GOOGLE_SHEETS_WEBAPP_URL) return;
  try {
    await fetch(GOOGLE_SHEETS_WEBAPP_URL, {
      method: "POST",
      mode: "no-cors",
      headers: { "Content-Type": "text/plain;charset=utf-8" },
      body: JSON.stringify(record),
    });
  } catch (e) {
    console.warn("Google Sheet POST failed:", e);
  }
}

// ─── Per-profile instrument score cache ────────────────────────
// Scores are written here whenever an instrument completes, then
// injected into the PA Coach system prompt so the AI can personalize
// coaching based on real collected data (the "real-time learning" mechanism).

function profileScoresKey(profileId) {
  return `confidentMoves_scores_${profileId}`;
}

function readProfileScores(profileId) {
  try {
    const raw = localStorage.getItem(profileScoresKey(profileId));
    if (!raw) return {};
    return JSON.parse(raw) ?? {};
  } catch { return {}; }
}

function writeProfileScore(profileId, instrumentKey, responses) {
  try {
    const existing = readProfileScores(profileId);
    existing[instrumentKey] = { ...responses, _savedAt: new Date().toISOString() };
    localStorage.setItem(profileScoresKey(profileId), JSON.stringify(existing));
  } catch (e) {
    console.warn("Profile score cache write failed:", e);
  }
}

/** Format collected instrument scores into a concise coaching-context block.
 *  Injected into the PA Coach system prompt (Conditions B/C) so the AI
 *  can reference real assessment data when coaching.
 *
 *  INJECTION STRATEGY = STRUCTURED (deterministic) INJECTION.
 *  We compute self-efficacy tiers and a goal-setting anchor in code (not by the
 *  model) and hand the AI a compact, already-interpreted summary. This is the
 *  right approach for patient scores because it is reproducible across arms,
 *  auditable for the trial, and cheap on tokens — versus asking the model to
 *  reason over raw item numbers (error-prone) or retrieving them via a tool. */
function buildScoreContextBlock(scores) {
  if (!scores || Object.keys(scores).length === 0) return "";

  const paseRange = (n) => n <= 8 ? "Low (0–8)" : n <= 16 ? "Moderate (9–16)" : "High (17–24)";
  const paseTier = (n) => n <= 8 ? "low" : n <= 16 ? "moderate" : "high";

  const serpaItemLabels = [
    "bad weather", "boredom with available activities", "vacation",
    "uninterested in available activities", "physical discomfort while active",
    "exercising alone", "not enjoying available activities",
    "difficult location access", "not liking available activities",
    "schedule conflicts", "self-consciousness about appearance",
    "no encouragement", "personal stress",
  ];

  const lines = [];

  if (scores.inst1) {
    const d = scores.inst1;
    const parts = [d.gender, d.education, d.employment, d.marital_status].filter(Boolean);
    if (parts.length) lines.push(`Demographics: ${parts.join(", ")}.`);
  }

  // Track each PA-domain self-efficacy score to pick a goal-setting anchor.
  const domains = [];

  const paseMap = [
    ["inst2", "J-R PASE (job-related PA SE)", "job/work PA"],
    ["inst3", "T-R PASE (transport PA SE)", "transport/active-commute PA"],
    ["inst4", "D-R PASE (domestic PA SE)", "home/domestic PA"],
  ];
  for (const [key, label, plain] of paseMap) {
    if (!scores[key]) continue;
    const d = scores[key];
    const tot = d.total_score ?? 0;
    lines.push(`${label}: ${tot}/24 — ${paseRange(tot)}.`);
    domains.push({ plain, score: tot, tier: paseTier(tot) });
  }

  if (scores.inst5) {
    const d = scores.inst5;
    const mt = d.moderate_total ?? 0;
    const vt = d.vigorous_total ?? 0;
    lines.push(`L-R PASE (leisure PA SE): moderate ${mt}/24 (${paseRange(mt)}), vigorous ${vt}/24 (${paseRange(vt)}).`);
    domains.push({ plain: "leisure/recreational PA", score: mt, tier: paseTier(mt) });
  }

  if (scores.inst6) {
    const d = scores.inst6;
    const tot = d.total_score ?? 0;
    const serpaTier = tot <= 17 ? "Low confidence (high barrier burden)" : tot <= 34 ? "Moderate" : "High confidence (low barriers)";
    const critical = serpaItemLabels
      .map((name, i) => ({ name, score: d[`item${i + 1}`] ?? 4 }))
      .filter(x => x.score <= 1)
      .map(x => x.name);
    lines.push(`SERPA (barrier SE): ${tot}/52 — ${serpaTier}.${critical.length ? ` Specific barriers (score ≤1): ${critical.join("; ")}.` : ""}`);
  }

  if (lines.length === 0) return "";

  // Goal-setting anchor (SE theory / mastery): start where confidence is highest,
  // then generalize; direct problem-solving toward the lowest-confidence domain.
  let anchorLine = "";
  if (domains.length) {
    const sorted = [...domains].sort((a, b) => b.score - a.score);
    const strongest = sorted[0];
    const weakest = sorted[sorted.length - 1];
    const overall = Math.round(domains.reduce((s, d) => s + d.score, 0) / domains.length);
    anchorLine = `\nSelf-efficacy snapshot: overall ${paseTier(overall)} (avg ${overall}/24). Highest-confidence domain: ${strongest.plain} (${strongest.tier}). Lowest-confidence domain: ${weakest.plain} (${weakest.tier}).`;
  }

  return `

PATIENT ASSESSMENT DATA — COLLECTED (inject into coaching; do not read raw numbers aloud unless asked):
${lines.join("\n")}${anchorLine}
→ Coaching focus: build first from the highest-confidence PA domain (a mastery win the patient already believes they can do), then extend that success toward lower-confidence domains and named barriers. Use SE theory sources (mastery, vicarious, verbal persuasion, affective) and PACE (partnership, acceptance, compassion, empowerment). Any goal-setting should begin small in a high-confidence domain and only add difficulty as confidence rises (confidence ≥7/10).`;
}

// ─── Conversation store ────────────────────────────────────────
const CONV_STORE_KEY = "confidentMoves_conversations";

function readConvStore() {
  try {
    const raw = localStorage.getItem(CONV_STORE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function writeConvStore(store) {
  try {
    localStorage.setItem(CONV_STORE_KEY, JSON.stringify(store));
  } catch (e) {
    console.warn("Conversation store write failed:", e);
  }
}

function makeConversationKey(moduleId, profileId, conditionId) {
  return `${moduleId}_${profileId}_${conditionId}`;
}

function saveConversation(conversationKey, moduleId, moduleLabel, messages, meta = {}) {
  const store = readConvStore();
  store[conversationKey] = {
    conversationKey,
    moduleId,
    moduleLabel,
    profileId: meta.profileId ?? "",
    profileName: meta.profileName ?? "",
    condition: meta.condition ?? "",
    conditionLabel: meta.conditionLabel ?? "",
    messages,
    lastUpdated: new Date().toISOString(),
  };
  writeConvStore(store);
}

function clearConversation(moduleId) {
  const store = readConvStore();
  delete store[moduleId];
  writeConvStore(store);
}

function convStoreToCsv(store) {
  // One row per turn. participantCode only — no real names in research exports.
  const headers = ["participantCode","condition","conditionLabel","moduleId","moduleLabel","lastUpdated","turnNumber","role","timestamp","message","referenceLookups"];
  const lines = [headers.join(",")];
  for (const entry of Object.values(store)) {
    entry.messages.forEach((msg, idx) => {
      lines.push([
        csvEscapeCell(entry.profileId ?? ""),
        csvEscapeCell(entry.condition ?? ""),
        csvEscapeCell(entry.conditionLabel ?? ""),
        csvEscapeCell(entry.moduleId ?? ""),
        csvEscapeCell(entry.moduleLabel ?? ""),
        csvEscapeCell(entry.lastUpdated ?? ""),
        csvEscapeCell(idx + 1),
        csvEscapeCell(msg.role ?? ""),
        csvEscapeCell(msg.ts ?? ""),
        csvEscapeCell(msg.content ?? ""),
        csvEscapeCell(formatReferenceLookups(msg.referenceLookups)),
      ].join(","));
    });
  }
  return lines.join("\r\n");
}

/** PASE scores summary — one row per participant.
 *  Non-technical friendly: open in Excel, each score in its own column with plain-English level. */
function paseScoresSummaryCsv() {
  const profiles = readProfiles();
  const paseLevel = n => n == null ? "" : n <= 8 ? "Low" : n <= 16 ? "Moderate" : "High";
  const serpaLevel = n => n == null ? "" : n <= 17 ? "Low confidence" : n <= 34 ? "Moderate" : "High confidence";
  const serpaBarrierNames = [
    "bad weather","boredom","vacation","uninterested","physical discomfort",
    "alone","don't enjoy activities","location access","don't like activities",
    "schedule conflicts","self-consciousness","no encouragement","personal stress",
  ];
  const headers = [
    "participantCode",
    "jrPase_score","jrPase_level",
    "trPase_score","trPase_level",
    "drPase_score","drPase_level",
    "lrPase_moderate_score","lrPase_moderate_level",
    "lrPase_vigorous_score","lrPase_vigorous_level",
    "serpa_score","serpa_level","serpa_criticalBarriers",
    "collectedAt",
  ];
  const lines = [headers.join(",")];
  for (const p of profiles) {
    const s = readProfileScores(p.id);
    if (!Object.keys(s).filter(k => !k.startsWith("_")).length) continue;
    const jr = s.inst2?.total_score ?? null;
    const tr = s.inst3?.total_score ?? null;
    const dr = s.inst4?.total_score ?? null;
    const lrM = s.inst5?.moderate_total ?? null;
    const lrV = s.inst5?.vigorous_total ?? null;
    const serp = s.inst6?.total_score ?? null;
    const critBarriers = serpaBarrierNames
      .map((name, i) => ({ name, score: s.inst6?.[`item${i+1}`] ?? 4 }))
      .filter(x => x.score <= 1).map(x => x.name).join("; ");
    lines.push([
      csvEscapeCell(p.id),
      csvEscapeCell(jr ?? ""), csvEscapeCell(paseLevel(jr)),
      csvEscapeCell(tr ?? ""), csvEscapeCell(paseLevel(tr)),
      csvEscapeCell(dr ?? ""), csvEscapeCell(paseLevel(dr)),
      csvEscapeCell(lrM ?? ""), csvEscapeCell(paseLevel(lrM)),
      csvEscapeCell(lrV ?? ""), csvEscapeCell(paseLevel(lrV)),
      csvEscapeCell(serp ?? ""), csvEscapeCell(serpaLevel(serp)),
      csvEscapeCell(critBarriers),
      csvEscapeCell(s.inst2?._savedAt ?? s.inst6?._savedAt ?? ""),
    ].join(","));
  }
  return lines.join("\r\n");
}

/** Benchmark CSV — one row per scored AI response.
 *  Has blank "human_*" columns so coders can fill in scores directly in Excel.
 *  Export → open in Excel → share with RA → calculate kappa.
 *  DISABLED — LLM-as-judge not in use yet; restore with judge block below. */
/*
function benchmarkCsv() {
  const judgeResults = readJudgeResults();
  const headers = [
    "scoredAt","participantCode","condition","module",
    "llm_total","llm_max",
    "llm_open_question","llm_affirm","llm_reflect_summary","llm_se_source","llm_bct","llm_personalization",
    "llm_rationale","responsePreview",
    "human_open_question","human_affirm","human_reflect_summary","human_se_source","human_bct","human_personalization",
    "human_total","coder_initials","coding_notes",
  ];
  const lines = [headers.join(",")];
  for (const r of judgeResults) {
    lines.push([
      csvEscapeCell(r.scoredAt ?? ""),
      csvEscapeCell(r.condition ? `[coded]` : ""),
      csvEscapeCell(r.condition ?? ""),
      csvEscapeCell(r.moduleLabel ?? ""),
      csvEscapeCell(r.total ?? ""),
      csvEscapeCell(r.maxTotal ?? 18),
      csvEscapeCell(r.scores?.open_question ?? ""),
      csvEscapeCell(r.scores?.affirm ?? ""),
      csvEscapeCell(r.scores?.reflect_summary ?? ""),
      csvEscapeCell(r.scores?.se_source ?? ""),
      csvEscapeCell(r.scores?.bct ?? ""),
      csvEscapeCell(r.scores?.personalization ?? ""),
      csvEscapeCell(r.rationale ?? ""),
      csvEscapeCell((r.messagePreview ?? "").slice(0, 200)),
      "","","","","","","","","",
    ].join(","));
  }
  return lines.join("\r\n");
}
*/

/** Push de-identified conversation turns to Google Sheets (if configured). */
async function pushConversationToSheet(entry) {
  if (!GOOGLE_SHEETS_WEBAPP_URL || !entry?.messages?.length) return;
  try {
    await fetch(GOOGLE_SHEETS_WEBAPP_URL, {
      method: "POST", mode: "no-cors",
      headers: { "Content-Type": "text/plain;charset=utf-8" },
      body: JSON.stringify({
        type: "conversation",
        participantCode: entry.profileId ?? "",
        condition: entry.condition ?? "",
        moduleId: entry.moduleId ?? "",
        rows: entry.messages.map((m, i) => ({
          turn: i + 1, role: m.role, ts: m.ts ?? "", message: m.content ?? "",
        })),
      }),
    });
  } catch (e) { console.warn("Sheet conversation push failed:", e); }
}

// ─── Theme tokens ──────────────────────────────────────────────
const T = {
  teal: "#0f766e",       // clinical teal — primary actions / brand
  tealLight: "#ccfbf1",
  tealMid: "#14b8a6",
  tealDark: "#115e59",
  purple: "#3730a3",     // deep indigo — secondary accent (trial / highlights)
  purpleLight: "#e0e7ff",
  amber: "#b45309",
  amberLight: "#fef3c7",
  red: "#dc2626",
  redLight: "#fee2e2",
  gray50: "#f8fafc",
  gray100: "#f1f5f9",
  gray200: "#e2e8f0",
  gray300: "#cbd5e1",
  gray400: "#94a3b8",
  gray500: "#64748b",
  gray600: "#475569",
  gray700: "#334155",
  gray800: "#1e293b",
  gray900: "#0f172a",
  // Chat — distinct assistant vs participant, calm contrast (no loud fills)
  chatAiBubble: "#f1f5f9",
  chatAiBubbleBorder: "#e2e8f0",
  chatAiText: "#1e293b",
  chatAiAvatarBg: "#e2e8f0",
  chatAiAvatarFg: "#475569",
  chatUserBubble: "#334155",
  chatUserText: "#f8fafc",
  chatUserAvatarBg: "#cbd5e1",
  chatUserAvatarFg: "#1e293b",
};

// ─── Research profiles & conditions ───────────────────────────
const PROFILES_STORAGE_KEY = "confidentMoves_profiles_v1";
const ACTIVE_PROFILE_KEY = "confidentMoves_active_profile";
const ACTIVE_CONDITION_KEY = "confidentMoves_active_condition";
const LEGACY_PROGRAM_STATE_KEY = "confidentMoves_program_v1";

const ALL_INSTRUMENT_KEYS = ["inst1", "inst2", "inst3", "inst4", "inst5", "inst6"];

const CONDITIONS = [
  { id: "A", label: "Nothing", description: "Baseline — no behavior-change theory, no instruments in prompt", instruments: [], includeTheory: false },
  { id: "B", label: "Theories only", description: "Self-efficacy (Bandura) + MI/PACE + BCT coaching — no instruments in prompt", instruments: [], includeTheory: true },
  { id: "C", label: "Theories + instruments", description: "Self-efficacy + MI/PACE + BCT coaching + all assessment instruments in the prompt at once", instruments: ALL_INSTRUMENT_KEYS, includeTheory: true },
];

const INSTRUMENT_LABELS = {
  inst1: "Demographic Information",
  inst2: "Instrument 2a — Job-Related PA Self-Efficacy (J-R PASE)",
  inst3: "Instrument 2b — Transportation-Related PA Self-Efficacy (T-R PASE)",
  inst4: "Instrument 2c — Domestic-Related PA Self-Efficacy (D-R PASE)",
  inst5: "Instrument 2d — Leisure-Related PA Self-Efficacy (L-R PASE)",
  inst6: "Barrier Self-Efficacy (SERPA)",
};

const INSTRUMENT_TEXTS = {
  inst1: `INSTRUMENT ITEMS (ask in this order)
1. What is your gender?
Female
Male
2. Please estimate your household's current annual income.
________________
3. What is your ethnicity?
Hispanic or Latino
Not Hispanic or Latino

4. What is your race?
American Indian or Alaska Native
Asian
Black or African American
Native Hawaiian or Other Pacific Islander
White
More than one race

5. Please indicate the highest level of education you have completed.
Grammar School
High School or equivalent
Vocational/Technical School (2 year)
Some College
College Graduate (4 year)
Master's Degree (MS)
Doctoral Degree (PhD)
Professional Degree (MD, JD, etc.)
Other

6. What is your current marital status?
Divorced
Living with partner
Married
Separated
Single
Widowed

7. What is your employment status?
Full Time
Part Time
Retired
Unemployed`,
  inst2: 
  `JOB-RELATED PHYSICAL ACTIVITY SELF-EFFICACY (J-R PASE)

Think about how confident you are in your current ability 
to engage in job-related physical activity at a moderate 
level of intensity in the next week.

Job-related refers to paid jobs, farming, volunteer work, 
coursework, and any other unpaid work done outside the home.
Physical activity refers to any physical activities done 
for at least 10 minutes at a time.
Moderate intensity means activities that take moderate 
physical effort and make you breathe somewhat harder 
than normal (e.g., work requiring moderate lifting or walking).

Scale: 0=No Confidence, 1=Low, 2=Moderate, 3=High, 4=Complete Confidence

ADMINISTRATION RULE: If patient answers 0 (No Confidence) 
for any item, do not ask further items. Record 0 for 
all remaining items.

Items (ask one at a time):
1. How confident are you in your ability to engage in 
   job-related PA at moderate intensity for at least 
   10 minutes in the next week? (0-4)

2. ...for at least 30 minutes in the next week? (0-4)

3. ...for at least 60 minutes in the next week? (0-4)

4. ...for at least 90 minutes in the next week? (0-4)

5. ...for at least 120 minutes in the next week? (0-4)

6. ...for at least 150 minutes in the next week? (0-4)

SCORING: Sum all answered items. Higher = greater SE for job-related PA.
Maximum = 24 (4 points × 6 items).
Score of 0 on item 1 = very low SE even for minimal job-related PA.

COACHING INTERPRETATION:
Score 0-8 (Low): Patient has low confidence for job-related PA.
  → Focus on identifying existing movement in work context
  → Micro-goals: notice and extend current work movement
  → Never suggest adding new activity at work yet

Score 9-16 (Moderate): Some confidence.
  → Collaborative goal-setting around work-based activity
  → Action planning: when/where during workday

Score 17-24 (High): Strong confidence.
  → Work is an SE strength — build from here
  → Can set more ambitious work-related PA goals

After all items complete, output:
[INSTRUMENT_DATA: {"instrument": "J-R PASE", "item1": , "item2": , "item3": , "item4": , "item5": , "item6": , "total_score": }]`,

  inst3: `TRANSPORTATION-RELATED PHYSICAL ACTIVITY SELF-EFFICACY (T-R PASE)

Think about how confident you are in your current ability 
to engage in transportation-related physical activity at 
a moderate level of intensity in the next week.

Transportation-related refers to how you travel from place 
to place — to work, stores, movies, and so on.
Physical activity refers to any physical activities done 
for at least 10 minutes at a time.
Moderate intensity means activities that take moderate 
physical effort and make you breathe somewhat harder 
than normal (e.g., walking or moderate cycling as transport).

Scale: 0=No Confidence, 1=Low, 2=Moderate, 3=High, 4=Complete Confidence

ADMINISTRATION RULE: If patient answers 0 (No Confidence) 
for any item, do not ask further items.

Items (ask one at a time):
1. How confident are you in your ability to engage in 
   transportation-related PA at moderate intensity for 
   at least 10 minutes in the next week? (0-4)

2. ...for at least 30 minutes in the next week? (0-4)

3. ...for at least 60 minutes in the next week? (0-4)

4. ...for at least 90 minutes in the next week? (0-4)

5. ...for at least 120 minutes in the next week? (0-4)

6. ...for at least 150 minutes in the next week? (0-4)

SCORING: Sum all answered items. Maximum = 24.

COACHING INTERPRETATION:
Score 0-8 (Low): Patient has low confidence walking/cycling as transport.
  → Explore current transportation patterns
  → Identify one regular destination where a short walk is possible
  → Parking further away, one transit stop early — both count
  → Micro-goal: add 5-10 minutes of walking to one regular trip

Score 9-16 (Moderate):
  → Action plan around specific regular trips
  → Which destinations, which days, how far

Score 17-24 (High): Transportation is an SE strength.
  → Build on existing active transport habits
  → Extend duration or add new destinations

After all items complete, output:
[INSTRUMENT_DATA: {"instrument": "T-R PASE", "item1": , "item2": , "item3": , "item4": , "item5": , "item6": , "total_score": }]`,

  inst4: `DOMESTIC-RELATED PHYSICAL ACTIVITY SELF-EFFICACY (D-R PASE)

Think about how confident you are in your current ability 
to engage in domestic-related physical activity at a 
moderate level of intensity in the next week.

Domestic-related refers to physical activities done in 
and around your home — housework, gardening, yard work, 
general maintenance, and caring for your family.
Physical activity refers to any physical activities done 
for at least 10 minutes at a time.
Moderate intensity means activities that take moderate 
physical effort and make you breathe somewhat harder 
than normal (e.g., housework requiring moderate lifting or walking).

Scale: 0=No Confidence, 1=Low, 2=Moderate, 3=High, 4=Complete Confidence

ADMINISTRATION RULE: If patient answers 0 (No Confidence) 
for any item, do not ask further items.

Items (ask one at a time):
1. How confident are you in your ability to engage in 
   domestic-related PA at moderate intensity for at 
   least 10 minutes in the next week? (0-4)

2. ...for at least 30 minutes in the next week? (0-4)

3. ...for at least 60 minutes in the next week? (0-4)

4. ...for at least 90 minutes in the next week? (0-4)

5. ...for at least 120 minutes in the next week? (0-4)

6. ...for at least 150 minutes in the next week? (0-4)

SCORING: Sum all answered items. Maximum = 24.

COACHING INTERPRETATION:
Score 0-8 (Low): Low confidence for domestic PA.
  → Validate that housework, gardening, and family care 
    ARE real physical activity
  → Help patient recognize movement they may not count
  → Reframe existing domestic activity as PA achievement
  → Micro-goal: notice and extend one existing domestic activity

Score 9-16 (Moderate):
  → Set goals around intensifying or extending existing 
    domestic movement
  → Gardening, active housework, active play with children

Score 17-24 (High): Home is an SE strength.
  → Evening or weekend domestic activity as starting point
  → Connect home-based success to other PA domains

After all items complete, output:
[INSTRUMENT_DATA: {"instrument": "D-R PASE", "item1": , "item2": , "item3": , "item4": , "item5": , "item6": , "total_score": }]`,
  inst5: `LEISURE-RELATED PHYSICAL ACTIVITY SELF-EFFICACY (L-R PASE)

Think about how confident you are in your current ability 
to engage in leisure-related physical activity in the 
next week.

Leisure-related refers to physical activities done solely 
for recreation, sport, exercise, or leisure.
Physical activity refers to any physical activities done 
for at least 10 minutes at a time.

PART A — MODERATE INTENSITY
Moderate intensity: activities that take moderate physical 
effort and make you breathe somewhat harder than normal 
(e.g., leisure activities requiring moderate lifting or walking).

Scale: 0=No Confidence, 1=Low, 2=Moderate, 3=High, 4=Complete Confidence
ADMINISTRATION RULE: Stop if patient answers 0 on any item.

Items — Moderate (ask one at a time):
1. How confident are you in your ability to engage in 
   leisure PA at MODERATE intensity for at least 
   10 minutes in the next week? (0-4)

2. ...for at least 30 minutes? (0-4)

3. ...for at least 60 minutes? (0-4)

4. ...for at least 90 minutes? (0-4)

5. ...for at least 120 minutes? (0-4)

6. ...for at least 150 minutes? (0-4)

PART B — VIGOROUS INTENSITY
Vigorous intensity: activities that take hard physical 
effort and make you breathe much harder than normal 
(e.g., running, intense exercise).

Items — Vigorous (ask one at a time):
1. How confident are you in your ability to engage in 
   leisure PA at VIGOROUS intensity for at least 
   10 minutes in the next week? (0-4)

2. ...for at least 15 minutes? (0-4)

3. ...for at least 30 minutes? (0-4)

4. ...for at least 45 minutes? (0-4)

5. ...for at least 60 minutes? (0-4)

6. ...for at least 75 minutes? (0-4)

SCORING: 
Moderate subscale: sum items A1-A6. Maximum = 24.
Vigorous subscale: sum items B1-B6. Maximum = 24.

COACHING INTERPRETATION:
Low moderate score (0-8): 
  → Start with most accessible leisure activity: walking
  → 10-minute neighborhood walks, mall walking, park visits
  → Never suggest vigorous activity yet

Low vigorous only (moderate score 9-24, vigorous 0-8):
  → Patient ready for moderate leisure but not vigorous
  → Do NOT suggest running or intense exercise
  → Build mastery at moderate first
  → "Your confidence for moderate activity is strong — 
    let's keep building there before stepping up intensity"

High both scores:
  → Leisure is a major SE strength
  → Explore recreational activities patient enjoys
  → Connect past leisure activities to current goals

After all items complete, output:
[INSTRUMENT_DATA: {"instrument": "L-R PASE", "moderate_items": {"item1": , "item2": , "item3": , "item4": , "item5": , "item6": }, "vigorous_items": {"item1": , "item2": , "item3": , "item4": , "item5": , "item6": }, "moderate_total": , "vigorous_total": }]`,
  inst6: `SELF-EFFICACY TO REGULATE PHYSICAL ACTIVITY (SERPA)

Think about how confident you are in your current ability 
to overcome possible barriers to engagement in a 
recommended amount of weekly physical activity for health.

Recommended amounts include:
- At least 150 minutes per week of moderate PA, OR
- At least 75 minutes per week of vigorous PA, OR
- An equivalent combination of both

Moderate PA: activities like carrying light loads, 
raking, moderate cycling — breathing somewhat harder than normal.
Vigorous PA: activities like heavy lifting, chopping wood, 
intense cycling — breathing much harder than normal.

Scale: 0=No Confidence, 1=Low, 2=Moderate, 3=High, 4=Complete Confidence

Items (ask one at a time — do NOT skip any):
How confident are you in your current ability to engage 
in a recommended amount of weekly physical activity when...

1. The weather is very bad? (0-4)
2. You are bored with the physical activities available to you? (0-4)
3. You are on vacation? (0-4)
4. You are uninterested in the physical activities available? (0-4)
5. You feel physical discomfort while being physically active? (0-4)
6. You have to be physically active by yourself? (0-4)
7. You do not enjoy the physical activities available? (0-4)
8. It is difficult to get to a location suitable for PA? (0-4)
9. You do not like the physical activities available? (0-4)
10. Your schedule conflicts with being physically active? (0-4)
11. You feel self-conscious about your appearance while being 
    physically active? (0-4)
12. You do not receive any encouragement for being physically 
    active? (0-4)
13. You are under personal stress? (0-4)

SCORING: Sum all 13 items. Maximum = 52.
Higher scores = greater confidence to overcome barriers.

COACHING INTERPRETATION BY ITEM:
Item 1 (weather, score 0-1): 
  → Major barrier. Explore indoor alternatives, home-based options.

Items 2, 4, 7, 9 (boredom/interest, score 0-1):
  → Patient finds available activities unappealing.
  → Explore what they have ever found tolerable.
  → Connect to higher-scoring PA domain.

Item 5 (physical discomfort, score 0-1):
  → CRITICAL: validate discomfort first. Never push through.
  → Lower intensity and shorter duration goals only.
  → Chair-based or gentle movement options.

Item 6 (alone, score 0-1):
  → Lack of social support undermines SE.
  → Explore social options. If none: normalize solo activity.

Item 8 (access, score 0-1):
  → Remove location dependency. Home-based first.
  → Transportation PA as alternative.

Item 10 (schedule, score 0-1):
  → CRITICAL: time barrier. Explore actual vs perceived time.
  → Identify schedule anchors. 10 minutes is enough.

Item 11 (self-consciousness, score 0-1):
  → CRITICAL for obesity population. Validate without judgment.
  → Home-based or low-visibility options first.
  → NEVER comment on appearance or weight.

Item 12 (no encouragement, score 0-1):
  → Your coaching role is to provide social persuasion SE source.
  → Provide specific genuine affirmation in every session.

Item 13 (stress, score 0-1):
  → Validate stress fully first.
  → Survival mode goals: smallest possible commitment.

After all items complete, output:
[INSTRUMENT_DATA: {"instrument": "SERPA", "item1": , "item2": , "item3": , "item4": , "item5": , "item6": , "item7": , "item8": , "item9": , "item10": , "item11": , "item12": , "item13": , "total_score": }]`,
};

const INSTRUMENT_PROMPT_SUMMARIES = {
  inst1: "Demographics context only: gender, income, ethnicity, race, education, marital status, employment. Use for context tailoring only.",
  inst2: "J-R PASE (job-related self-efficacy): 6 items, score 0-24. Higher means greater confidence for work/volunteer/course-related activity.",
  inst3: "T-R PASE (transport self-efficacy): 6 items, score 0-24. Higher means greater confidence for active transport (walking/cycling).",
  inst4: "D-R PASE (domestic self-efficacy): 6 items, score 0-24. Higher means greater confidence for home/family physical activity.",
  inst5: "L-R PASE (leisure self-efficacy): 12 items with moderate/vigorous subdomains. Use profile (moderate vs vigorous confidence) to set intensity.",
  inst6: "SERPA (barrier self-efficacy): 13 items, score 0-52. Lower scores = lower confidence to overcome barriers. Items with score ≤1 are critical targets: schedule conflicts (item 10), self-consciousness (item 11), stress (item 13), physical discomfort (item 5).",
};

const COACHING_SYSTEM_KEYS = ["chat", "checkin", "education"];

// ─── Instrument assessment (quiz-widget) infrastructure ──────────────────────

const QUIZ_SCALE_TAG = '[QUIZ: {"type":"scale","min":0,"max":4,"labels":["No Confidence","Low","Moderate","High","Complete Confidence"]}]';

const INSTRUMENT_SHORT_LABELS = {
  inst1: "Demographics",
  inst2: "2a · J-R PASE",
  inst3: "2b · T-R PASE",
  inst4: "2c · D-R PASE",
  inst5: "2d · L-R PASE",
  inst6: "SERPA",
};

/** Four dedicated PASE assessment agents — one isolated AI per instrument (advisor design). */
const PASE_INSTRUMENT_AGENTS = [
  {
    key: "inst2",
    code: "2a",
    label: "Instrument 2a — Job-Related PA Self-Efficacy (J-R PASE)",
    shortLabel: "2a · Job",
    aiName: "Job Activity SE Assistant",
    aiInitials: "JA",
    domainFocus: "job-related physical activity only (paid work, volunteer work, coursework, unpaid work outside the home)",
  },
  {
    key: "inst3",
    code: "2b",
    label: "Instrument 2b — Transportation-Related PA Self-Efficacy (T-R PASE)",
    shortLabel: "2b · Transport",
    aiName: "Transport Activity SE Assistant",
    aiInitials: "TR",
    domainFocus: "transportation-related physical activity only (walking or cycling to get from place to place)",
  },
  {
    key: "inst4",
    code: "2c",
    label: "Instrument 2c — Domestic-Related PA Self-Efficacy (D-R PASE)",
    shortLabel: "2c · Domestic",
    aiName: "Domestic Activity SE Assistant",
    aiInitials: "DM",
    domainFocus: "domestic-related physical activity only (housework, gardening, yard work, caring for family at home)",
  },
  {
    key: "inst5",
    code: "2d",
    label: "Instrument 2d — Leisure-Related PA Self-Efficacy (L-R PASE)",
    shortLabel: "2d · Leisure",
    aiName: "Leisure Activity SE Assistant",
    aiInitials: "LR",
    domainFocus: "leisure-related physical activity only (recreation, sport, exercise for enjoyment)",
  },
];

function getPaseAgent(instrumentKey) {
  return PASE_INSTRUMENT_AGENTS.find(a => a.key === instrumentKey) ?? null;
}

const PASE_ITEM_COUNTS = { inst2: 6, inst3: 6, inst4: 6, inst5: 12 };

/** Where all research data lives in this browser (localStorage). */
const DATA_STORAGE_MAP = [
  { key: "confidentMoves_conversations", label: "Conversation logs", desc: "Every chat turn, per module × profile × condition" },
  { key: "confidentMoves_instrument_log", label: "Instrument submissions", desc: "Completed [INSTRUMENT_DATA] JSON records with timestamps" },
  { key: "confidentMoves_scores_{profileId}", label: "Per-patient score cache", desc: "Latest PASE/SERPA scores injected into PA Coach (one key per profile)" },
  { key: "confidentMoves_profiles_v1", label: "Mock profiles", desc: "Editable participant demographics and program fields" },
  { key: "confidentMoves_program_{profileId}", label: "Program state", desc: "Program day, weight, PA minutes per profile" },
  { key: "confidentMoves_api_metrics", label: "API feasibility metrics", desc: "Call count and rate-limit hits for Aim 3 reporting" },
  { key: "confidentMoves_active_profile", label: "UI state", desc: "Active profile, condition, tab (not exportable research data)" },
];

function countAssessmentAnswers(convStore, moduleId, profileId) {
  let maxAnswered = 0;
  for (const cond of ["A", "B", "C"]) {
    const ck = makeConversationKey(moduleId, profileId, cond);
    const entry = convStore[ck];
    if (!entry?.messages) continue;
    const n = entry.messages.filter(m =>
      m.role === "user" && !/^▶\s*Begin/i.test(String(m.content).trim())
    ).length;
    maxAnswered = Math.max(maxAnswered, n);
  }
  return maxAnswered;
}

function isInstrumentComplete(profileId, instrumentKey, scores, instrumentLog) {
  const s = scores[instrumentKey];
  if (s) {
    if (instrumentKey === "inst5") return s.moderate_total != null || s.vigorous_total != null;
    if (instrumentKey === "inst1") return Boolean(s.gender || s.employment);
    if (instrumentKey === "inst6") return s.total_score != null;
    return s.total_score != null;
  }
  return instrumentLog.some(r => r.participantId === profileId && r.instrumentKey === instrumentKey);
}

function buildFeasibilityReport() {
  const convStore = readConvStore();
  const instrumentLog = readInstrumentLog();
  const apiMetrics = readApiMetrics();
  const profiles = readProfiles();

  const totalUserTurns = Object.values(convStore).reduce(
    (acc, e) => acc + (e.messages?.filter(m => m.role === "user").length ?? 0), 0,
  );
  const totalAssistantTurns = Object.values(convStore).reduce(
    (acc, e) => acc + (e.messages?.filter(m => m.role === "assistant").length ?? 0), 0,
  );

  const profileReports = profiles.map(p => {
    const scores = readProfileScores(p.id);
    const paseAgents = PASE_INSTRUMENT_AGENTS.map(agent => {
      const expected = PASE_ITEM_COUNTS[agent.key] ?? 6;
      const moduleId = `assess_${agent.code}`;
      const answered = countAssessmentAnswers(convStore, moduleId, p.id);
      const complete = isInstrumentComplete(p.id, agent.key, scores, instrumentLog);
      const saved = scores[agent.key];
      const totalScore = saved?.total_score ?? saved?.moderate_total ?? null;
      let status = "not_started";
      if (complete) status = "complete";
      else if (answered > 0) status = "in_progress";

      return {
        code: agent.code,
        key: agent.key,
        label: agent.label,
        shortLabel: agent.shortLabel,
        status,
        answered,
        expected,
        completionPct: complete ? 100 : Math.min(99, Math.round((answered / expected) * 100)),
        totalScore,
        savedAt: saved?._savedAt ?? null,
      };
    });

    const serpaComplete = isInstrumentComplete(p.id, "inst6", scores, instrumentLog);
    const serpaAnswered = countAssessmentAnswers(convStore, "assess_inst6", p.id);

    return {
      profileId: p.id,
      profileName: p.name,
      paseAgents,
      serpa: { complete: serpaComplete, answered: serpaAnswered, expected: 13, totalScore: scores.inst6?.total_score ?? null },
      scoresCollected: Object.keys(scores).filter(k => !k.startsWith("_")),
    };
  });

  return {
    generatedAt: new Date().toISOString(),
    // PRODUCTION: use active LLM model id from server config or last /api/chat response metadata
    //   (e.g. claude-haiku-4-5-20251001), not GROQ_MODEL hardcoded.
    model: GROQ_MODEL,
    apiMetrics,
    conversationLogs: Object.keys(convStore).length,
    totalUserTurns,
    totalAssistantTurns,
    estimatedApiCalls: apiMetrics.totalCalls || totalUserTurns,
    instrumentSubmissions: instrumentLog.length,
    profileReports,
    storage: DATA_STORAGE_MAP,
  };
}

// Trimmed administration-only texts; no coaching interpretation, just items + scale + QUIZ tags.
const INSTRUMENT_ADMIN_TEXTS = {
  inst1: `DEMOGRAPHICS — ask each item one at a time, in order.
For items with fixed answer options, append the [QUIZ: ...] tag shown on the very last line of your response.
For item 2 (income) there is no tag — the user types freely.
1. What is your gender?
[QUIZ: {"type":"choice","options":["Female","Male"]}]
2. Please estimate your household's current annual income. (free text — no quiz tag)
3. What is your ethnicity?
[QUIZ: {"type":"choice","options":["Hispanic or Latino","Not Hispanic or Latino"]}]
4. What is your race?
[QUIZ: {"type":"choice","options":["American Indian or Alaska Native","Asian","Black or African American","Native Hawaiian or Other Pacific Islander","White","More than one race"]}]
5. Highest level of education completed?
[QUIZ: {"type":"choice","options":["Grammar School","High School or equivalent","Vocational/Technical School (2 year)","Some College","College Graduate (4 year)","Master's Degree (MS)","Doctoral Degree (PhD)","Professional Degree (MD, JD, etc.)","Other"]}]
6. Current marital status?
[QUIZ: {"type":"choice","options":["Divorced","Living with partner","Married","Separated","Single","Widowed"]}]
7. Current employment status?
[QUIZ: {"type":"choice","options":["Full Time","Part Time","Retired","Unemployed"]}]
After all 7 items: [INSTRUMENT_DATA: {"instrument":"Demographics","gender":"","income":"","ethnicity":"","race":"","education":"","marital_status":"","employment":""}]`,

  inst2: `J-R PASE — Job-Related PA Self-Efficacy.
Context to read to participant: "Think about how confident you are to engage in JOB-RELATED physical activity (paid work, volunteer work, coursework) at MODERATE intensity in the next week."
Scale: 0=No Confidence, 1=Low, 2=Moderate, 3=High, 4=Complete Confidence.
STOP RULE: If participant answers 0 on item 1, record 0 for all remaining items and skip to [INSTRUMENT_DATA].
Ask each item one at a time. After each question, append on the very last line: ${QUIZ_SCALE_TAG}
1. How confident are you in your ability to engage in job-related PA at moderate intensity for at least 10 minutes in the next week?
2. …for at least 30 minutes?
3. …for at least 60 minutes?
4. …for at least 90 minutes?
5. …for at least 120 minutes?
6. …for at least 150 minutes?
After all items: [INSTRUMENT_DATA: {"instrument":"J-R PASE","item1":0,"item2":0,"item3":0,"item4":0,"item5":0,"item6":0,"total_score":0}]`,

  inst3: `T-R PASE — Transportation-Related PA Self-Efficacy.
Context to read to participant: "Think about how confident you are to engage in TRANSPORTATION-RELATED physical activity (walking or cycling to get from place to place) at MODERATE intensity in the next week."
Scale: 0=No Confidence, 1=Low, 2=Moderate, 3=High, 4=Complete Confidence.
STOP RULE: If participant answers 0 on item 1, record 0 for all remaining items and skip to [INSTRUMENT_DATA].
Ask each item one at a time. After each question, append on the very last line: ${QUIZ_SCALE_TAG}
1. How confident are you in your ability to engage in transportation-related PA at moderate intensity for at least 10 minutes in the next week?
2. …for at least 30 minutes?
3. …for at least 60 minutes?
4. …for at least 90 minutes?
5. …for at least 120 minutes?
6. …for at least 150 minutes?
After all items: [INSTRUMENT_DATA: {"instrument":"T-R PASE","item1":0,"item2":0,"item3":0,"item4":0,"item5":0,"item6":0,"total_score":0}]`,

  inst4: `D-R PASE — Domestic-Related PA Self-Efficacy.
Context to read to participant: "Think about how confident you are to engage in DOMESTIC-RELATED physical activity (housework, gardening, yard work, caring for family) at MODERATE intensity in the next week."
Scale: 0=No Confidence, 1=Low, 2=Moderate, 3=High, 4=Complete Confidence.
STOP RULE: If participant answers 0 on item 1, record 0 for all remaining items and skip to [INSTRUMENT_DATA].
Ask each item one at a time. After each question, append on the very last line: ${QUIZ_SCALE_TAG}
1. How confident are you in your ability to engage in domestic-related PA at moderate intensity for at least 10 minutes in the next week?
2. …for at least 30 minutes?
3. …for at least 60 minutes?
4. …for at least 90 minutes?
5. …for at least 120 minutes?
6. …for at least 150 minutes?
After all items: [INSTRUMENT_DATA: {"instrument":"D-R PASE","item1":0,"item2":0,"item3":0,"item4":0,"item5":0,"item6":0,"total_score":0}]`,

  inst5: `L-R PASE — Leisure-Related PA Self-Efficacy.
Scale: 0=No Confidence, 1=Low, 2=Moderate, 3=High, 4=Complete Confidence.
STOP RULE: Stop the current part if participant answers 0 on any item; record 0 for remaining items in that part.
Ask each item one at a time. After each question, append on the very last line: ${QUIZ_SCALE_TAG}

PART A — MODERATE intensity:
Context: "Think about how confident you are to engage in LEISURE physical activity (recreation, sport, exercise) at MODERATE intensity in the next week."
A1. How confident are you to engage in leisure PA at MODERATE intensity for at least 10 minutes in the next week?
A2. …for at least 30 minutes?
A3. …for at least 60 minutes?
A4. …for at least 90 minutes?
A5. …for at least 120 minutes?
A6. …for at least 150 minutes?

PART B — VIGOROUS intensity:
Context: "Now think about VIGOROUS intensity leisure activities — those that make you breathe much harder than normal (e.g., running, intense cycling)."
B1. How confident are you to engage in leisure PA at VIGOROUS intensity for at least 10 minutes in the next week?
B2. …for at least 15 minutes?
B3. …for at least 30 minutes?
B4. …for at least 45 minutes?
B5. …for at least 60 minutes?
B6. …for at least 75 minutes?
After all items: [INSTRUMENT_DATA: {"instrument":"L-R PASE","moderate_items":{"item1":0,"item2":0,"item3":0,"item4":0,"item5":0,"item6":0},"vigorous_items":{"item1":0,"item2":0,"item3":0,"item4":0,"item5":0,"item6":0},"moderate_total":0,"vigorous_total":0}]`,

  inst6: `SERPA — Self-Efficacy to Regulate Physical Activity (13 items, no stop rule — ask all).
Context: "How confident are you that you can still meet the recommended weekly PA goal (150 min/wk moderate OR 75 min/wk vigorous) when the following happens?"
Scale: 0=No Confidence, 1=Low, 2=Moderate, 3=High, 4=Complete Confidence.
Ask each item one at a time. After each question, append on the very last line: ${QUIZ_SCALE_TAG}
1. The weather is very bad?
2. You are bored with the physical activities available?
3. You are on vacation?
4. You are uninterested in the physical activities available?
5. You feel physical discomfort while being physically active?
6. You have to be physically active by yourself?
7. You do not enjoy the physical activities available?
8. It is difficult to get to a location suitable for PA?
9. You do not like the physical activities available?
10. Your schedule conflicts with being physically active?
11. You feel self-conscious about your appearance while being physically active?
12. You do not receive any encouragement for being physically active?
13. You are under personal stress?
After all 13 items: [INSTRUMENT_DATA: {"instrument":"SERPA","item1":0,"item2":0,"item3":0,"item4":0,"item5":0,"item6":0,"item7":0,"item8":0,"item9":0,"item10":0,"item11":0,"item12":0,"item13":0,"total_score":0}]`,
};

const DEFAULT_PROFILES = [
  {
    id: "PT-0001",
    name: "Son Vu",
    trial: "Pilot Trial",
    drug: "Semaglutide 2.4mg",
    totalWeeks: 24,
    startProgramDay: 1,
    bmi: { current: 34.2, baseline: 38.1 },
    weight: { current: 198, baseline: 207, unit: "lbs", goal: 177 },
    adherence: 94,
    doctor: "Confident Moves Obesity Care Team",
    conditions: ["Type 2 diabetes", "Hypertension"],
    medications: ["Metformin 500mg", "Lisinopril 10mg"],
    pa: { weeklyGoalMins: 150, goalDays: 5, topBarrier: "time constraints", favoriteActivity: "walking" },
  },
  {
    id: "PT-0002",
    name: "Maria Reyes",
    trial: "Pilot Trial",
    drug: "Semaglutide 2.4mg",
    totalWeeks: 24,
    startProgramDay: 1,
    bmi: { current: 36.1, baseline: 37.8 },
    weight: { current: 212, baseline: 218, unit: "lbs", goal: 185 },
    adherence: 88,
    doctor: "Confident Moves Obesity Care Team",
    conditions: ["Hypertension", "Osteoarthritis"],
    medications: ["Lisinopril 10mg", "Acetaminophen PRN"],
    pa: { weeklyGoalMins: 150, goalDays: 5, topBarrier: "joint pain", favoriteActivity: "water aerobics" },
  },
  {
    id: "PT-0003",
    name: "David Kim",
    trial: "Pilot Trial",
    drug: "Semaglutide 2.4mg",
    totalWeeks: 24,
    startProgramDay: 1,
    bmi: { current: 31.5, baseline: 33.2 },
    weight: { current: 188, baseline: 195, unit: "lbs", goal: 170 },
    adherence: 91,
    doctor: "Confident Moves Obesity Care Team",
    conditions: ["Prediabetes"],
    medications: ["Metformin 500mg"],
    pa: { weeklyGoalMins: 150, goalDays: 5, topBarrier: "lack of social support", favoriteActivity: "cycling" },
  },
  {
    id: "PT-0004",
    name: "Angela Brooks",
    trial: "Pilot Trial",
    drug: "Semaglutide 2.4mg",
    totalWeeks: 24,
    startProgramDay: 1,
    bmi: { current: 29.8, baseline: 32.6 },
    weight: { current: 172, baseline: 188, unit: "lbs", goal: 155 },
    adherence: 97,
    doctor: "Confident Moves Obesity Care Team",
    conditions: ["Hyperlipidemia"],
    medications: ["Atorvastatin 20mg"],
    pa: { weeklyGoalMins: 150, goalDays: 5, topBarrier: "travel schedule", favoriteActivity: "running" },
  },
  {
    id: "PT-0005",
    name: "Robert Ellis",
    trial: "Pilot Trial",
    drug: "Semaglutide 2.4mg",
    totalWeeks: 24,
    startProgramDay: 1,
    bmi: { current: 41.3, baseline: 42.5 },
    weight: { current: 268, baseline: 276, unit: "lbs", goal: 230 },
    adherence: 79,
    doctor: "Confident Moves Obesity Care Team",
    conditions: ["Type 2 diabetes", "Sleep apnea", "Chronic low back pain"],
    medications: ["Metformin 1000mg", "CPAP therapy"],
    pa: { weeklyGoalMins: 150, goalDays: 5, topBarrier: "physical discomfort / pain", favoriteActivity: "walking the dog" },
  },
];

function cloneProfiles(profiles) {
  return JSON.parse(JSON.stringify(profiles));
}

// ─── Mock instrument batteries (5 datasets) ────────────────────────────────────
// Ready-made survey-battery responses per profile so RAG + structured injection
// have realistic self-efficacy data to work with during development/demo. These
// mirror the exact shape the assessment flow writes (PASE items 0–4 → /24; SERPA
// 13 items 0–4 → /52). Each profile has a distinct SE pattern for testing:
//   PT-0001 mixed (low job SE, time barrier)      PT-0002 pain-driven low domestic/leisure SE
//   PT-0003 low social-support barrier            PT-0004 uniformly HIGH SE (contrast arm)
//   PT-0005 uniformly LOW SE, multiple barriers
const MOCK_PROFILE_SCORES = {
  "PT-0001": {
    inst1: { gender: "Male", education: "Graduate degree", employment: "Employed full-time", marital_status: "Single" },
    inst2: { item1: 1, item2: 1, item3: 2, item4: 1, item5: 1, item6: 2, total_score: 8 },
    inst3: { item1: 2, item2: 3, item3: 2, item4: 3, item5: 2, item6: 2, total_score: 14 },
    inst4: { item1: 3, item2: 2, item3: 3, item4: 2, item5: 3, item6: 2, total_score: 15 },
    inst5: { moderate_total: 13, vigorous_total: 7 },
    inst6: { item1: 3, item2: 2, item3: 3, item4: 2, item5: 3, item6: 3, item7: 3, item8: 3, item9: 3, item10: 1, item11: 2, item12: 3, item13: 1, total_score: 32 },
  },
  "PT-0002": {
    inst1: { gender: "Female", education: "Some college", employment: "Employed part-time", marital_status: "Married" },
    inst2: { item1: 2, item2: 2, item3: 3, item4: 2, item5: 2, item6: 3, total_score: 14 },
    inst3: { item1: 1, item2: 2, item3: 1, item4: 2, item5: 1, item6: 2, total_score: 9 },
    inst4: { item1: 1, item2: 1, item3: 2, item4: 1, item5: 1, item6: 2, total_score: 8 },
    inst5: { moderate_total: 7, vigorous_total: 3 },
    inst6: { item1: 3, item2: 3, item3: 3, item4: 2, item5: 1, item6: 2, item7: 3, item8: 3, item9: 3, item10: 3, item11: 3, item12: 2, item13: 3, total_score: 34 },
  },
  "PT-0003": {
    inst1: { gender: "Male", education: "Bachelor's degree", employment: "Employed full-time", marital_status: "Divorced" },
    inst2: { item1: 3, item2: 3, item3: 3, item4: 2, item5: 3, item6: 3, total_score: 17 },
    inst3: { item1: 3, item2: 2, item3: 3, item4: 3, item5: 2, item6: 3, total_score: 16 },
    inst4: { item1: 2, item2: 3, item3: 2, item4: 3, item5: 2, item6: 3, total_score: 15 },
    inst5: { moderate_total: 15, vigorous_total: 12 },
    inst6: { item1: 3, item2: 3, item3: 3, item4: 3, item5: 3, item6: 1, item7: 3, item8: 3, item9: 3, item10: 3, item11: 3, item12: 1, item13: 3, total_score: 35 },
  },
  "PT-0004": {
    inst1: { gender: "Female", education: "Graduate degree", employment: "Employed full-time", marital_status: "Married" },
    inst2: { item1: 3, item2: 4, item3: 3, item4: 4, item5: 3, item6: 3, total_score: 20 },
    inst3: { item1: 4, item2: 3, item3: 4, item4: 3, item5: 4, item6: 3, total_score: 21 },
    inst4: { item1: 3, item2: 3, item3: 4, item4: 3, item5: 3, item6: 3, total_score: 19 },
    inst5: { moderate_total: 20, vigorous_total: 18 },
    inst6: { item1: 4, item2: 3, item3: 4, item4: 3, item5: 4, item6: 3, item7: 4, item8: 3, item9: 4, item10: 1, item11: 4, item12: 3, item13: 3, total_score: 43 },
  },
  "PT-0005": {
    inst1: { gender: "Male", education: "High school", employment: "Not employed (disability)", marital_status: "Widowed" },
    inst2: { item1: 1, item2: 0, item3: 1, item4: 1, item5: 0, item6: 1, total_score: 4 },
    inst3: { item1: 1, item2: 1, item3: 0, item4: 1, item5: 1, item6: 0, total_score: 4 },
    inst4: { item1: 1, item2: 1, item3: 1, item4: 0, item5: 1, item6: 1, total_score: 5 },
    inst5: { moderate_total: 5, vigorous_total: 2 },
    inst6: { item1: 1, item2: 1, item3: 1, item4: 1, item5: 0, item6: 1, item7: 1, item8: 1, item9: 1, item10: 1, item11: 1, item12: 1, item13: 1, total_score: 12 },
  },
};

/** Seed mock instrument batteries for any profile that has none yet.
 *  Non-destructive: skips a profile if scores already exist (e.g. from a real
 *  assessment session). Runs once at app init so RAG + injection have data. */
function seedMockScores() {
  for (const [profileId, battery] of Object.entries(MOCK_PROFILE_SCORES)) {
    const existing = readProfileScores(profileId);
    if (existing && Object.keys(existing).length > 0) continue;
    for (const [instrumentKey, responses] of Object.entries(battery)) {
      writeProfileScore(profileId, instrumentKey, responses);
    }
  }
}

function readProfiles() {
  try {
    const raw = localStorage.getItem(PROFILES_STORAGE_KEY);
    if (!raw) return cloneProfiles(DEFAULT_PROFILES);
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed) || parsed.length === 0) return cloneProfiles(DEFAULT_PROFILES);
    // Append any default profiles missing from an older cached list (e.g. new mock patients).
    const ids = new Set(parsed.map(p => p?.id));
    const merged = [...parsed];
    for (const def of DEFAULT_PROFILES) {
      if (!ids.has(def.id)) merged.push(JSON.parse(JSON.stringify(def)));
    }
    return merged;
  } catch {
    return cloneProfiles(DEFAULT_PROFILES);
  }
}

function writeProfiles(profiles) {
  try {
    localStorage.setItem(PROFILES_STORAGE_KEY, JSON.stringify(profiles));
  } catch (e) {
    console.warn("Profile save failed:", e);
  }
}

function getTotalProgramDays(patient) {
  return patient.totalWeeks * 7;
}

function programStateKey(profileId) {
  return `confidentMoves_program_${profileId}`;
}

function defaultProgramState(patient) {
  const day = Math.max(1, Number(patient.startProgramDay) || 1);
  return {
    programDay: day,
    currentWeight: patient.weight.current,
    weekPaMins: 0,
    activeDaysThisWeek: 0,
    lastPaLogProgramDay: null,
    lastActiveMarkProgramDay: null,
  };
}

function loadProgramState(profileId, patient) {
  try {
    const raw = localStorage.getItem(programStateKey(profileId));
    if (!raw) return null;
    const d = JSON.parse(raw);
    const base = defaultProgramState(patient);
    if (typeof d.programDay !== "number" || d.programDay < 1) return null;
    return { ...base, ...d, programDay: d.programDay };
  } catch {
    return null;
  }
}

function migrateLegacyProgramState(profileId) {
  try {
    const legacy = localStorage.getItem(LEGACY_PROGRAM_STATE_KEY);
    if (!legacy || localStorage.getItem(programStateKey(profileId))) return;
    localStorage.setItem(programStateKey(profileId), legacy);
  } catch {}
}

function estBmiFromWeight(weightLbs, patient) {
  const w = Number(weightLbs);
  if (!Number.isFinite(w) || w <= 0) return patient.bmi.baseline.toFixed(1);
  return ((patient.bmi.baseline * w) / patient.weight.baseline).toFixed(1);
}

// ─── Lightweight RAG knowledge base ───────────────────────────
// Each chunk is a self-contained clinical or theory fact the AI can
// cite in a response. At build time we keyword-match the patient's
// last message + their barrier/domain to select the top 2–3 chunks.
// This is "retrieval" not "training" — chunks are injected into the
// system prompt at inference time; the model weights never change.

const KNOWLEDGE_CHUNKS = [
  // ── PA Guidelines ──────────────────────────────────────────
  {
    id: "pa_guidelines_moderate",
    tags: ["pa", "guidelines", "moderate", "minutes", "150", "weekly"],
    text: "ACSM/AHA (2023): Adults need ≥150 min/week moderate-intensity PA OR ≥75 min/week vigorous PA. Bouts of ≥10 min count. For adults with obesity, start at 50–75% of target and progress gradually.",
  },
  {
    id: "pa_guidelines_obesity",
    tags: ["pa", "obesity", "weight", "bmi", "exercise"],
    text: "For adults with obesity, PA recommendations prioritize low-impact activity (walking, swimming, cycling) to reduce joint stress. Duration > intensity in early stages. 200–300 min/week needed for weight maintenance after loss.",
  },
  {
    id: "pa_guidelines_fragmented",
    tags: ["pa", "busy", "time", "short", "break", "10", "minutes", "schedule"],
    text: "Fragmented PA (3×10 min/day) produces comparable cardiovascular benefits to continuous bouts. Walking after meals is particularly effective for glycemic control in T2D and prediabetes.",
  },
  // ── Self-Efficacy Coaching Rules ───────────────────────────
  {
    id: "se_low_job",
    tags: ["job", "work", "volunteer", "course", "low", "confidence", "jr pase", "2a"],
    text: "Low J-R PASE (0–8): Do NOT suggest adding new work-based activity. Instead identify movement that already exists in the work context (walking between tasks, standing breaks). Reframe existing movement as PA achievement.",
  },
  {
    id: "se_low_transport",
    tags: ["transport", "walk", "cycle", "commute", "bus", "car", "low", "tr pase", "2b"],
    text: "Low T-R PASE (0–8): Start with one regular destination where a short walk is possible. Parking further away or getting off transit one stop early both count. Micro-goal: add 5–10 min to one existing trip.",
  },
  {
    id: "se_low_domestic",
    tags: ["home", "house", "housework", "garden", "domestic", "family", "low", "dr pase", "2c"],
    text: "Low D-R PASE (0–8): Validate that housework, gardening, and family care ARE physical activity. Help patient recognize movement they may not count as exercise. Reframe existing domestic activity as PA achievement.",
  },
  {
    id: "se_low_leisure",
    tags: ["leisure", "recreation", "sport", "gym", "exercise", "hobby", "low", "lr pase", "2d"],
    text: "Low L-R PASE moderate (0–8): Start with most accessible leisure activity — walking. 10-min neighborhood walks, mall walking, park visits. Do NOT suggest vigorous activity until moderate confidence improves.",
  },
  {
    id: "se_high_any",
    tags: ["high", "confidence", "strong", "doing well", "good", "motivated"],
    text: "High SE domain: Treat this as a strength to build from. Connect past success in this domain to other lower-SE domains. Extend goals gradually — if current plan is working, ask what would make it 10% bigger.",
  },
  // ── SERPA Barrier Coaching Rules ───────────────────────────
  {
    id: "serpa_schedule",
    tags: ["schedule", "time", "busy", "no time", "too tired", "work", "evening"],
    text: "Schedule barrier (SERPA item 10): Explore actual vs perceived time. Identify 2–3 schedule anchors (morning routine, lunch, post-dinner). Implementation intention: 'When I [anchor], I will do [10-min walk].' 10 minutes is enough — start there.",
  },
  {
    id: "serpa_self_conscious",
    tags: ["self-conscious", "embarrassed", "appearance", "gym", "people watching", "weight", "fat", "ashamed"],
    text: "Self-consciousness barrier (SERPA item 11): CRITICAL for obesity population. Validate without judgment — never reference weight or appearance. Suggest home-based or low-visibility options first: home walking, private outdoor time, online exercise videos.",
  },
  {
    id: "serpa_discomfort",
    tags: ["pain", "hurts", "discomfort", "joint", "knee", "back", "tired", "fatigue", "sore"],
    text: "Physical discomfort barrier (SERPA item 5): Validate discomfort first — never push through pain. Suggest chair-based exercise, water walking, or gentle stretching. Lower intensity AND shorter duration. Escalate if pain is new or severe.",
  },
  {
    id: "serpa_weather",
    tags: ["weather", "rain", "cold", "hot", "outside", "outdoor", "winter"],
    text: "Weather barrier (SERPA item 1): Remove outdoor dependency. Home-based options: walking in place, stairs, exercise videos, chair exercises. Indoor public spaces: mall walking, community centers. Reframe: weather-proof activity exists.",
  },
  {
    id: "serpa_alone",
    tags: ["alone", "lonely", "no one", "partner", "friend", "support", "social", "by myself"],
    text: "Social support barrier (SERPA item 6): Lack of support undermines SE. Explore social options — walking groups, online communities, phone PA with a friend. If none available, normalize solo activity as a personal strength.",
  },
  {
    id: "serpa_stress",
    tags: ["stress", "stressed", "overwhelmed", "anxious", "depressed", "mental", "rough", "hard week"],
    text: "Stress barrier (SERPA item 13): Validate stress fully before any PA suggestion. In high-stress periods, survival-mode goals only: smallest possible commitment (5 min, one walk). PA itself reduces stress — frame as relief, not obligation.",
  },
  {
    id: "serpa_no_encouragement",
    tags: ["no encouragement", "no one cares", "alone", "nobody", "support", "motivation"],
    text: "No encouragement barrier (SERPA item 12): The coaching AI's primary role here is to BE the social persuasion SE source. Provide specific, genuine affirmation every session. Reference patient's own stated wins, not generic praise.",
  },
  // ── GLP-1 + Exercise ───────────────────────────────────────
  {
    id: "glp1_nausea_exercise",
    tags: ["nausea", "sick", "side effect", "injection", "medication", "semaglutide", "glp1", "tired"],
    text: "GLP-1 receptor agonists (semaglutide, tirzepatide): Early weeks commonly cause nausea, fatigue, and reduced appetite. These may limit exercise tolerance. Reduce intensity and duration during first 4–8 weeks. Nausea improves with dose stabilization. Always escalate if severe.",
  },
  {
    id: "glp1_muscle",
    tags: ["muscle", "strength", "resistance", "lean", "mass", "glp1", "semaglutide"],
    text: "GLP-1 therapy causes weight loss that includes muscle mass loss. Resistance exercise (2×/week) is recommended alongside aerobic PA to preserve lean mass. Even light resistance training (bands, bodyweight) is beneficial.",
  },
  // ── BCT Decision Rules ─────────────────────────────────────
  {
    id: "bct_action_plan",
    tags: ["plan", "when", "where", "how", "goal", "specific", "schedule"],
    text: "Action planning (BCT): Effective PA plans specify WHEN (day + time), WHERE (location), HOW LONG. Implementation intention format: 'If [situation], then I will [PA behavior].' Patient must agree — never prescribe.",
  },
  {
    id: "bct_confidence_check",
    tags: ["confident", "confidence", "sure", "can do", "try", "maybe", "0 to 10"],
    text: "Confidence check (BCT): Ask 0–10 confidence scale for the specific plan. If <7, shrink the goal until confidence rises. The right goal is one the patient will actually do, not the optimal clinical goal.",
  },
  {
    id: "bct_problem_solving",
    tags: ["barrier", "problem", "obstacle", "hard", "difficult", "challenge", "stuck"],
    text: "Problem solving (BCT): Identify one barrier → brainstorm two options → patient picks. Avoid solving for them. Use 'What have you tried before?' to elicit their own solutions first.",
  },
];

/** Lightweight RAG retriever — keyword-based (no embeddings needed for feasibility).
 *  Takes the patient's last message + their known barriers/domains and returns
 *  the top matching chunks to inject into the system prompt.
 *  Phase 2: replace with semantic vector search (pgvector / Pinecone).
 *
 *  PRODUCTION Phase 5 — reference library via function calling:
 *    • Move KNOWLEDGE_CHUNKS + retrieveChunks to server; expose as tools search_references / get_reference_by_id.
 *    • Remove always-on buildRagBlock() injection in send() when tools are enabled — model pulls refs on demand.
 *    • Keep retrieveChunks for Condition B/C only, or disable for Condition A to preserve experimental arm.
 *    • Embeddings search can live in the same server module as /api/chat. */
function retrieveChunks(userMessage, patientBarrier = "", patientScores = {}, topN = 3) {
  const query = `${userMessage} ${patientBarrier}`.toLowerCase();

  // Also build domain signals from scores (trigger low-SE rules automatically)
  const domainSignals = [];
  if (patientScores.inst2?.total_score != null && patientScores.inst2.total_score <= 8) domainSignals.push("jr pase low job work");
  if (patientScores.inst3?.total_score != null && patientScores.inst3.total_score <= 8) domainSignals.push("tr pase low transport commute");
  if (patientScores.inst4?.total_score != null && patientScores.inst4.total_score <= 8) domainSignals.push("dr pase low home domestic");
  if (patientScores.inst5?.moderate_total != null && patientScores.inst5.moderate_total <= 8) domainSignals.push("lr pase low leisure exercise");
  if (patientScores.inst6) {
    const s = patientScores.inst6;
    if ((s.item10 ?? 4) <= 1) domainSignals.push("schedule time busy");
    if ((s.item11 ?? 4) <= 1) domainSignals.push("self-conscious appearance embarrassed");
    if ((s.item5 ?? 4) <= 1) domainSignals.push("pain discomfort joint");
    if ((s.item13 ?? 4) <= 1) domainSignals.push("stress stressed overwhelmed");
    if ((s.item6 ?? 4) <= 1) domainSignals.push("alone no support social");
    if ((s.item1 ?? 4) <= 1) domainSignals.push("weather outdoor rain cold");
  }

  const fullQuery = `${query} ${domainSignals.join(" ")}`;

  const scored = KNOWLEDGE_CHUNKS.map(chunk => {
    const hits = chunk.tags.filter(tag => fullQuery.includes(tag)).length;
    return { chunk, hits };
  }).filter(x => x.hits > 0)
    .sort((a, b) => b.hits - a.hits)
    .slice(0, topN);

  return scored.map(x => x.chunk);
}

/** Format retrieved chunks into a compact block for injection into the system prompt. */
function buildRagBlock(chunks) {
  if (!chunks.length) return "";
  return `\nRETRIEVED KNOWLEDGE (use these facts to ground your response — do not copy verbatim; synthesize naturally):\n${chunks.map((c, i) => `[${i + 1}] ${c.text}`).join("\n")}`;
}

function buildInstrumentInjectionBlock(conditionId) {
  const cond = CONDITIONS.find(c => c.id === conditionId);
  if (!cond?.instruments?.length) return "";
  const parts = cond.instruments.map(key => {
    const label = INSTRUMENT_LABELS[key] ?? key;
    // const text = INSTRUMENT_TEXTS[key] ?? "";
    const text = INSTRUMENT_PROMPT_SUMMARIES[key] ?? "";
    return `--- ${key.toUpperCase()}: ${label} ---\n${text}`;
  });
  return `\n\nRESEARCH ASSESSMENT INSTRUMENTS (compact coaching reference; all instruments included below. Use these to tailor support, but do not run full survey administration in this chat unless the participant requests it):\n\n${parts.join("\n\n")}`;
}

function applyConditionToSystems(systems, conditionId) {
  const block = buildInstrumentInjectionBlock(conditionId);
  if (!block) return systems;
  const out = { ...systems };
  for (const key of COACHING_SYSTEM_KEYS) {
    if (out[key]) out[key] = out[key] + block;
  }
  return out;
}

/** Days from real "today" until the next scheduled visit (sidebar + Progress). */
const VISIT_OFFSET_DAYS = 14;

function formatTodayLong(date = new Date()) {
  return date.toLocaleDateString(undefined, {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

/** Calendar-safe date add (local midnight baseline). */
function addDays(base, days) {
  const d = new Date(base.getFullYear(), base.getMonth(), base.getDate());
  d.setDate(d.getDate() + days);
  return d;
}

/** Short date for visits (e.g. May 22, 2026). */
function formatVisitShort(date) {
  return date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

function getNextVisitFromToday(from = new Date()) {
  return addDays(from, VISIT_OFFSET_DAYS);
}

function formatMonthDayShort(date) {
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

/** Build prompts that match the current program day and self-reported metrics. 
 * Dr. Lee and Joon can add additional prompts to the system messages if needed.
 * Son and Dr. Lee can add additional prompts to the system messages if needed.
 * We will design prompts for the other instruments later.
*/
function buildSystems(rt, patient, conditionId = "A") {
  const {
    programDay,
    programWeek,
    currentWeight,
    estBmi,
    weeklyPaMins,
    weeklyPaGoal,
    activeDaysThisWeek,
    activeDaysGoal,
  } = rt;
  const totalProgramDays = getTotalProgramDays(patient);
  const withTheory = CONDITIONS.find(c => c.id === conditionId)?.includeTheory ?? conditionId !== "A";

  // Read any instrument scores collected for this profile (from assessment sessions).
  // For Conditions B/C, inject these into coaching prompts so the AI can personalize.
  const profileScores = withTheory ? readProfileScores(patient.id) : {};
  const scoreContextBlock = buildScoreContextBlock(profileScores);

  const patientContext = `Patient: ${patient.name}, Trial: ${patient.trial}, Drug: ${patient.drug}.
Program timeline: Day ${programDay} of ${totalProgramDays} (week ${programWeek} of ${patient.totalWeeks}).
Self-report: weight ${currentWeight} lbs; estimated BMI ~${estBmi} (baseline BMI ${patient.bmi.baseline}); this week's PA total ${weeklyPaMins} / ${weeklyPaGoal} min; active days with movement this week: ${activeDaysThisWeek} / ${activeDaysGoal}.
Comorbidities: ${patient.conditions.join(", ")}.
Top PA barrier: ${patient.pa.topBarrier}. Favorite activity: ${patient.pa.favoriteActivity}.`;

  const operatingRules = `OPERATING RULES
- Never diagnose, prescribe, or change medication instructions. Defer medical decisions to the care team.
- Keep replies warm and focused (about 3–6 sentences) unless the user asks for detail. Include one reflective or open question when it fits.
- Flag severe side effects, distress, self-harm, or safety concerns with [ESCALATE].
- You may discuss GLP-1 therapy, obesity, nutrition, PA, and the trial in general educational terms.`;

  const paceMiBlock = `Weave in naturally; do not label "PACE" to the patient.
1) Partnership: Work with them as an equal collaborator. Reflect their words. Ask permission before advice ("Would it be OK if we talked about…?"). Explore goals they care about.
2) Acceptance: Validate their experience without judgment. Accept ambivalence — do not argue or lecture. Treat setbacks as information, not failure.
3) Compassion: Respond warmly to struggle, shame, or stigma. Never shame weight, appearance, or willpower. Acknowledge how hard change can be in their context.
4) Empowerment: Elicit their ideas and choices. They decide the next step. Build on their strengths and past wins. Use open questions, affirm effort, reflect, and summarize.`;

  const miExamplesBlock = `MI-CONSISTENT VS MI-CONTRAINDICATED RESPONSES (calibration — never copy verbatim; match the spirit):
These illustrate MI-consistent (reflect, validate, open question, patient choice) versus MI-contraindicated (lecture, pressure, shame, prescribe, argue). Always respond MI-consistently.

EXAMPLE 1 — Fatigue / low energy (affective SE source; Acceptance + Compassion)
Patient: "I want to try but I'm just so tired all the time. By the end of the day I have nothing left."
MI-CONTRAINDICATED (avoid): "You need to push through the fatigue — 30 minutes of exercise daily is the goal and you'll feel better once you start."
Why contraindicated: Directs without permission; dismisses experience; prescribes a plan; empty reassurance.
MI-CONSISTENT: "Finishing the day with nothing left sounds really draining. What has helped on days when you had even a little energy — even five minutes?"
Why consistent: Reflects fatigue; validates without arguing; open question; elicits their experience (Empowerment).

EXAMPLE 2 — Self-consciousness about appearance (SERPA item 11; Compassion + Partnership)
Patient: "I feel embarrassed exercising in public because of my size. I don't want people looking at me."
MI-CONTRAINDICATED (avoid): "Once you lose more weight you'll feel more comfortable at the gym — you should start going anyway."
Why contraindicated: Comments on weight/appearance; argues against their concern; prescribes without collaboration.
MI-CONSISTENT: "Feeling watched can make anywhere feel uncomfortable. What kinds of movement have felt okay when you're on your own?"
Why consistent: Validates shame without judgment; no appearance talk; explores their ideas; patient-led next step.`;

  const chatTrained = `You are ObesityCare AI, a clinical support assistant embedded in an obesity management trial platform. Your coaching is grounded in two theory-aligned pillars: (1) social cognitive theory — specifically Bandura's self-efficacy — and (2) motivational interviewing, expressed through the PACE relational components (Partnership, Acceptance, Compassion, Empowerment). You also draw on common elements of evidence-based behavior change techniques (BCTs) used in lifestyle trials (e.g., goal setting, action planning, problem solving, self-monitoring of behavior).

${patientContext}${scoreContextBlock}

THEORY 1 — SELF-EFFICACY (Bandura)
Self-efficacy is the person's confidence that they can perform a behavior in a given context. In practice, support mastery, credible encouragement, context, and emotional safety — not generic praise.
1) Mastery experiences: Elicit past successes — even small (e.g., "What helped the last time you fit in movement?"). Tie next steps to those successes.
2) Vicarious experience: When fitting, normalize with cohort-appropriate language (trials like this; many people with busy schedules) — never compare one patient invidiously to another.
3) Verbal/social persuasion: Use authentic, specific encouragement tied to their own data or stated intent — avoid empty reassurance or pressure.
4) Physiological/affective states: Acknowledge fatigue, stress, side effects, mood. Reframe discomfort as information for a smaller or adjusted plan, not as failure. Escalate severe symptoms.

THEORY 2 — MOTIVATIONAL INTERVIEWING (PACE)
${paceMiBlock}

${miExamplesBlock}

EVIDENCE-ALIGNED STRATEGIES (use when relevant; do not stack all in one reply)
- Collaborative goal setting: Patient-chosen priority; ask what feels "doable this week" before suggesting specifics.
- Action planning: When/where/how long; break into steps the patient agrees to (implementation intentions: "If [situation], then I will [micro-action]").
- Confidence / importance: Brief 0–10 check ("How confident are you that you can do that plan?"); if confidence is low, shrink the step until confidence rises.
- Problem solving: Identify barrier → brainstorm one or two options → patient picks; avoid solving for them.

INTERNAL REASONING (chain-of-thought — NEVER print this section; it is your private checklist):
Before writing every coaching reply, silently answer these four questions, then write only the final response:
  Q1. SE source: Which of the four Bandura sources (mastery / vicarious / verbal-social persuasion / physiological-affective) is most relevant to what the patient just said? If none clearly applies, note that.
  Q2. PACE element: Which of the four MI components (Partnership / Acceptance / Compassion / Empowerment) should most shape the tone of this reply?
  Q3. Score signal: Is there collected assessment data (PASE domain or SERPA barrier) that should change what I say or suggest? If yes, which domain and how?
  Q4. One thing: What is the single most useful thing to say or ask right now? (Only one — do not stack multiple BCTs or ask multiple questions.)
Write only the final coaching response — never show Q1–Q4 to the patient.

PROACTIVE ASSESSMENT SUGGESTION: If the patient mentions a domain of uncertainty or a barrier that maps to an assessment instrument, you may — once per session, naturally — invite them to complete the relevant tool (e.g., "You mentioned worrying about being active alone — would you like to take a quick 2-minute survey so I can understand your confidence better? You can find it in the Assessments bar at the top"). Keep it optional, brief, and framed as useful to them.

QUIZ TAGS (assessment administration only): When directly asking a research instrument question that has fixed answer options (PASE confidence scale OR demographic categories), append the appropriate [QUIZ: {...}] tag as the very last line of your response with no text after it. Use [QUIZ: {"type":"scale","min":0,"max":4,"labels":["No Confidence","Low","Moderate","High","Complete Confidence"]}] for any 0–4 confidence item; use [QUIZ: {"type":"choice","options":["A","B","..."]}] for categorical choices. NEVER include [QUIZ:] in regular coaching or conversational replies — only when administering a specific scored item.

${operatingRules}`;

  const chatBaseline = `You are ObesityCare AI, a supportive assistant for participants in an obesity management trial. Answer questions helpfully about physical activity, lifestyle, and the program. Be warm and practical. Do not use formal behavior-change theory frameworks, named psychological models (e.g., self-efficacy, motivational interviewing, PACE), or structured coaching protocols unless the participant explicitly asks for them.

${patientContext}

${operatingRules}`;

  const checkinTrained = `You are ObesityCare Confident Moves AI conducting a structured daily check-in for a clinical trial participant. Use brief, collaborative language grounded in self-efficacy support and motivational interviewing (PACE: Partnership, Acceptance, Compassion, Empowerment). Acknowledge effort; ask one thing at a time; no judgment; reflect before the next question. QUIZ TAGS: If administering a research instrument question with fixed options, append [QUIZ: {...}] as the very last line (see PA Coach format instructions).
Patient: ${patient.name}, program day ${programDay} (week ${programWeek}), Drug: ${patient.drug}.
Self-reported weight ${currentWeight} lbs; weekly PA minutes so far ${weeklyPaMins} / ${weeklyPaGoal}.
Conduct a brief, empathetic check-in. Ask ONE question at a time about:
1. Hunger/appetite (1-10 scale)
2. Side effects (nausea, fatigue, injection site reactions)
3. Mood and energy
4. Medication adherence
5. Any concerns
Keep each question short. After 5 exchanges, summarize the check-in data in a JSON block like: [CHECKIN_DATA: {...}]`;

  const checkinBaseline = `You are ObesityCare AI conducting a structured daily check-in for a clinical trial participant. Be polite and efficient. Do not use motivational interviewing, PACE, self-efficacy, or other named behavior-change approaches.
Patient: ${patient.name}, program day ${programDay} (week ${programWeek}), Drug: ${patient.drug}.
Self-reported weight ${currentWeight} lbs; weekly PA minutes so far ${weeklyPaMins} / ${weeklyPaGoal}.
Ask ONE question at a time about:
1. Hunger/appetite (1-10 scale)
2. Side effects (nausea, fatigue, injection site reactions)
3. Mood and energy
4. Medication adherence
5. Any concerns
Keep each question short. After 5 exchanges, summarize the check-in data in a JSON block like: [CHECKIN_DATA: {...}]`;

  const educationTrained = `You are ObesityCare AI, an educational assistant specializing in obesity medicine, GLP-1 therapy, nutrition, and lifestyle modification. Prefer clear, evidence-based statements; when citing mechanisms or guidelines, speak at a population level and avoid overstating certainty. When discussing behavior change, you may briefly reference well-supported ideas (e.g., realistic action planning, building self-efficacy through small successes, and person-centered support via partnership, acceptance, compassion, and empowerment) without claiming individualized treatment.
The participant is on program day ${programDay} of ${totalProgramDays}. Keep responses to 3-5 sentences unless the user asks for more detail.
Always end with an invitation to ask a follow-up question.`;

  const educationBaseline = `You are ObesityCare AI, an educational assistant for obesity medicine, GLP-1 therapy, nutrition, and lifestyle topics. Give clear, factual answers at a general population level. Do not frame answers using behavior-change theory, self-efficacy, motivational interviewing, PACE, or named coaching models unless the participant asks.
The participant is on program day ${programDay} of ${totalProgramDays}. Keep responses to 3-5 sentences unless the user asks for more detail.
Always end with an invitation to ask a follow-up question.`;

  return {
  chat: withTheory ? chatTrained : chatBaseline,

  checkin: withTheory ? checkinTrained : checkinBaseline,

  eligibility: `You are ObesityCare AI screening a patient for trial eligibility.
Current trial: STEP-OB-24 (Semaglutide extended therapy)
Inclusion criteria: BMI ≥ 30 (or ≥ 27 with comorbidity), age 18-70, no prior GLP-1 therapy, willing to modify lifestyle.
Exclusion criteria: pregnancy, severe renal impairment, personal/family history of MTC, pancreatitis history, active eating disorder.
Ask ONE screening question at a time. Be clinical but friendly. After collecting enough info, give a clear ELIGIBLE / POTENTIALLY ELIGIBLE / NOT ELIGIBLE verdict with reasoning. Never give a definitive medical clearance — always say the care team will review.`,

  education: withTheory ? educationTrained : educationBaseline,
  };
}

function patientInitials(name) {
  const p = String(name).trim().split(/\s+/).filter(Boolean);
  if (p.length >= 2) return (p[0][0] + p[1][0]).toUpperCase();
  return (p[0] || "U").slice(0, 2).toUpperCase();
}

// ─── Micro-components ─────────────────────────────────────────
function Avatar({ initials, color = T.teal, bg = T.tealLight, size = 32 }) {
  return (
    <div style={{
      width: size, height: size, borderRadius: "50%",
      background: bg, color, display: "flex", alignItems: "center",
      justifyContent: "center", fontWeight: 600, fontSize: size * 0.35,
      flexShrink: 0, fontFamily: "'DM Sans', sans-serif"
    }}>{initials}</div>
  );
}

function Badge({ children, color = T.teal, bg = T.tealLight }) {
  return (
    <span style={{
      background: bg, color, fontSize: 11, fontWeight: 600,
      padding: "3px 9px", borderRadius: 20, letterSpacing: ".02em"
    }}>{children}</span>
  );
}

function MetricCard({ label, value, sub, progress, color = T.teal, compact = false, sidebar = false }) {
  const bigger = compact && sidebar;
  const pad = compact ? (bigger ? "9px 11px" : "8px 10px") : "12px 14px";
  const valSize = bigger ? 17 : compact ? 15 : 20;
  const barH = compact ? (bigger ? 5 : 4) : 5;
  const labelSize = bigger ? 11 : compact ? 10 : 11;
  const subSize = bigger ? 10.5 : compact ? 9.5 : 11;
  return (
    <div style={{
      background: "#fff", border: `1px solid ${T.gray200}`,
      borderRadius: 10, padding: pad, marginBottom: compact ? 0 : 8,
    }}>
      <div style={{ fontSize: labelSize, color: T.gray500, marginBottom: compact ? 2 : 4, fontWeight: 500 }}>{label}</div>
      <div style={{ fontSize: valSize, fontWeight: 700, color: T.gray800, lineHeight: 1.2 }}>{value}</div>
      {sub && <div style={{ fontSize: subSize, color, marginTop: compact ? 2 : 3, fontWeight: 500, lineHeight: 1.35 }}>{sub}</div>}
      {progress != null && (
        <div style={{ height: barH, background: T.gray200, borderRadius: 3, marginTop: compact ? 6 : 8, overflow: "hidden" }}>
          <div style={{ width: `${progress}%`, height: "100%", background: color, borderRadius: 3, transition: "width .6s ease" }} />
        </div>
      )}
    </div>
  );
}

function TypingDots() {
  return (
    <div style={{ display: "flex", gap: 4, padding: "12px 16px", alignItems: "center" }}>
      {[0, 1, 2].map(i => (
        <div key={i} style={{
          width: 7, height: 7, borderRadius: "50%", background: T.gray400,
          animation: "bounce .9s infinite", animationDelay: `${i * 0.2}s`
        }} />
      ))}
    </div>
  );
}

/** Renders inline clickable answer buttons for research instrument questions.
 *  quiz.type === "scale"  → numbered confidence buttons (0–4)
 *  quiz.type === "choice" → text option buttons */
function QuizWidget({ quiz, onSelect }) {
  const [selected, setSelected] = useState(null);

  const pick = (displayValue) => {
    if (selected !== null) return;
    setSelected(displayValue);
    onSelect(displayValue);
  };

  if (quiz.type === "scale") {
    const min = quiz.min ?? 0;
    const max = quiz.max ?? 4;
    const labels = quiz.labels ?? [];
    return (
      <div style={{ marginTop: 12, display: "flex", gap: 7, flexWrap: "wrap" }}>
        {Array.from({ length: max - min + 1 }, (_, i) => i + min).map(val => {
          const label = labels[val - min] ?? String(val);
          const isSel = selected === `${val} — ${label}`;
          const isDimmed = selected !== null && !isSel;
          return (
            <button
              key={val}
              onClick={() => pick(`${val} — ${label}`)}
              disabled={selected !== null}
              style={{
                display: "flex", flexDirection: "column", alignItems: "center",
                padding: "8px 11px", borderRadius: 10,
                border: `2px solid ${isSel ? T.teal : T.gray300}`,
                background: isSel ? T.tealLight : "#fff",
                color: isSel ? T.teal : T.gray700,
                cursor: selected !== null ? "default" : "pointer",
                fontFamily: "'DM Sans', sans-serif",
                transition: "all .15s",
                minWidth: 54,
                opacity: isDimmed ? 0.35 : 1,
                boxShadow: isSel ? `0 0 0 1px ${T.teal}` : "none",
              }}
            >
              <span style={{ fontSize: 19, fontWeight: 700, lineHeight: 1 }}>{val}</span>
              <span style={{ fontSize: 9, marginTop: 4, textAlign: "center", lineHeight: 1.25, maxWidth: 58 }}>{label}</span>
            </button>
          );
        })}
      </div>
    );
  }

  if (quiz.type === "choice") {
    return (
      <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 5 }}>
        {(quiz.options ?? []).map(opt => {
          const isSel = selected === opt;
          const isDimmed = selected !== null && !isSel;
          return (
            <button
              key={opt}
              onClick={() => pick(opt)}
              disabled={selected !== null}
              style={{
                textAlign: "left", padding: "8px 14px", borderRadius: 8,
                border: `1.5px solid ${isSel ? T.teal : T.gray300}`,
                background: isSel ? T.tealLight : "#fff",
                color: isSel ? T.teal : T.gray700,
                cursor: selected !== null ? "default" : "pointer",
                fontFamily: "'DM Sans', sans-serif",
                fontSize: 13, fontWeight: isSel ? 600 : 400,
                transition: "all .15s",
                opacity: isDimmed ? 0.35 : 1,
              }}
            >
              {isSel ? "✓ " : ""}{opt}
            </button>
          );
        })}
      </div>
    );
  }

  return null;
}

// ─── Chat Engine ──────────────────────────────────────────────
function ChatEngine({ systemKey, systems, placeholder, quickReplies = [], intro, persistInstrument, conversationKey, conversationLabel, conversationMeta, patient, assistantInitials = "AI", apiMinIntervalMs, ragEnabled = false, ragScores = null }) {
  const initialMessages = useMemo(() => {
    if (conversationKey) {
      const saved = readConvStore()[conversationKey]?.messages;
      if (saved && saved.length > 0) return saved;
    }
    return intro ? [{ role: "assistant", content: intro, ts: new Date().toISOString() }] : [];
  }, [conversationKey]); // remount when profile/condition key changes

  const [messages, setMessages] = useState(initialMessages);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    setMessages(initialMessages);
  }, [initialMessages]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const persistConv = useCallback((msgs) => {
    if (!conversationKey || !conversationMeta) return;
    saveConversation(
      conversationKey,
      conversationMeta.moduleId,
      conversationLabel ?? conversationMeta.moduleId,
      msgs,
      conversationMeta,
    );
  }, [conversationKey, conversationLabel, conversationMeta]);

  const newSession = useCallback(() => {
    if (loading) return;
    const freshIntro = intro ? [{ role: "assistant", content: intro, ts: new Date().toISOString() }] : [];
    if (conversationKey) clearConversation(conversationKey);
    setMessages(freshIntro);
    setInput("");
  }, [loading, intro, conversationKey]);

  const send = useCallback(async (text) => {
    const userMsg = text.trim();
    if (!userMsg || loading) return;
    setInput("");
    const userEntry = { role: "user", content: userMsg, ts: new Date().toISOString() };
    const newMessages = [...messages, userEntry];
    const withPlaceholder = [...newMessages, { role: "assistant", content: "", ts: new Date().toISOString() }];
    setMessages(withPlaceholder);
    setLoading(true);

    const apiMessages = newMessages.slice(-8).map(m => ({ role: m.role, content: m.content }));

    let activeSystemPrompt = systems[systemKey];

    // ─── FUNCTION CALLING — model pulls references on demand (OFF by default) ──────
    // Active for PA Coach chat in Conditions B/C (ragEnabled) only when
    // FUNCTION_CALLING_ENABLED. When off, we use always-on RAG injection below.
    const useTools = FUNCTION_CALLING_ENABLED && ragEnabled && systemKey === "chat" && !persistInstrument
      && messageLikelyNeedsReferences(userMsg);

    // ─── RAG (keyword injection) — ACTIVE when function calling is off ─────────────
    // Retrieves the top matching knowledge chunks for the patient's message +
    // their known barriers/low-SE domains (from injected scores) and grounds the
    // reply. This is the current retrieval strategy for Conditions B/C.
    if (!useTools && ragEnabled && systemKey === "chat" && !persistInstrument) {
      const chunks = retrieveChunks(
        userMsg,
        patient?.pa?.topBarrier ?? "",
        ragScores ?? {},
        3,
      );
      const ragBlock = buildRagBlock(chunks);
      if (ragBlock) activeSystemPrompt = activeSystemPrompt + ragBlock;
    }

    if (useTools) activeSystemPrompt = activeSystemPrompt + buildToolInstructionBlock();

    let finalText = "";
    let finalMsgs = withPlaceholder;
    const toolsUsed = [];
    try {
      const streamHandler = (chunk) => {
        finalText = chunk;
        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = { ...updated[updated.length - 1], content: chunk };
          finalMsgs = updated;
          return updated;
        });
      };

      if (useTools) {
        // Function-calling path (non-streaming tool loop; llama-3.3-70b-versatile).
        await callGroqWithTools(apiMessages, activeSystemPrompt, streamHandler, {
          minIntervalMs: apiMinIntervalMs ?? MIN_GROQ_INTERVAL_MS,
          maxTokens: CHAT_MAX_TOKENS,
          onToolEvent: (ev) => {
            if (ev?.name !== "search_references") return;
            let ids = [];
            try { ids = (JSON.parse(ev.result)?.results ?? []).map(r => r.id); } catch {}
            toolsUsed.push({ query: ev.args?.query ?? "", ids, at: new Date().toISOString() });
          },
        });
      } else {
        // PRODUCTION: await callLLM(...) — same args; implementation delegates to /api/chat or Groq dev fallback.
        await callGroq(apiMessages, activeSystemPrompt, streamHandler, {
          minIntervalMs: apiMinIntervalMs ?? (persistInstrument ? ASSESSMENT_GROQ_INTERVAL_MS : MIN_GROQ_INTERVAL_MS),
          maxTokens: persistInstrument ? ASSESSMENT_MAX_TOKENS : CHAT_MAX_TOKENS,
        });
      }

      // After streaming completes, parse [QUIZ:] from the final response (or infer in assessment mode).
      const { quiz: parsedQuiz, cleanText: cleanedText } = parseQuizFromMessage(finalText);
      let quiz = parsedQuiz;
      let displayText = parsedQuiz ? cleanedText : finalText;
      if (!quiz && persistInstrument?.key) {
        quiz = inferAssessmentQuiz(finalText, persistInstrument.key);
      }
      if (quiz) {
        finalText = displayText;
        setMessages(prev => {
          const updated = [...prev];
          const lastIdx = updated.length - 1;
          if (updated[lastIdx]?.role === "assistant") {
            updated[lastIdx] = { ...updated[lastIdx], content: displayText, quiz };
          }
          finalMsgs = updated;
          return updated;
        });
      }

      if (!finalText?.trim() && !finalText?.startsWith("⚠️")) {
        finalText = "I'm here with you — could you say a bit more about what you'd like to focus on?";
        streamHandler(finalText);
      }

      if (useTools && toolsUsed.length && finalText && !finalText.startsWith("⚠️")) {
        recordReferenceLookups({
          profileId: conversationMeta?.profileId ?? "",
          condition: conversationMeta?.condition ?? "",
          moduleId: conversationMeta?.moduleId ?? systemKey,
          query: toolsUsed.map(t => t.query).join(" | "),
          referenceIds: [...new Set(toolsUsed.flatMap(t => t.ids ?? []))],
          toolsUsed,
        });
        setMessages(prev => {
          const updated = [...prev];
          const lastIdx = updated.length - 1;
          if (updated[lastIdx]?.role === "assistant") {
            let msg = { ...updated[lastIdx], referenceLookups: toolsUsed };
            if (SHOW_TOOL_DEBUG) {
              const footer = "\n\n———\n🔧 Function calling: " + formatReferenceLookups(toolsUsed);
              msg = { ...msg, content: (msg.content ?? "") + footer };
              finalText = finalText + footer;
            }
            updated[lastIdx] = msg;
          }
          finalMsgs = updated;
          return updated;
        });
      } else if (useTools && toolsUsed.length === 0 && finalText && !finalText.startsWith("⚠️")) {
        // Tools enabled but model skipped lookup — still note for research logs
        setMessages(prev => {
          const updated = [...prev];
          const lastIdx = updated.length - 1;
          if (updated[lastIdx]?.role === "assistant") {
            updated[lastIdx] = { ...updated[lastIdx], referenceLookups: [] };
          }
          finalMsgs = updated;
          return updated;
        });
      }

      if (persistInstrument && finalText && !finalText.startsWith("⚠️")) {
        const parsed = extractInstrumentJson(finalText);
        if (parsed) {
          await persistInstrumentSubmission({
            instrumentKey: persistInstrument.key,
            instrumentLabel: persistInstrument.label ?? persistInstrument.key,
            responses: parsed,
            patient: patient ?? { id: conversationMeta?.profileId, name: conversationMeta?.profileName },
            condition: conversationMeta?.condition,
            conditionLabel: conversationMeta?.conditionLabel,
          });
          const sheetNote = GOOGLE_SHEETS_WEBAPP_URL
            ? "\n\n✓ Responses saved in this browser and sent to the research Google Sheet (if the web app is deployed)."
            : "\n\n✓ Responses saved in this browser. Download JSON/CSV from My Progress, or add VITE_GOOGLE_SHEETS_WEBAPP_URL for sheet sync.";
          setMessages(prev => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last?.role === "assistant") {
              updated[updated.length - 1] = { ...last, content: last.content + sheetNote };
            }
            finalMsgs = updated;
            return updated;
          });
        }
      }
    } catch (e) {
      // PRODUCTION: generic "LLM API error" messaging; 401/403 → proxy/auth; 529 → Anthropic overloaded.
      console.error("❌ Groq API error:", e);
      const errMsg = e?.message || String(e);
      let friendlyMsg = `⚠️ Error: ${errMsg}\n\nCheck the browser console (F12) for details.`;
      if (errMsg.includes("400")) {
        const groqDetail = errMsg.includes("{") ? errMsg.slice(errMsg.indexOf("{")) : errMsg;
        friendlyMsg = `⚠️ API Error 400: Bad request.\n\n${groqDetail.slice(0, 400)}`;
        if (errMsg.includes("tool_use_failed")) {
          friendlyMsg += "\n\n(Tool calling failed — the app should auto-fallback on retry. If this persists, set VITE_FUNCTION_CALLING=false or add VITE_GROQ_MODEL=llama-3.3-70b-versatile to .env and restart.)";
        }
      }
      if (errMsg.includes("403")) friendlyMsg = "⚠️ API Error 403: Permission denied — your Groq key may be invalid or expired.";
      if (errMsg.includes("429")) friendlyMsg = "⚠️ Groq rate limit (429). The app auto-retried several times — wait 30–60 seconds, then send your message again. Tip: pause ~4 seconds between quiz answers during assessments. Free-tier limits are tight; using llama-3.1-8b-instant helps.";
      if (errMsg.includes("Failed to fetch") || errMsg.includes("NetworkError")) friendlyMsg = "⚠️ Network error: Could not reach Groq API. Check your internet connection.";
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = { ...updated[updated.length - 1], content: friendlyMsg };
        finalMsgs = updated;
        return updated;
      });
    }
    setLoading(false);
    persistConv(finalMsgs);
    // Push de-identified conversation to Google Sheets on every saved turn (fire-and-forget).
    if (conversationKey && conversationMeta) {
      pushConversationToSheet({
        profileId: conversationMeta.profileId ?? "",
        condition: conversationMeta.condition ?? "",
        moduleId: conversationMeta.moduleId ?? "",
        messages: finalMsgs,
      });
    }
  }, [messages, loading, systemKey, systems, persistInstrument, persistConv, patient, conversationMeta, conversationKey]);

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input); }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      <div style={{ flex: 1, minHeight: 0, overflowY: "auto", overscrollBehavior: "contain", padding: "16px 20px 20px", display: "flex", flexDirection: "column", gap: 12 }}>
        {messages.map((m, i) => {
          const isLast = i === messages.length - 1;
          const effectiveQuiz = m.quiz ?? (
            persistInstrument?.key && m.role === "assistant" && isLast && !loading && !m.quizAnswered
              ? inferAssessmentQuiz(m.content, persistInstrument.key)
              : null
          );
          return (
          <div key={i} style={{ display: "flex", gap: 10, alignItems: "flex-start", flexDirection: m.role === "user" ? "row-reverse" : "row", maxWidth: "90%", alignSelf: m.role === "user" ? "flex-end" : "flex-start" }}>
            <Avatar
              initials={m.role === "user" ? patientInitials(patient?.name ?? "User") : assistantInitials}
              color={m.role === "user" ? T.chatUserAvatarFg : T.chatAiAvatarFg}
              bg={m.role === "user" ? T.chatUserAvatarBg : T.chatAiAvatarBg}
              size={30}
            />
            <div style={{
              padding: "11px 15px", borderRadius: m.role === "user" ? "16px 4px 16px 16px" : "4px 16px 16px 16px",
              background: m.role === "user" ? T.chatUserBubble : T.chatAiBubble,
              border: m.role === "user" ? "1px solid transparent" : `1px solid ${T.chatAiBubbleBorder}`,
              color: m.role === "user" ? T.chatUserText : T.chatAiText,
              fontSize: 13.5, lineHeight: 1.6, whiteSpace: "pre-wrap",
              minHeight: m.content === "" ? 40 : "auto",
              boxShadow: m.role === "user" ? "0 1px 2px rgba(15, 23, 42, 0.06)" : "0 1px 2px rgba(15, 23, 42, 0.04)",
            }}>
              {m.content === "" ? <TypingDots /> : m.content}
              {effectiveQuiz && !m.quizAnswered && isLast && !loading && (
                <QuizWidget
                  key={`quiz-${i}`}
                  quiz={effectiveQuiz}
                  onSelect={(answer) => {
                    setMessages(prev => {
                      const updated = [...prev];
                      updated[i] = { ...updated[i], quizAnswered: true, quiz: effectiveQuiz };
                      return updated;
                    });
                    send(answer);
                  }}
                />
              )}
            </div>
          </div>
          );
        })}
        {loading && messages[messages.length - 1]?.role !== "assistant" && (
          <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
            <Avatar initials={assistantInitials} color={T.chatAiAvatarFg} bg={T.chatAiAvatarBg} size={30} />
            <div style={{
              padding: "10px 14px",
              background: T.chatAiBubble,
              border: `1px solid ${T.chatAiBubbleBorder}`,
              borderRadius: "4px 16px 16px 16px",
              boxShadow: "0 1px 2px rgba(15, 23, 42, 0.04)",
            }}>
              <TypingDots />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {quickReplies.length > 0 && messages.length <= 2 && (
        <div style={{ padding: "0 20px 10px", display: "flex", flexWrap: "wrap", gap: 6 }}>
          {quickReplies.map(r => (
            <button key={r} onClick={() => send(r)} style={{
              fontSize: 12, padding: "5px 12px", borderRadius: 20,
              border: `1px solid ${T.gray300}`, background: "#fff",
              color: T.gray600, cursor: "pointer", fontFamily: "'DM Sans', sans-serif",
              transition: "all .15s"
            }}
              onMouseOver={e => { e.target.style.borderColor = T.teal; e.target.style.color = T.teal; e.target.style.background = T.tealLight; }}
              onMouseOut={e => { e.target.style.borderColor = T.gray300; e.target.style.color = T.gray600; e.target.style.background = "#fff"; }}
            >{r}</button>
          ))}
        </div>
      )}

      <div style={{ padding: "10px 20px 16px", borderTop: `1px solid ${T.gray200}`, display: "flex", flexDirection: "column", gap: 8 }}>
        <div style={{ display: "flex", gap: 8, alignItems: "flex-end" }}>
          <textarea
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder={placeholder}
            rows={1}
            style={{
              flex: 1, resize: "none", border: `1px solid ${T.gray300}`, borderRadius: 12,
              padding: "9px 14px", fontSize: 13.5, fontFamily: "'DM Sans', sans-serif",
              color: T.gray800, outline: "none", lineHeight: 1.5, maxHeight: 80,
              background: "#fff", transition: "border .15s"
            }}
            onFocus={e => e.target.style.borderColor = T.teal}
            onBlur={e => e.target.style.borderColor = T.gray300}
          />
          <button onClick={() => send(input)} disabled={loading || !input.trim()} style={{
            width: 38, height: 38, borderRadius: "50%", border: "none",
            background: loading || !input.trim() ? T.gray300 : T.teal,
            cursor: loading || !input.trim() ? "not-allowed" : "pointer",
            display: "flex", alignItems: "center", justifyContent: "center",
            transition: "background .15s", flexShrink: 0
          }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </div>
        <div style={{ display: "flex", justifyContent: "flex-end" }}>
          <button
            type="button"
            onClick={() => {
              if (messages.length <= 1) return;
              if (!window.confirm("Start a new session? The current chat will be cleared from this browser. Export from My Progress first if you need to keep it.")) return;
              newSession();
            }}
            disabled={loading}
            style={{
              fontFamily: "'DM Sans', sans-serif", fontSize: 11.5, fontWeight: 500,
              padding: "4px 12px", borderRadius: 6,
              border: `1px solid ${T.gray300}`, background: "#fff",
              color: T.gray500, cursor: loading ? "not-allowed" : "pointer",
              opacity: loading ? 0.5 : 1, transition: "all .15s",
            }}
            onMouseOver={e => { if (!loading) { e.currentTarget.style.borderColor = T.red; e.currentTarget.style.color = T.red; } }}
            onMouseOut={e => { e.currentTarget.style.borderColor = T.gray300; e.currentTarget.style.color = T.gray500; }}
          >
            ↺ New session
          </button>
        </div>
      </div>
    </div>
  );
}

function buildConvProps(moduleId, moduleLabel, patient, conditionId, conditionLabel) {
  const conversationKey = makeConversationKey(moduleId, patient.id, conditionId);
  return {
    conversationKey,
    conversationLabel: moduleLabel,
    conversationMeta: {
      moduleId,
      profileId: patient.id,
      profileName: patient.name,
      condition: conditionId,
      conditionLabel,
    },
    patient,
  };
}

/** Build a system prompt for a standalone instrument assessment session.
 *  PASE instruments (2a–2d) each get a dedicated agent persona and isolated scope. */
function buildAssessmentSystem(instrumentKey, patient) {
  const label = INSTRUMENT_LABELS[instrumentKey] ?? instrumentKey;
  const adminText = INSTRUMENT_ADMIN_TEXTS[instrumentKey] ?? "";
  const paseAgent = getPaseAgent(instrumentKey);

  const identityBlock = paseAgent
    ? `You are the ${paseAgent.aiName} — a dedicated, isolated assessment agent for ${paseAgent.label}.
You administer ONLY ${paseAgent.domainFocus}. You do NOT ask about other PA domains (job, transport, domestic, leisure) or other instruments.
If the participant asks about a different domain, warmly redirect: "I'm here specifically for ${paseAgent.label.split("—")[1]?.trim() ?? "this assessment"} — your PA Coach can help with broader activity planning."`
    : `You are ObesityCare AI administering the ${label} research instrument to a clinical trial participant.`;

  return `${identityBlock}
Work through each question one at a time in a warm, non-judgmental way (PACE: Partnership, Acceptance, Compassion, Empowerment).

Patient: ${patient.name}, Trial: ${patient.trial}.

CRITICAL — QUIZ TAGS (REQUIRED ON EVERY SCORED ITEM): Every response that asks a scored item MUST end with the exact [QUIZ: ...] tag on the very last line — including item 2, 3, 4, etc. Copy the tag verbatim from the instrument text below. Zero characters after the closing ]. Example last line for any PASE/SERPA confidence item:
${QUIZ_SCALE_TAG}
If a question has no [QUIZ:] tag in the instrument text (e.g. free-text income), do not include one.

${adminText}

RULES:
- Ask exactly one question per response; wait for the participant's answer before continuing.
- Structure each turn as: (optional) one brief acknowledgment sentence, then the next numbered question, then the [QUIZ:] tag on its own last line.
- Brief acknowledgments must NOT replace the [QUIZ:] tag — the tag is still required every time.
- Follow all STOP RULE instructions in the instrument text.
- Never diagnose or recommend treatment changes. Flag distress with [ESCALATE].
- When the full instrument is complete and you have output [INSTRUMENT_DATA: {...}], briefly summarize the scores in plain language and note what they suggest for their activity plan in THIS domain only.`;
}

// ─── Modules ──────────────────────────────────────────────────
function ChatModule({ systems, patient, programDay, programWeek, conditionId, conditionLabel }) {
  const [assessKey, setAssessKey] = useState(null);

  // RAG is active in Conditions B and C (theory-informed) for the main PA Coach only.
  const withTheory = CONDITIONS.find(c => c.id === conditionId)?.includeTheory ?? false;
  const ragScores = withTheory ? readProfileScores(patient.id) : {};

  const paseAgent = assessKey ? getPaseAgent(assessKey) : null;
  const isPaseAssessment = Boolean(paseAgent);
  const assessLabel = isPaseAssessment
    ? paseAgent.label
    : (INSTRUMENT_LABELS[assessKey] ?? assessKey);
  const assessInitials = isPaseAssessment ? paseAgent.aiInitials : "AI";
  const assessModuleId = isPaseAssessment ? `assess_${paseAgent.code}` : `assess_${assessKey}`;

  if (assessKey) {
    const assessSys = { chat: buildAssessmentSystem(assessKey, patient) };
    const assessConv = buildConvProps(
      assessModuleId,
      isPaseAssessment ? `${paseAgent.code} ${paseAgent.aiName}` : `${INSTRUMENT_SHORT_LABELS[assessKey] ?? assessKey} Assessment`,
      patient,
      conditionId,
      conditionLabel,
    );
    const assessIntro = isPaseAssessment
      ? `Hello ${patient.name}! I'm the ${paseAgent.aiName} — I administer ${paseAgent.label} only.\n\nI'll ask about your confidence for ${paseAgent.domainFocus} over the next week, one question at a time. Tap the answer buttons when they appear, or type your response.\n\nWhen you're ready, click below to begin.`
      : `Let's go through the ${assessLabel} together, ${patient.name}. I'll ask each question one at a time — you can tap an answer button when they appear, or type your response.\n\nWhen you're ready, click the button below to begin.`;

    return (
      <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
        <div style={{
          padding: "8px 16px", borderBottom: `1px solid ${T.tealBorder ?? T.teal}`,
          display: "flex", alignItems: "center", gap: 10,
          background: T.tealLight, flexShrink: 0, flexWrap: "wrap",
        }}>
          <button
            onClick={() => setAssessKey(null)}
            style={{
              fontSize: 12, fontWeight: 600, color: T.teal, background: "none",
              border: "none", cursor: "pointer", padding: "2px 0",
              fontFamily: "'DM Sans', sans-serif",
            }}
          >← PA Coach</button>
          <span style={{ color: T.gray400, fontSize: 12 }}>|</span>
          <span style={{ fontSize: 12, fontWeight: 600, color: T.teal }}>
            {isPaseAssessment ? `${paseAgent.code} · ${paseAgent.aiName}` : `📋 ${assessLabel}`}
          </span>
          {isPaseAssessment && (
            <span style={{ fontSize: 10, color: T.gray500, marginLeft: "auto" }}>
              Dedicated agent · separate conversation log
            </span>
          )}
        </div>
        <ChatEngine
          key={assessConv.conversationKey}
          systems={assessSys}
          systemKey="chat"
          placeholder="Select an answer button or type a free-text response..."
          intro={assessIntro}
          quickReplies={["▶ Begin the assessment"]}
          persistInstrument={{ key: assessKey, label: assessLabel }}
          assistantInitials={assessInitials}
          {...assessConv}
          patient={patient}
        />
      </div>
    );
  }

  const conv = buildConvProps("chat", "PA Coach", patient, conditionId, conditionLabel);
  const OTHER_ASSESS_KEYS = ["inst1", "inst6"];

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      <div style={{
        padding: "6px 16px", borderBottom: `1px solid ${T.gray200}`,
        display: "flex", flexDirection: "column", gap: 6,
        background: T.gray100 ?? "#f8f9fa", flexShrink: 0,
      }}>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
          <span style={{ fontSize: 10, fontWeight: 700, color: T.gray500, letterSpacing: ".06em", textTransform: "uppercase", marginRight: 2 }}>
            PASE agents (4 separate AIs)
          </span>
          {PASE_INSTRUMENT_AGENTS.map(agent => (
            <button
              key={agent.key}
              onClick={() => setAssessKey(agent.key)}
              title={agent.label}
              style={{
                fontSize: 11, padding: "4px 11px", borderRadius: 12,
                border: `1px solid ${T.teal}`, background: "#fff",
                color: T.teal, cursor: "pointer",
                fontFamily: "'DM Sans', sans-serif", transition: "all .15s",
                fontWeight: 600,
              }}
              onMouseOver={e => { e.currentTarget.style.background = T.tealLight; }}
              onMouseOut={e => { e.currentTarget.style.background = "#fff"; }}
            >
              {agent.shortLabel}
            </button>
          ))}
        </div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
          <span style={{ fontSize: 10, fontWeight: 600, color: T.gray400, marginRight: 2 }}>
            Other
          </span>
          {OTHER_ASSESS_KEYS.map(k => (
            <button
              key={k}
              onClick={() => setAssessKey(k)}
              style={{
                fontSize: 11, padding: "3px 10px", borderRadius: 12,
                border: `1px solid ${T.gray300}`, background: "#fff",
                color: T.gray600, cursor: "pointer",
                fontFamily: "'DM Sans', sans-serif", transition: "all .15s",
              }}
              onMouseOver={e => { e.currentTarget.style.borderColor = T.teal; e.currentTarget.style.color = T.teal; }}
              onMouseOut={e => { e.currentTarget.style.borderColor = T.gray300; e.currentTarget.style.color = T.gray600; }}
            >
              {INSTRUMENT_SHORT_LABELS[k] ?? k}
            </button>
          ))}
        </div>
      </div>
      <ChatEngine
        key={conv.conversationKey}
        systems={systems}
        systemKey="chat"
        placeholder="Ask about activity goals, barriers, motivation, or your treatment..."
        intro={`Hello ${patient.name}! I'm your Confident Moves PA coach. I'm here to help you build a sustainable, personalized physical activity routine that works with your body and your life.\n\nToday is program day ${programDay} (week ${programWeek}) — every step counts. We can work on setting goals, finding activities you enjoy, or problem-solving barriers like ${patient.pa.topBarrier}.\n\nWhat's on your mind today?`}
        quickReplies={["Help me set a PA goal", "I'm struggling to stay motivated", "What activity suits my fitness level?", "How does exercise help with my weight loss?", "Had some side effects this week"]}
        ragEnabled={withTheory}
        ragScores={ragScores}
        {...conv}
        patient={patient}
      />
    </div>
  );
}

function CheckInModule({ systems, patient, programDay, programWeek, conditionId, conditionLabel }) {
  const conv = buildConvProps("checkin", "Daily Check-in", patient, conditionId, conditionLabel);
  return (
    <ChatEngine
      key={conv.conversationKey}
      systems={systems}
      systemKey="checkin"
      placeholder="Answer today's check-in questions..."
      intro={`Good morning ${patient.name}! Time for your program day ${programDay} check-in (week ${programWeek}) — it takes about 2 minutes and helps your care team track your whole-person progress.\n\nI'll ask a few short questions covering activity, energy, appetite, and how you're feeling. Let's start: On a scale of 1–10, how would you rate your hunger and appetite today compared to before you started the program?`}
      quickReplies={["1–3 (much less hungry)", "4–6 (somewhat less hungry)", "7–10 (about the same)", "I forgot my medication today"]}
      {...conv}
    />
  );
}

function EligibilityModule({ systems, patient, conditionId, conditionLabel }) {
  const conv = buildConvProps("eligibility", "Eligibility Screener", patient, conditionId, conditionLabel);
  return (
    <ChatEngine
      key={conv.conversationKey}
      systems={systems}
      systemKey="eligibility"
      placeholder="Answer the screening questions..."
      intro={`Welcome to the STEP-OB-24 trial eligibility screener.\n\nThis brief questionnaire helps determine if you may qualify for our extended semaglutide therapy study. This is not a medical assessment — your care team will review and confirm any eligibility decision.\n\nLet's start with the basics: What is your current height and weight? (You can give approximate values)`}
      quickReplies={["I'm 5'6\", 210 lbs", "I'm 5'8\", 230 lbs", "I'm 5'4\", 195 lbs", "I'd rather answer questions one by one"]}
      {...conv}
    />
  );
}

function EducationModule({ systems, patient, programDay, conditionId, conditionLabel }) {
  const conv = buildConvProps("education", "Learn & Explore", patient, conditionId, conditionLabel);
  return (
    <ChatEngine
      key={conv.conversationKey}
      systems={systems}
      systemKey="education"
      placeholder="Ask about physical activity, nutrition, your treatment, or obesity medicine..."
      intro={`Welcome to the Confident Moves Learning Hub! I can explain physical activity guidelines in plain language, share evidence-based tips for building sustainable habits, help you understand your treatment, or answer questions about obesity medicine and whole-person health.\n\nYou're on program day ${programDay}. What would you like to learn about today?`}
      quickReplies={[
        "How much PA do I need each week?",
        "Best exercises with obesity or joint pain?",
        "How does GLP-1 therapy work?",
        "Why does motivation fluctuate?",
        "How does PA help beyond weight loss?",
      ]}
      {...conv}
    />
  );
}

const fieldInputStyle = {
  width: "100%", padding: "6px 8px", borderRadius: 6, border: `1px solid ${T.gray300}`,
  fontSize: 12, fontFamily: "'DM Sans', sans-serif", background: "#fff",
};

function ProfileEditorForm({ draft, onChange, onSave, onResetOne, onResetAll }) {
  const set = (path, value) => {
    const next = JSON.parse(JSON.stringify(draft));
    if (path === "conditions") next.conditions = String(value).split(",").map(s => s.trim()).filter(Boolean);
    else if (path === "medications") next.medications = String(value).split(",").map(s => s.trim()).filter(Boolean);
    else if (path.startsWith("pa.")) next.pa[path.slice(3)] = value;
    else if (path.startsWith("weight.")) next.weight[path.slice(7)] = Number(value) || value;
    else if (path.startsWith("bmi.")) next.bmi[path.slice(4)] = Number(value) || value;
    else next[path] = path === "totalWeeks" || path === "startProgramDay" || path === "adherence" ? Number(value) || 0 : value;
    onChange(next);
  };

  return (
    <div style={{ marginTop: 8, padding: "10px", background: "#fff", borderRadius: 8, border: `1px solid ${T.gray200}` }}>
      <div style={{ fontSize: 11, color: T.gray500, marginBottom: 8, lineHeight: 1.45 }}>
        Edit fields for <strong>{draft.name}</strong> ({draft.id}). Saved to this browser.
      </div>
      <div style={{ display: "grid", gap: 6, maxHeight: 280, overflowY: "auto" }}>
        {[
          ["name", "Name", draft.name, "text"],
          ["trial", "Trial", draft.trial, "text"],
          ["drug", "Drug", draft.drug, "text"],
          ["totalWeeks", "Total weeks", draft.totalWeeks, "number"],
          ["startProgramDay", "Start program day", draft.startProgramDay, "number"],
          ["weight.baseline", "Weight baseline (lb)", draft.weight.baseline, "number"],
          ["weight.current", "Weight current (lb)", draft.weight.current, "number"],
          ["weight.goal", "Weight goal (lb)", draft.weight.goal, "number"],
          ["bmi.baseline", "BMI baseline", draft.bmi.baseline, "number"],
          ["bmi.current", "BMI current", draft.bmi.current, "number"],
          ["conditions", "Conditions (comma-separated)", draft.conditions.join(", "), "text"],
          ["medications", "Medications (comma-separated)", draft.medications.join(", "), "text"],
          ["pa.topBarrier", "Top PA barrier", draft.pa.topBarrier, "text"],
          ["pa.favoriteActivity", "Favorite activity", draft.pa.favoriteActivity, "text"],
          ["pa.weeklyGoalMins", "Weekly PA goal (min)", draft.pa.weeklyGoalMins, "number"],
          ["pa.goalDays", "Active days goal / week", draft.pa.goalDays, "number"],
        ].map(([path, label, val, type]) => (
          <div key={path}>
            <label style={{ display: "block", fontSize: 10.5, color: T.gray600, marginBottom: 2 }}>{label}</label>
            <input type={type} value={val} onChange={e => set(path, e.target.value)} style={fieldInputStyle} />
          </div>
        ))}
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 10 }}>
        <button type="button" onClick={onSave} style={{
          fontFamily: "'DM Sans', sans-serif", fontWeight: 600, fontSize: 11.5,
          padding: "6px 10px", borderRadius: 6, border: "none", background: T.teal, color: "#fff", cursor: "pointer",
        }}>Save profile</button>
        <button type="button" onClick={onResetOne} style={{
          fontFamily: "'DM Sans', sans-serif", fontWeight: 600, fontSize: 11.5,
          padding: "6px 10px", borderRadius: 6, border: `1px solid ${T.amber}`, background: T.amberLight, color: T.amber, cursor: "pointer",
        }}>Reset to default</button>
        <button type="button" onClick={onResetAll} style={{
          fontFamily: "'DM Sans', sans-serif", fontWeight: 600, fontSize: 11.5,
          padding: "6px 10px", borderRadius: 6, border: `1px solid ${T.gray300}`, background: "#fff", color: T.gray600, cursor: "pointer",
        }}>Reset all 3 profiles</button>
      </div>
    </div>
  );
}

function ResearchSidebarPanel({
  profiles, activeProfileId, activeCondition, editingProfile, profileDraft,
  onSelectProfile, onSelectCondition, onToggleEdit, onDraftChange, onSaveProfile, onResetProfile, onResetAllProfiles,
}) {
  const activeCond = CONDITIONS.find(c => c.id === activeCondition) ?? CONDITIONS[0];
  const tabBtn = (active) => ({
    fontFamily: "'DM Sans', sans-serif", fontWeight: 600, fontSize: 12,
    padding: "8px 10px", borderRadius: 8, cursor: "pointer", textAlign: "left", width: "100%",
    background: active ? T.tealLight : T.gray50,
    border: active ? `1px solid ${T.tealMid}` : `1px solid ${T.gray200}`,
    color: active ? T.tealDark : T.gray700,
  });

  return (
    <div style={{
      padding: "9px 11px 11px", margin: "0 10px 8px", borderRadius: 10,
      background: T.purpleLight, border: `1px solid ${T.gray300}`,
    }}>
      <div style={{ fontSize: 12, fontWeight: 600, color: T.purple, textTransform: "uppercase", letterSpacing: ".06em", marginBottom: 6 }}>
        Research testing
      </div>
      <p style={{ fontSize: 11, color: T.gray600, lineHeight: 1.45, marginBottom: 8 }}>
        A = baseline. B = self-efficacy + MI/PACE + BCTs. C = same theories + all 6 instruments in the coaching prompt. Each profile × condition saves a separate chat log.
      </p>

      <div style={{ fontSize: 11, fontWeight: 600, color: T.gray500, marginBottom: 4 }}>Profile</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 4, marginBottom: 8 }}>
        {profiles.map(p => (
          <button key={p.id} type="button" style={tabBtn(p.id === activeProfileId)} onClick={() => onSelectProfile(p.id)}>
            {p.name} <span style={{ fontWeight: 400, opacity: 0.75 }}>({p.id})</span>
          </button>
        ))}
      </div>
      <button type="button" onClick={onToggleEdit} style={{
        ...tabBtn(editingProfile), marginBottom: editingProfile ? 0 : 8, fontSize: 11.5,
      }}>
        {editingProfile ? "Hide profile editor" : "Edit active profile…"}
      </button>
      {editingProfile && profileDraft && (
        <ProfileEditorForm
          draft={profileDraft}
          onChange={onDraftChange}
          onSave={onSaveProfile}
          onResetOne={onResetProfile}
          onResetAll={onResetAllProfiles}
        />
      )}

      <div style={{ fontSize: 11, fontWeight: 600, color: T.gray500, margin: "10px 0 4px" }}>Condition</div>
      <div style={{ display: "flex", gap: 4, marginBottom: 6 }}>
        {CONDITIONS.map(c => (
          <button key={c.id} type="button" onClick={() => onSelectCondition(c.id)} style={{
            flex: 1, fontFamily: "'DM Sans', sans-serif", fontWeight: 700, fontSize: 13,
            padding: "8px 0", borderRadius: 8, cursor: "pointer",
            background: activeCondition === c.id ? T.teal : "#fff",
            color: activeCondition === c.id ? "#fff" : T.gray700,
            border: `1px solid ${activeCondition === c.id ? T.teal : T.gray300}`,
          }}>{c.id}</button>
        ))}
      </div>
      <div style={{ fontSize: 11, color: T.gray600, lineHeight: 1.4 }}>
        <strong>{activeCond.label}:</strong> {activeCond.description}
      </div>
    </div>
  );
}

function ConversationCard({ entry, onClear }) {
  const [expanded, setExpanded] = useState(false);
  const userCount = entry.messages.filter(m => m.role === "user").length;
  const aiCount = entry.messages.filter(m => m.role === "assistant").length;
  const lastUpdated = entry.lastUpdated
    ? new Date(entry.lastUpdated).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })
    : "—";
  const subtitle = [
    entry.profileName && `${entry.profileName}`,
    entry.condition && `Cond ${entry.condition}`,
    `${userCount} user · ${aiCount} AI`,
    lastUpdated,
  ].filter(Boolean).join(" · ");

  return (
    <div style={{ background: "#fff", border: `1px solid ${T.gray200}`, borderRadius: 10, marginBottom: 10, overflow: "hidden" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "11px 14px", cursor: "pointer" }} onClick={() => setExpanded(x => !x)}>
        <div style={{
          width: 32, height: 32, borderRadius: 8, background: T.tealLight,
          display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
        }}>
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke={T.teal} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 13.5, fontWeight: 600, color: T.gray800 }}>{entry.moduleLabel}</div>
          <div style={{ fontSize: 11.5, color: T.gray500, marginTop: 1 }}>{subtitle}</div>
        </div>
        <button
          type="button"
          onClick={e => { e.stopPropagation(); onClear(); }}
          style={{
            fontFamily: "'DM Sans', sans-serif", fontSize: 11.5, fontWeight: 600,
            padding: "4px 10px", borderRadius: 6, border: `1px solid ${T.gray300}`,
            background: "#fff", color: T.gray500, cursor: "pointer", flexShrink: 0,
          }}
        >Clear</button>
        <svg
          width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={T.gray400} strokeWidth="2.5"
          strokeLinecap="round" strokeLinejoin="round"
          style={{ flexShrink: 0, transform: expanded ? "rotate(180deg)" : "none", transition: "transform .2s" }}
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </div>

      {expanded && (
        <div style={{ borderTop: `1px solid ${T.gray100}`, padding: "14px 16px", display: "flex", flexDirection: "column", gap: 10, maxHeight: 420, overflowY: "auto" }}>
          {entry.messages.map((m, i) => (
            <div key={i} style={{
              display: "flex", gap: 8, alignItems: "flex-start",
              flexDirection: m.role === "user" ? "row-reverse" : "row",
              maxWidth: "92%", alignSelf: m.role === "user" ? "flex-end" : "flex-start",
            }}>
              <Avatar
                initials={m.role === "user" ? patientInitials(entry.profileName ?? "User") : "AI"}
                color={m.role === "user" ? T.chatUserAvatarFg : T.chatAiAvatarFg}
                bg={m.role === "user" ? T.chatUserAvatarBg : T.chatAiAvatarBg}
                size={26}
              />
              <div>
                <div style={{
                  padding: "9px 13px",
                  borderRadius: m.role === "user" ? "14px 4px 14px 14px" : "4px 14px 14px 14px",
                  background: m.role === "user" ? T.chatUserBubble : T.chatAiBubble,
                  border: m.role === "user" ? "1px solid transparent" : `1px solid ${T.chatAiBubbleBorder}`,
                  color: m.role === "user" ? T.chatUserText : T.chatAiText,
                  fontSize: 12.5, lineHeight: 1.6, whiteSpace: "pre-wrap",
                }}>
                  {m.content || <em style={{ opacity: 0.5 }}>(empty)</em>}
                </div>
                {m.ts && (
                  <div style={{ fontSize: 10, color: T.gray400, marginTop: 2, textAlign: m.role === "user" ? "right" : "left" }}>
                    {new Date(m.ts).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── LLM-as-judge — DISABLED (restore when benchmarking resumes) ───────────────
/*
const JUDGE_RUBRIC_CRITERIA = [
  { id: "open_question",   label: "Open question (MI)",      desc: "AI asks rather than tells; avoids yes/no or leading questions" },
  { id: "affirm",          label: "Affirmation (PACE/A)",    desc: "Validates experience without judgment; acknowledges effort" },
  { id: "reflect_summary", label: "Reflect / Summarize (MI)",desc: "Mirrors patient words back; summarises before moving on" },
  { id: "se_source",       label: "SE source used",          desc: "References mastery, vicarious, verbal persuasion, or affective state" },
  { id: "bct",             label: "BCT present",             desc: "Goal-setting, action planning, problem-solving, or confidence check" },
  { id: "personalization", label: "Personalisation",         desc: "References patient's specific barrier, score, or stated context" },
];

const JUDGE_SCORE_LABELS = ["0 — Absent", "1 — Weak", "2 — Adequate", "3 — Strong"];

const JUDGE_RESULTS_KEY = "confidentMoves_judge_results";

function readJudgeResults() {
  try { return JSON.parse(localStorage.getItem(JUDGE_RESULTS_KEY) ?? "[]") ?? []; } catch { return []; }
}
function saveJudgeResult(result) {
  try {
    const prev = readJudgeResults();
    prev.unshift(result);
    localStorage.setItem(JUDGE_RESULTS_KEY, JSON.stringify(prev.slice(0, 50)));
  } catch {}
}

function buildJudgeSystemPrompt() {
  return `You are a research assistant scoring AI chatbot responses for a PA behaviour-change study. You will receive one AI assistant message. Score it on exactly 6 criteria using integers 0–3.

Criteria:
${JUDGE_RUBRIC_CRITERIA.map((c, i) => `${i + 1}. ${c.label}: ${c.desc}\n   0=Absent, 1=Weak, 2=Adequate, 3=Strong`).join("\n")}

Reply ONLY with valid JSON — no prose, no markdown:
{"open_question":0,"affirm":0,"reflect_summary":0,"se_source":0,"bct":0,"personalization":0,"rationale":"one sentence explaining the lowest score"}`;
}

async function scoreResponseWithJudge(aiText, conditionId) {
  const msgs = [{ role: "user", content: `AI response to score (Condition ${conditionId}):\n\n${aiText}` }];
  let raw = "";
  // PRODUCTION: same callLLM() path; judge can use Haiku with maxTokens:200 or Sonnet for stricter scoring.
  // Consider non-streaming on server for judge (JSON-only output, easier to validate).
  await callGroq(msgs, buildJudgeSystemPrompt(), chunk => { raw = chunk; }, { maxTokens: JUDGE_MAX_TOKENS, minIntervalMs: 3000 });
  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start === -1 || end === -1) throw new Error("Judge returned no JSON");
  return JSON.parse(raw.slice(start, end + 1));
}

function JudgePanel() {
  const [convStore] = useState(() => readConvStore());
  const [selectedKey, setSelectedKey] = useState("");
  const [selectedMsgIdx, setSelectedMsgIdx] = useState(-1);
  const [judging, setJudging] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [history, setHistory] = useState(() => readJudgeResults());

  const entries = Object.values(convStore).filter(e => e.messages?.some(m => m.role === "assistant" && m.content?.length > 30));
  const selectedEntry = entries.find(e => e.conversationKey === selectedKey);
  const assistantMsgs = selectedEntry?.messages?.filter(m => m.role === "assistant" && m.content?.length > 30) ?? [];

  const runJudge = async () => {
    if (!selectedEntry || selectedMsgIdx < 0) return;
    const msg = assistantMsgs[selectedMsgIdx];
    setJudging(true); setResult(null); setError("");
    try {
      const scores = await scoreResponseWithJudge(msg.content, selectedEntry.condition ?? "?");
      const total = JUDGE_RUBRIC_CRITERIA.reduce((s, c) => s + (scores[c.id] ?? 0), 0);
      const r = {
        scoredAt: new Date().toISOString(),
        profileName: selectedEntry.profileName,
        condition: selectedEntry.condition,
        moduleLabel: selectedEntry.moduleLabel,
        messagePreview: msg.content.slice(0, 120),
        scores,
        total,
        maxTotal: JUDGE_RUBRIC_CRITERIA.length * 3,
        rationale: scores.rationale ?? "",
      };
      setResult(r);
      saveJudgeResult(r);
      setHistory(readJudgeResults());
    } catch (e) {
      setError(String(e?.message ?? e));
    }
    setJudging(false);
  };

  const dlHistory = () => {
    const h = readJudgeResults();
    if (!h.length) return;
    triggerDownload(`judge-scores-${Date.now()}.json`, "application/json", JSON.stringify(h, null, 2));
  };

  const scoreColor = (v) => v <= 0 ? T.red : v === 1 ? T.amber : v === 2 ? T.gray600 : T.teal;
  const btnBase = { border: "none", fontFamily: "'DM Sans', sans-serif", fontWeight: 600, fontSize: 12, padding: "7px 12px", borderRadius: 8, cursor: "pointer" };

  return (
    <div style={{ marginBottom: 24, padding: 16, background: "#fff", border: `1px solid ${T.gray200}`, borderRadius: 10 }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: T.gray500, textTransform: "uppercase", letterSpacing: ".05em", marginBottom: 4 }}>
        LLM-as-judge scorer (benchmarking)
      </div>
      <p style={{ fontSize: 12.5, color: T.gray600, lineHeight: 1.55, marginBottom: 12 }}>
        Pick any AI response from your conversation logs and score it on 6 theory-fidelity criteria (0–3 each, max 18). Compare across Conditions A / B / C to benchmark theory alignment.
      </p>

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 10 }}>
        <select value={selectedKey} onChange={e => { setSelectedKey(e.target.value); setSelectedMsgIdx(-1); setResult(null); }}
          style={{ fontSize: 12, padding: "6px 10px", borderRadius: 6, border: `1px solid ${T.gray300}`, flex: 1, minWidth: 160, fontFamily: "'DM Sans', sans-serif" }}>
          <option value="">— Choose conversation log —</option>
          {entries.map(e => (
            <option key={e.conversationKey} value={e.conversationKey}>
              {e.profileName} · Cond {e.condition} · {e.moduleLabel}
            </option>
          ))}
        </select>
        {assistantMsgs.length > 0 && (
          <select value={selectedMsgIdx} onChange={e => { setSelectedMsgIdx(Number(e.target.value)); setResult(null); }}
            style={{ fontSize: 12, padding: "6px 10px", borderRadius: 6, border: `1px solid ${T.gray300}`, flex: 2, minWidth: 200, fontFamily: "'DM Sans', sans-serif" }}>
            <option value={-1}>— Choose AI response to score —</option>
            {assistantMsgs.map((m, i) => (
              <option key={i} value={i}>Response #{i + 1}: {m.content.slice(0, 60)}…</option>
            ))}
          </select>
        )}
      </div>

      {selectedMsgIdx >= 0 && assistantMsgs[selectedMsgIdx] && (
        <div style={{ fontSize: 11.5, color: T.gray600, background: T.gray100 ?? "#f8fafc", padding: "8px 12px", borderRadius: 6, marginBottom: 10, whiteSpace: "pre-wrap", maxHeight: 80, overflowY: "auto", lineHeight: 1.5 }}>
          {assistantMsgs[selectedMsgIdx].content}
        </div>
      )}

      <button onClick={runJudge} disabled={judging || selectedMsgIdx < 0}
        style={{ ...btnBase, background: judging || selectedMsgIdx < 0 ? T.gray300 : T.teal, color: "#fff", cursor: judging || selectedMsgIdx < 0 ? "not-allowed" : "pointer", marginBottom: 12 }}>
        {judging ? "Scoring…" : "▶ Score this response"}
      </button>

      {error && <div style={{ fontSize: 12, color: T.red, marginBottom: 10 }}>Error: {error}</div>}

      {result && (
        <div style={{ background: T.tealLight, border: `1px solid ${T.teal}`, borderRadius: 8, padding: "12px 14px", marginBottom: 12 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: T.teal, marginBottom: 8 }}>
            Total: {result.total} / {result.maxTotal} — {result.profileName} · Condition {result.condition}
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "5px 16px", marginBottom: 8 }}>
            {JUDGE_RUBRIC_CRITERIA.map(c => (
              <div key={c.id} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
                <span style={{ fontWeight: 700, color: scoreColor(result.scores[c.id] ?? 0), minWidth: 14 }}>{result.scores[c.id] ?? 0}</span>
                <span style={{ color: T.gray700 }}>{c.label}</span>
              </div>
            ))}
          </div>
          {result.rationale && <div style={{ fontSize: 11.5, color: T.gray600, fontStyle: "italic" }}>Note: {result.rationale}</div>}
        </div>
      )}

      {history.length > 0 && (
        <details>
          <summary style={{ fontSize: 12, fontWeight: 600, color: T.gray600, cursor: "pointer", marginBottom: 6 }}>
            Score history ({history.length} scored)
          </summary>
          <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 8 }}>
            {history.slice(0, 10).map((h, i) => (
              <div key={i} style={{ fontSize: 11.5, display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                <span style={{ fontWeight: 700, color: T.teal, minWidth: 40 }}>{h.total}/{h.maxTotal}</span>
                <span style={{ color: T.gray600 }}>{h.profileName} · Cond {h.condition} · {h.moduleLabel}</span>
                <span style={{ color: T.gray400 }}>{h.messagePreview?.slice(0, 50)}…</span>
              </div>
            ))}
          </div>
          <button onClick={dlHistory} style={{ ...btnBase, marginTop: 8, background: "#fff", color: T.teal, border: `1px solid ${T.teal}` }}>
            Export all scores JSON
          </button>
        </details>
      )}
    </div>
  );
}
*/

function FeasibilityPanel() {
  const [report, setReport] = useState(() => buildFeasibilityReport());
  const refresh = () => setReport(buildFeasibilityReport());
  const { apiMetrics, profileReports } = report;

  const statusStyle = (status) => {
    if (status === "complete") return { bg: T.tealLight, color: T.teal, label: "Complete" };
    if (status === "in_progress") return { bg: T.amberLight, color: T.amber, label: "In progress" };
    return { bg: T.gray100 ?? "#f1f5f9", color: T.gray500, label: "Not started" };
  };

  const btnBase = {
    border: "none", fontFamily: "'DM Sans', sans-serif",
    fontWeight: 600, fontSize: 12, padding: "7px 12px", borderRadius: 8, cursor: "pointer",
  };

  return (
    <div style={{
      marginBottom: 24, padding: 16, background: "#fff",
      border: `1px solid ${T.teal}`, borderRadius: 10,
    }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: T.teal, textTransform: "uppercase", letterSpacing: ".05em", marginBottom: 4 }}>
        Feasibility dashboard (Aim 3)
      </div>
      <p style={{ fontSize: 12.5, color: T.gray600, lineHeight: 1.55, marginBottom: 14 }}>
        Pilot metrics for your advisor: instrument completion, API usage, rate limits, and where data is stored in this browser.
      </p>

      {/*
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: 10, marginBottom: 16 }}>
        {[
          { label: "API calls", value: report.estimatedApiCalls },
          { label: "Rate limits (429)", value: apiMetrics.rateLimitHits, warn: apiMetrics.rateLimitHits > 0 },
          { label: "Chat logs", value: report.conversationLogs },
          { label: "User turns", value: report.totalUserTurns },
          { label: "Instruments saved", value: report.instrumentSubmissions },
          // PRODUCTION: display LLM_MODEL from server (e.g. "haiku-4.5") instead of GROQ_MODEL
          { label: "Model", value: GROQ_MODEL.split("-").slice(-2).join("-"), small: true },
        ].map(({ label, value, warn, small }) => (
          <div key={label} style={{ padding: "10px 12px", background: warn ? T.amberLight : T.gray100 ?? "#f8fafc", borderRadius: 8, border: `1px solid ${warn ? T.amber : T.gray200}` }}>
            <div style={{ fontSize: 10, color: T.gray500, marginBottom: 4 }}>{label}</div>
            <div style={{ fontSize: small ? 11 : 18, fontWeight: 700, color: warn ? T.amber : T.gray800, wordBreak: "break-word" }}>{value}</div>
          </div>
        ))}
      </div>
      */}

      {apiMetrics.rateLimitHits > 0 && (
        <div style={{ fontSize: 12, color: T.amber, background: T.amberLight, padding: "10px 12px", borderRadius: 8, marginBottom: 14, lineHeight: 1.5 }}>
          Rate limit hit {apiMetrics.rateLimitHits} time{apiMetrics.rateLimitHits !== 1 ? "s" : ""}.
          Assessments need ~1 API call per question — pause ~4s between answers. App auto-retries; if it still fails, wait 60s and continue (your answers are saved in the chat log).
        </div>
      )}

      {profileReports.map(pr => (
        <div key={pr.profileId} style={{ marginBottom: 14 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: T.gray700, marginBottom: 8 }}>{pr.profileName} ({pr.profileId})</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {pr.paseAgents.map(a => {
              const st = statusStyle(a.status);
              return (
                <div key={a.code} style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 12, flexWrap: "wrap" }}>
                  <span style={{ width: 72, fontWeight: 600, color: T.gray600 }}>{a.shortLabel}</span>
                  <span style={{ fontSize: 10, fontWeight: 600, padding: "2px 8px", borderRadius: 10, background: st.bg, color: st.color }}>{st.label}</span>
                  <div style={{ flex: 1, minWidth: 100, height: 6, background: T.gray200, borderRadius: 3, overflow: "hidden" }}>
                    <div style={{ width: `${a.completionPct}%`, height: "100%", background: a.status === "complete" ? T.teal : T.amber, borderRadius: 3 }} />
                  </div>
                  <span style={{ color: T.gray500, minWidth: 88 }}>{a.answered}/{a.expected} items</span>
                  {a.totalScore != null && <span style={{ color: T.teal, fontWeight: 600 }}>score {a.totalScore}</span>}
                </div>
              );
            })}
            <div style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 12 }}>
              <span style={{ width: 72, fontWeight: 600, color: T.gray600 }}>SERPA</span>
              <span style={{ fontSize: 10, fontWeight: 600, padding: "2px 8px", borderRadius: 10, background: statusStyle(pr.serpa.complete ? "complete" : pr.serpa.answered ? "in_progress" : "not_started").bg, color: statusStyle(pr.serpa.complete ? "complete" : pr.serpa.answered ? "in_progress" : "not_started").color }}>
                {pr.serpa.complete ? "Complete" : pr.serpa.answered ? "In progress" : "Not started"}
              </span>
              <span style={{ color: T.gray500 }}>{pr.serpa.answered}/{pr.serpa.expected}</span>
              {pr.serpa.totalScore != null && <span style={{ color: T.teal, fontWeight: 600 }}>score {pr.serpa.totalScore}</span>}
            </div>
          </div>
        </div>
      ))}

      <details style={{ marginBottom: 12 }}>
        <summary style={{ fontSize: 12, fontWeight: 600, color: T.gray600, cursor: "pointer" }}>Where is data stored? (localStorage keys)</summary>
        <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 6 }}>
          {DATA_STORAGE_MAP.map(row => (
            <div key={row.key} style={{ fontSize: 11.5, color: T.gray600, lineHeight: 1.45 }}>
              <code style={{ fontSize: 10.5, background: T.gray100, padding: "1px 5px", borderRadius: 4 }}>{row.key}</code>
              <strong style={{ color: T.gray700 }}> — {row.label}:</strong> {row.desc}
            </div>
          ))}
          <p style={{ fontSize: 11, color: T.gray500, margin: "6px 0 0" }}>
            View raw data: browser DevTools → Application → Local Storage → your site URL. Export JSON/CSV below before clearing browser data.
          </p>
        </div>
      </details>

      <button
        type="button"
        onClick={() => triggerDownload(`confident-moves-feasibility-${Date.now()}.json`, "application/json", JSON.stringify(report, null, 2))}
        style={{ ...btnBase, background: T.teal, color: "#fff", marginRight: 8 }}
      >
        Export feasibility report JSON
      </button>
      <button type="button" onClick={refresh} style={{ ...btnBase, background: "#fff", color: T.teal, border: `1px solid ${T.teal}` }}>
        Refresh metrics
      </button>
    </div>
  );
}

function HistoryModule({ exportPrefix = "export" }) {
  const [convStore, setConvStore] = useState(() => readConvStore());

  const refreshConvStore = () => setConvStore(readConvStore());

  const handleClearConv = (conversationKey) => {
    clearConversation(conversationKey);
    refreshConvStore();
  };

  const now = new Date();
  const nextA = getNextVisitFromToday(now);
  const nextB = addDays(now, VISIT_OFFSET_DAYS * 2);

  const visits = [
    { date: formatVisitShort(addDays(now, -84)), type: "Enrollment visit", note: "Baseline recorded. BMI 38.1. Consent signed. First injection administered.", status: "completed" },
    { date: formatVisitShort(addDays(now, -63)), type: "Check-in call", note: "Week 2 follow-up. Tolerating medication well. Started dietary log.", status: "completed" },
    { date: formatVisitShort(addDays(now, -42)), type: "Telehealth visit", note: "Week 4 assessment. Weight trending down. Mild nausea reported. Dose maintained.", status: "completed" },
    { date: formatVisitShort(addDays(now, -10)), type: "Telehealth visit", note: "Recent check-in. PA minutes on track. Plan adjustments for the next block.", status: "completed" },
    {
      date: formatVisitShort(nextA),
      type: "Telehealth visit",
      note: `Scheduled ${VISIT_OFFSET_DAYS} days from today — week review with Dr. Patel.`,
      status: "upcoming",
    },
    {
      date: formatVisitShort(nextB),
      type: "In-person visit",
      note: `Follow-up visit ${VISIT_OFFSET_DAYS * 2} days from today — labs and weight assessment.`,
      status: "upcoming",
    },
  ];

  const sideEffects = [
    { date: formatMonthDayShort(addDays(now, -4)), effect: "Mild nausea", severity: "low", time: "2h post-injection" },
    { date: formatMonthDayShort(addDays(now, -18)), effect: "Fatigue", severity: "low", time: "afternoon" },
    { date: formatMonthDayShort(addDays(now, -32)), effect: "Nausea", severity: "medium", time: "morning" },
    { date: formatMonthDayShort(addDays(now, -50)), effect: "Injection site redness", severity: "low", time: "day 1" },
  ];

  const severityColor = { low: T.teal, medium: T.amber, high: T.red };
  const severityBg = { low: T.tealLight, medium: T.amberLight, high: T.redLight };

  const exportLog = readInstrumentLog();
  const convEntries = Object.values(convStore);
  const totalConvMsgs = convEntries.reduce((acc, e) => acc + e.messages.length, 0);

  const btnBase = {
    border: "none", fontFamily: "'DM Sans', sans-serif",
    fontWeight: 600, fontSize: 12.5, padding: "8px 14px", borderRadius: 8, transition: "opacity .15s",
  };

  const dlInstJson = () => {
    const log = readInstrumentLog();
    if (!log.length) return;
    triggerDownload(`confident-moves-instruments-${exportPrefix}-${Date.now()}.json`, "application/json", JSON.stringify(log, null, 2));
  };
  const dlInstCsv = () => {
    const log = readInstrumentLog();
    if (!log.length) return;
    triggerDownload(`confident-moves-instruments-${exportPrefix}-${Date.now()}.csv`, "text/csv;charset=utf-8", instrumentLogToCsv(log));
  };
  const dlConvJson = () => {
    const store = readConvStore();
    if (!Object.keys(store).length) return;
    triggerDownload(`confident-moves-conversations-${exportPrefix}-${Date.now()}.json`, "application/json", JSON.stringify(store, null, 2));
  };
  const dlConvCsv = () => {
    const store = readConvStore();
    if (!Object.keys(store).length) return;
    triggerDownload(`confident-moves-conversations-${exportPrefix}-${Date.now()}.csv`, "text/csv;charset=utf-8", convStoreToCsv(store));
  };
  const dlPaseScoresCsv = () => {
    const csv = paseScoresSummaryCsv();
    if (!csv.split("\r\n").length > 1) return;
    triggerDownload(`confident-moves-pase-scores-${Date.now()}.csv`, "text/csv;charset=utf-8", csv);
  };
  // LLM-as-judge export disabled — restore with judge block
  // const dlBenchmarkCsv = () => {
  //   const csv = benchmarkCsv();
  //   triggerDownload(`confident-moves-benchmark-${Date.now()}.csv`, "text/csv;charset=utf-8", csv);
  // };

  return (
    <div style={{ padding: 20, overflowY: "auto", height: "100%" }}>

      <FeasibilityPanel />

      {/* <JudgePanel /> — LLM-as-judge disabled; restore when benchmarking resumes */}

      {/* ── Conversation Logs ── */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: T.gray500, textTransform: "uppercase", letterSpacing: ".05em", marginBottom: 4 }}>
          Conversation logs
        </div>
        <p style={{ fontSize: 12.5, color: T.gray600, lineHeight: 1.55, marginBottom: 12 }}>
          Every message is stored per profile × condition × module ({convEntries.length} log{convEntries.length !== 1 ? "s" : ""}, {totalConvMsgs} turns). Expand to compare how the AI responds across profiles and conditions A/B/C.
        </p>

        {convEntries.length === 0 && (
          <div style={{ fontSize: 12.5, color: T.gray400, fontStyle: "italic", padding: "12px 0" }}>
            No conversations stored yet — start chatting in any tab and the logs will appear here.
          </div>
        )}

        {convEntries
          .sort((a, b) => (b.lastUpdated ?? "").localeCompare(a.lastUpdated ?? ""))
          .map(entry => (
          <ConversationCard key={entry.conversationKey ?? entry.moduleId} entry={entry} onClear={() => handleClearConv(entry.conversationKey ?? entry.moduleId)} />
        ))}

        <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 10 }}>
          <button type="button" disabled={convEntries.length === 0} onClick={dlConvJson} style={{
            ...btnBase, border: "none", background: T.teal, color: "#fff",
            cursor: convEntries.length ? "pointer" : "not-allowed", opacity: convEntries.length ? 1 : 0.45,
          }}>Export conversations JSON</button>
          <button type="button" disabled={convEntries.length === 0} onClick={dlConvCsv} style={{
            ...btnBase, background: "#fff", color: T.tealDark, border: `1px solid ${T.teal}`,
            cursor: convEntries.length ? "pointer" : "not-allowed", opacity: convEntries.length ? 1 : 0.45,
          }}>Export conversations CSV (de-identified)</button>
        </div>
      </div>

      {/* ── Research-ready data exports ── */}
      <div style={{ marginBottom: 24, padding: 16, background: "#fff", border: `1px solid ${T.teal}`, borderRadius: 10 }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: T.tealDark, textTransform: "uppercase", letterSpacing: ".05em", marginBottom: 6 }}>
          Research exports (Excel-ready)
        </div>
        <p style={{ fontSize: 12.5, color: T.gray600, lineHeight: 1.6, marginBottom: 14 }}>
          These CSVs are designed for non-technical staff. Open in Excel or Google Sheets directly. All files use <strong>participant code</strong> (PT-0001) — no names are exported.
        </p>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
          <div>
            <button type="button" onClick={dlPaseScoresCsv} style={{
              ...btnBase, background: T.teal, color: "#fff", border: "none", cursor: "pointer",
            }}>PASE / SERPA scores summary (CSV)</button>
            <p style={{ fontSize: 11, color: T.gray500, margin: "4px 0 0 0" }}>
              One row per participant. Each instrument score and plain-English level in its own column.
            </p>
          </div>
          {/* LLM-as-judge benchmark export — disabled
          <div>
            <button type="button" onClick={dlBenchmarkCsv} style={{
              ...btnBase, background: T.purple, color: "#fff", border: "none", cursor: "pointer",
            }}>Benchmark / LLM-judge scores (CSV)</button>
            <p style={{ fontSize: 11, color: T.gray500, margin: "4px 0 0 0" }}>
              One row per scored AI response. Includes blank <em>human_*</em> columns for your RA to fill in — compare LLM vs human ratings.
            </p>
          </div>
          */}
        </div>
        {GOOGLE_SHEETS_WEBAPP_URL && (
          <p style={{ fontSize: 11.5, color: T.teal, marginTop: 12, lineHeight: 1.5 }}>
            Google Sheets sync is active — conversation turns are pushed automatically on every message send.
          </p>
        )}
        {!GOOGLE_SHEETS_WEBAPP_URL && (
          <p style={{ fontSize: 11.5, color: T.gray400, marginTop: 12, lineHeight: 1.5 }}>
            Add <code>VITE_GOOGLE_SHEETS_WEBAPP_URL</code> to your <code>.env</code> to enable automatic Google Sheets sync.
          </p>
        )}
      </div>

      {/* ── Assessment data export ── */}
      <div style={{
        marginBottom: 24, padding: 16, background: "#fff", border: `1px solid ${T.gray200}`, borderRadius: 10,
      }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: T.gray500, textTransform: "uppercase", letterSpacing: ".05em", marginBottom: 8 }}>
          Assessment data export
        </div>
        <p style={{ fontSize: 12.5, color: T.gray600, lineHeight: 1.55, marginBottom: 14 }}>
          Legacy instrument JSON exports from earlier sessions ({exportLog.length} record{exportLog.length !== 1 ? "s" : ""}), if any were saved with an <code style={{ fontSize: 11 }}>[INSTRUMENT_DATA: …]</code> tag.
        </p>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
          <button type="button" disabled={exportLog.length === 0} onClick={dlInstJson} style={{
            ...btnBase, border: "none", background: T.teal, color: "#fff", cursor: exportLog.length ? "pointer" : "not-allowed", opacity: exportLog.length ? 1 : 0.45,
          }}>Download JSON</button>
          <button type="button" disabled={exportLog.length === 0} onClick={dlInstCsv} style={{
            ...btnBase, background: "#fff", color: T.tealDark, border: `1px solid ${T.teal}`, cursor: exportLog.length ? "pointer" : "not-allowed", opacity: exportLog.length ? 1 : 0.45,
          }}>Download CSV</button>
        </div>
      </div>

      {/* ── Visit history ── */}
      <div style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: T.gray500, textTransform: "uppercase", letterSpacing: ".05em", marginBottom: 12 }}>Visit history</div>
        {visits.map((v, i) => (
          <div key={i} style={{
            display: "flex", gap: 12, marginBottom: 12,
            background: v.status === "upcoming" ? T.purpleLight : "#fff",
            border: `1px solid ${v.status === "upcoming" ? T.gray300 : T.gray200}`,
            borderRadius: 10, padding: "12px 14px"
          }}>
            <div style={{
              width: 8, height: 8, borderRadius: "50%", marginTop: 5, flexShrink: 0,
              background: v.status === "upcoming" ? T.purple : T.teal
            }} />
            <div>
              <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 4 }}>
                <span style={{ fontSize: 13, fontWeight: 600, color: T.gray800 }}>{v.type}</span>
                <Badge
                  children={v.status === "upcoming" ? "Upcoming" : "Completed"}
                  color={v.status === "upcoming" ? T.purple : T.teal}
                  bg={v.status === "upcoming" ? T.purpleLight : T.tealLight}
                />
              </div>
              <div style={{ fontSize: 11, color: T.gray500, marginBottom: 4 }}>{v.date}</div>
              <div style={{ fontSize: 12.5, color: T.gray600, lineHeight: 1.5 }}>{v.note}</div>
            </div>
          </div>
        ))}
      </div>

      {/* ── Side effect log ── */}
      <div>
        <div style={{ fontSize: 13, fontWeight: 600, color: T.gray500, textTransform: "uppercase", letterSpacing: ".05em", marginBottom: 12 }}>Side effect log</div>
        {sideEffects.map((s, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, padding: "9px 14px", background: "#fff", border: `1px solid ${T.gray200}`, borderRadius: 8, marginBottom: 6 }}>
            <div style={{ fontSize: 11, color: T.gray400, width: 42 }}>{s.date}</div>
            <div style={{ flex: 1, fontSize: 13, color: T.gray700 }}>{s.effect}</div>
            <div style={{ fontSize: 11, color: T.gray400 }}>{s.time}</div>
            <Badge children={s.severity} color={severityColor[s.severity]} bg={severityBg[s.severity]} />
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────
const TABS = [
  { id: "chat", label: "PA Coach", icon: "🏃" },
  { id: "checkin", label: "Daily Check-in", icon: "✅" },
  { id: "eligibility", label: "Program Eligibility", icon: "🔬" },
  { id: "education", label: "Learn & Explore", icon: "📚" },
  { id: "history", label: "My Progress", icon: "📈" },
];

function loadActiveProfileId() {
  try {
    const id = localStorage.getItem(ACTIVE_PROFILE_KEY);
    if (id && DEFAULT_PROFILES.some(p => p.id === id)) return id;
  } catch {}
  return DEFAULT_PROFILES[0].id;
}

function loadActiveCondition() {
  try {
    const c = localStorage.getItem(ACTIVE_CONDITION_KEY);
    if (c && CONDITIONS.some(x => x.id === c)) return c;
  } catch {}
  return "A";
}

function applyProgramStateToReact(setters, state) {
  setters.setProgramDay(state.programDay);
  setters.setCurrentWeight(state.currentWeight);
  setters.setWeekPaMins(state.weekPaMins);
  setters.setActiveDaysThisWeek(state.activeDaysThisWeek);
  setters.setLastPaLogProgramDay(state.lastPaLogProgramDay ?? null);
  setters.setLastActiveMarkProgramDay(state.lastActiveMarkProgramDay ?? null);
  setters.setDraftWeight(String(state.currentWeight));
  setters.setDraftTodayPa("");
}

export default function App() {
  migrateLegacyProgramState(DEFAULT_PROFILES[0].id);
  seedMockScores();

  const [profiles, setProfiles] = useState(() => readProfiles());
  const [activeProfileId, setActiveProfileId] = useState(loadActiveProfileId);
  const [activeCondition, setActiveCondition] = useState(loadActiveCondition);
  const [editingProfile, setEditingProfile] = useState(false);
  const [profileDraft, setProfileDraft] = useState(null);

  const patient = useMemo(
    () => profiles.find(p => p.id === activeProfileId) ?? profiles[0] ?? DEFAULT_PROFILES[0],
    [profiles, activeProfileId],
  );

  const conditionMeta = useMemo(
    () => CONDITIONS.find(c => c.id === activeCondition) ?? CONDITIONS[0],
    [activeCondition],
  );

  const totalProgramDays = getTotalProgramDays(patient);
  const saved = loadProgramState(patient.id, patient);
  const initial = saved ?? defaultProgramState(patient);

  const [activeTab, setActiveTab] = useState(() => {
    const saved = localStorage.getItem("confidentMoves_active_tab");
    if (saved && TABS.some(t => t.id === saved)) return saved;
    return "chat";
  });

  useEffect(() => {
    if (!TABS.some(t => t.id === activeTab)) setActiveTab("chat");
    else {
      try { localStorage.setItem("confidentMoves_active_tab", activeTab); } catch {}
    }
  }, [activeTab]);
  const [programDay, setProgramDay] = useState(initial.programDay);
  const [currentWeight, setCurrentWeight] = useState(initial.currentWeight);
  const [weekPaMins, setWeekPaMins] = useState(initial.weekPaMins);
  const [activeDaysThisWeek, setActiveDaysThisWeek] = useState(initial.activeDaysThisWeek);
  const [lastPaLogProgramDay, setLastPaLogProgramDay] = useState(initial.lastPaLogProgramDay ?? null);
  const [lastActiveMarkProgramDay, setLastActiveMarkProgramDay] = useState(initial.lastActiveMarkProgramDay ?? null);

  const [draftWeight, setDraftWeight] = useState(String(initial.currentWeight));
  const [draftTodayPa, setDraftTodayPa] = useState("");

  const [todayLabel, setTodayLabel] = useState(() => formatTodayLong());
  const [nextVisitLabel, setNextVisitLabel] = useState(() => formatVisitShort(getNextVisitFromToday()));

  useEffect(() => {
    const refresh = () => {
      const now = new Date();
      setTodayLabel(formatTodayLong(now));
      setNextVisitLabel(formatVisitShort(getNextVisitFromToday(now)));
    };
    refresh();
    const id = setInterval(refresh, 60_000);
    return () => clearInterval(id);
  }, []);

  const programWeek = Math.min(Math.ceil(programDay / 7), patient.totalWeeks);
  const prevProgramWeekRef = useRef(programWeek);

  useEffect(() => {
    if (editingProfile) setProfileDraft(cloneProfiles([patient])[0]);
  }, [editingProfile, activeProfileId]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    try {
      localStorage.setItem(ACTIVE_PROFILE_KEY, activeProfileId);
      localStorage.setItem(ACTIVE_CONDITION_KEY, activeCondition);
    } catch {}
  }, [activeProfileId, activeCondition]);

  useEffect(() => {
    try {
      localStorage.setItem(
        programStateKey(activeProfileId),
        JSON.stringify({
          programDay,
          currentWeight,
          weekPaMins,
          activeDaysThisWeek,
          lastPaLogProgramDay,
          lastActiveMarkProgramDay,
        }),
      );
    } catch (e) {
      console.warn("Program state save failed:", e);
    }
  }, [activeProfileId, programDay, currentWeight, weekPaMins, activeDaysThisWeek, lastPaLogProgramDay, lastActiveMarkProgramDay]);

  useEffect(() => {
    const prev = prevProgramWeekRef.current;
    if (programWeek > prev) {
      setWeekPaMins(0);
      setActiveDaysThisWeek(0);
      setLastPaLogProgramDay(null);
      setLastActiveMarkProgramDay(null);
    }
    prevProgramWeekRef.current = programWeek;
  }, [programWeek]);

  useEffect(() => {
    setDraftWeight(String(currentWeight));
  }, [currentWeight]);

  const runtime = useMemo(() => {
    const w = Number(currentWeight);
    const weightOk = Number.isFinite(w) && w > 0 ? w : patient.weight.current;
    return {
      programDay,
      programWeek,
      currentWeight: weightOk,
      estBmi: estBmiFromWeight(weightOk, patient),
      weeklyPaMins: weekPaMins,
      weeklyPaGoal: patient.pa.weeklyGoalMins,
      activeDaysThisWeek,
      activeDaysGoal: patient.pa.goalDays,
    };
  }, [programDay, programWeek, currentWeight, weekPaMins, activeDaysThisWeek, patient]);

  const systems = useMemo(
    () => applyConditionToSystems(buildSystems(runtime, patient, activeCondition), activeCondition),
    [runtime, patient, activeCondition],
  );

  const moduleProps = useMemo(() => ({
    systems,
    patient,
    programDay,
    programWeek,
    conditionId: activeCondition,
    conditionLabel: conditionMeta.label,
  }), [systems, patient, programDay, programWeek, activeCondition, conditionMeta.label]);

  const weightLoss = Math.max(0, patient.weight.baseline - currentWeight);
  const toGoal = patient.weight.baseline - patient.weight.goal;
  const weightGoalPct = toGoal > 0 ? Math.min(100, Math.round((weightLoss / toGoal) * 100)) : 0;
  const trialPct = Math.min(100, Math.round((programDay / totalProgramDays) * 100));
  const paPct = Math.min(100, Math.round((weekPaMins / patient.pa.weeklyGoalMins) * 100));
  const activeDaysPct = Math.min(100, Math.round((activeDaysThisWeek / patient.pa.goalDays) * 100));

  const loadProfileProgramState = useCallback((profile) => {
    const st = loadProgramState(profile.id, profile) ?? defaultProgramState(profile);
    applyProgramStateToReact(
      { setProgramDay, setCurrentWeight, setWeekPaMins, setActiveDaysThisWeek, setLastPaLogProgramDay, setLastActiveMarkProgramDay, setDraftWeight, setDraftTodayPa },
      st,
    );
    prevProgramWeekRef.current = Math.min(Math.ceil(st.programDay / 7), profile.totalWeeks);
  }, []);

  const handleSelectProfile = (profileId) => {
    if (profileId === activeProfileId) return;
    const next = profiles.find(p => p.id === profileId);
    if (!next) return;
    if (!window.confirm(`Switch to ${next.name}? This profile's saved program day and chats are kept separately.`)) return;
    setActiveProfileId(profileId);
    loadProfileProgramState(next);
    setEditingProfile(false);
  };

  const handleSelectCondition = (conditionId) => {
    setActiveCondition(conditionId);
  };

  const handleSaveProfile = () => {
    if (!profileDraft) return;
    const updated = profiles.map(p => (p.id === profileDraft.id ? profileDraft : p));
    setProfiles(updated);
    writeProfiles(updated);
    if (profileDraft.id === activeProfileId) {
      const st = loadProgramState(profileDraft.id, profileDraft) ?? defaultProgramState(profileDraft);
      applyProgramStateToReact(
        { setProgramDay, setCurrentWeight, setWeekPaMins, setActiveDaysThisWeek, setLastPaLogProgramDay, setLastActiveMarkProgramDay, setDraftWeight, setDraftTodayPa },
        st,
      );
    }
    setEditingProfile(false);
    setProfileDraft(null);
  };

  const handleResetProfile = () => {
    const def = DEFAULT_PROFILES.find(p => p.id === activeProfileId);
    if (!def || !window.confirm(`Reset ${def.name} to factory defaults?`)) return;
    const updated = profiles.map(p => (p.id === def.id ? cloneProfiles([def])[0] : p));
    setProfiles(updated);
    writeProfiles(updated);
    setProfileDraft(cloneProfiles([def])[0]);
    if (activeProfileId === def.id) loadProfileProgramState(def);
  };

  const handleResetAllProfiles = () => {
    if (!window.confirm("Reset all 3 profiles to factory defaults?")) return;
    const fresh = cloneProfiles(DEFAULT_PROFILES);
    setProfiles(fresh);
    writeProfiles(fresh);
    setProfileDraft(cloneProfiles([fresh.find(p => p.id === activeProfileId) ?? fresh[0]])[0]);
    loadProfileProgramState(fresh.find(p => p.id === activeProfileId) ?? fresh[0]);
  };

  const applySelfReport = () => {
    const w = Number(draftWeight);
    if (Number.isFinite(w) && w > 0) setCurrentWeight(w);
    const add = Number(draftTodayPa);
    if (Number.isFinite(add) && add > 0 && lastPaLogProgramDay !== programDay) {
      setWeekPaMins((x) => x + add);
      setLastPaLogProgramDay(programDay);
    }
    setDraftTodayPa("");
  };

  const markActiveToday = () => {
    if (lastActiveMarkProgramDay === programDay) return;
    setActiveDaysThisWeek((x) => Math.min(x + 1, 7));
    setLastActiveMarkProgramDay(programDay);
  };

  const goNextProgramDay = () => {
    setProgramDay((d) => Math.min(d + 1, totalProgramDays));
  };

  const resetProgramStart = () => {
    const d = defaultProgramState(patient);
    applyProgramStateToReact(
      { setProgramDay, setCurrentWeight, setWeekPaMins, setActiveDaysThisWeek, setLastPaLogProgramDay, setLastActiveMarkProgramDay, setDraftWeight, setDraftTodayPa },
      d,
    );
    prevProgramWeekRef.current = Math.min(Math.ceil(d.programDay / 7), patient.totalWeeks);
  };

  const initials = patientInitials(patient.name);

  return (
    <div style={{ fontFamily: "'DM Sans', 'Helvetica Neue', sans-serif", background: T.gray50, flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        @keyframes bounce { 0%,80%,100%{transform:translateY(0)} 40%{transform:translateY(-5px)} }
        ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-thumb { background: ${T.gray300}; border-radius: 2px; }
        textarea { font-family: 'DM Sans', sans-serif !important; }
      `}</style>

      {/* Top bar */}
      <div style={{ background: "#fff", borderBottom: `1px solid ${T.gray200}`, padding: "0 24px", display: "flex", alignItems: "center", gap: 16, minHeight: 56, flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 32, height: 32, borderRadius: 8, background: T.teal, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="14" cy="3" r="1.5"/>
              <path d="M17 7l-3 4-4-1-3 6"/>
              <path d="M7 16l-2 4"/>
              <path d="M14 11l2 5"/>
            </svg>
          </div>
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: T.gray900, lineHeight: 1 }}>Confident Moves</div>
            <div style={{ fontSize: 11, color: T.gray400, lineHeight: 1.4 }}>Personalized PA support</div>
          </div>
        </div>

        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <Badge children={`${patient.trial}`} color={T.purple} bg={T.purpleLight} />
          <Badge children={`Condition ${activeCondition}`} color={T.purple} bg={T.purpleLight} />
          <Badge children={`Day ${programDay} / ${totalProgramDays}`} color={T.tealDark} bg={T.tealLight} />
          <Badge children={`Week ${programWeek}/${patient.totalWeeks}`} color={T.teal} bg={T.tealLight} />
          <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "5px 12px", border: `1px solid ${T.gray200}`, borderRadius: 20, background: T.gray50 }}>
            <Avatar initials={initials} size={22} />
            <span style={{ fontSize: 12, fontWeight: 500, color: T.gray700 }}>{patient.name}</span>
          </div>
        </div>
      </div>

      <div style={{ flex: 1, display: "flex", minHeight: 0, overflow: "hidden" }}>
        {/* Sidebar — wider + denser layout so tabs & goals usually fit without scrolling */}
        <div style={{
          width: "clamp(300px, 28vw, 400px)",
          minWidth: 300,
          background: "#fff",
          borderRight: `1px solid ${T.gray200}`,
          flexShrink: 0,
          minHeight: 0,
          alignSelf: "stretch",
          overflowY: "auto",
          overscrollBehavior: "contain",
          WebkitOverflowScrolling: "touch",
        }}>
          <div style={{ padding: "11px 14px 8px" }}>
            <div style={{ fontSize: 12.5, fontWeight: 600, color: T.gray500, textTransform: "uppercase", letterSpacing: ".06em" }}>My Journey</div>
          </div>

          <ResearchSidebarPanel
            profiles={profiles}
            activeProfileId={activeProfileId}
            activeCondition={activeCondition}
            editingProfile={editingProfile}
            profileDraft={profileDraft}
            onSelectProfile={handleSelectProfile}
            onSelectCondition={handleSelectCondition}
            onToggleEdit={() => setEditingProfile(x => !x)}
            onDraftChange={setProfileDraft}
            onSaveProfile={handleSaveProfile}
            onResetProfile={handleResetProfile}
            onResetAllProfiles={handleResetAllProfiles}
          />

          <div style={{
            padding: "9px 11px 11px",
            margin: "0 10px 8px",
            borderRadius: 10,
            background: T.gray50,
            border: `1px solid ${T.gray200}`,
          }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: T.gray500, textTransform: "uppercase", letterSpacing: ".06em", marginBottom: 7 }}>Self-report · day {programDay}</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 8 }}>
              <div>
                <label style={{ display: "block", fontSize: 11.5, color: T.gray600, marginBottom: 3 }}>Weight (lb)</label>
                <input
                  value={draftWeight}
                  onChange={e => setDraftWeight(e.target.value)}
                  inputMode="decimal"
                  style={{
                    width: "100%", padding: "7px 9px", borderRadius: 8, border: `1px solid ${T.gray300}`,
                    fontSize: 13.5, fontFamily: "'DM Sans', sans-serif", background: "#fff",
                  }}
                />
              </div>
              <div>
                <label style={{ display: "block", fontSize: 11.5, color: T.gray600, marginBottom: 3 }}>
                  PA (min){lastPaLogProgramDay === programDay ? " ✓" : ""}
                </label>
                <input
                  value={draftTodayPa}
                  onChange={e => setDraftTodayPa(e.target.value)}
                  inputMode="numeric"
                  placeholder={lastPaLogProgramDay === programDay ? "—" : "30"}
                  disabled={lastPaLogProgramDay === programDay}
                  style={{
                    width: "100%", padding: "7px 9px", borderRadius: 8, border: `1px solid ${T.gray300}`,
                    fontSize: 13.5, fontFamily: "'DM Sans', sans-serif", background: "#fff",
                    opacity: lastPaLogProgramDay === programDay ? 0.65 : 1,
                  }}
                />
              </div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, marginBottom: 6 }}>
              <button type="button" onClick={applySelfReport} style={{
                fontFamily: "'DM Sans', sans-serif", fontWeight: 600, fontSize: 12.5,
                padding: "8px 9px", borderRadius: 8, border: "none", background: T.teal, color: "#fff", cursor: "pointer",
              }}>Save</button>
              <button type="button" onClick={markActiveToday} disabled={lastActiveMarkProgramDay === programDay} style={{
                fontFamily: "'DM Sans', sans-serif", fontWeight: 600, fontSize: 12.5,
                padding: "8px 9px", borderRadius: 8, border: `1px solid ${T.teal}`, background: "#fff", color: T.tealDark,
                cursor: lastActiveMarkProgramDay === programDay ? "not-allowed" : "pointer", opacity: lastActiveMarkProgramDay === programDay ? 0.55 : 1,
              }}>{lastActiveMarkProgramDay === programDay ? "Active ✓" : "+ Active day"}</button>
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              <button type="button" onClick={goNextProgramDay} disabled={programDay >= totalProgramDays} style={{
                flex: 1, fontFamily: "'DM Sans', sans-serif", fontWeight: 600, fontSize: 11.5,
                padding: "7px 7px", borderRadius: 8, border: `1px solid ${T.gray300}`, background: "#fff", color: T.gray800,
                cursor: programDay >= totalProgramDays ? "not-allowed" : "pointer",
              }}>Next day</button>
              <button type="button" onClick={resetProgramStart} style={{
                flex: 1, fontFamily: "'DM Sans', sans-serif", fontWeight: 600, fontSize: 11.5,
                padding: "7px 7px", borderRadius: 8, border: `1px solid ${T.amber}`, background: T.amberLight, color: T.amber,
                cursor: "pointer",
              }}>Reset</button>
            </div>
          </div>

          <div style={{
            padding: "7px 10px 10px",
            borderTop: `1px solid ${T.gray100}`,
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 6,
          }}>
            {TABS.map(tab => (
              <button key={tab.id} type="button" onClick={() => setActiveTab(tab.id)} style={{
                display: "flex", alignItems: "center", gap: 7, padding: "9px 11px",
                borderRadius: 8,
                background: activeTab === tab.id ? T.tealLight : T.gray50,
                border: activeTab === tab.id ? `1px solid ${T.tealMid}` : `1px solid ${T.gray200}`,
                boxShadow: activeTab === tab.id ? `inset 0 0 0 1px ${T.tealLight}` : "none",
                cursor: "pointer", textAlign: "left",
                transition: "background .15s, border-color .15s",
              }}>
                <span style={{ fontSize: 16, lineHeight: 1 }}>{tab.icon}</span>
                <span style={{
                  fontSize: 13, fontWeight: activeTab === tab.id ? 600 : 500,
                  color: activeTab === tab.id ? T.tealDark : T.gray600, lineHeight: 1.25,
                }}>{tab.label}</span>
              </button>
            ))}
          </div>

          <div style={{ padding: "13px 11px", borderTop: `1px solid ${T.gray200}`, background: "#fff" }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: T.gray500, textTransform: "uppercase", letterSpacing: ".06em", marginBottom: 8 }}>My Goals</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              <MetricCard label="Weekly PA" value={`${weekPaMins}/${patient.pa.weeklyGoalMins}′`} sub={`${paPct}%`} progress={paPct} compact sidebar />
              <MetricCard label="Active days" value={`${activeDaysThisWeek}/${patient.pa.goalDays}`} sub="week" progress={activeDaysPct} compact sidebar />
              <MetricCard
                label="Weight lost"
                value={`−${weightLoss} lbs`}
                sub={`${weightGoalPct}% · BMI ${estBmiFromWeight(currentWeight, patient)}`}
                progress={weightGoalPct}
                color={T.amber}
                compact
                sidebar
              />
              <MetricCard label="Program" value={`${programDay}d`} sub={`${trialPct}% done`} progress={trialPct} color={T.purple} compact sidebar />
            </div>
            <div style={{ fontSize: 11.5, color: T.gray500, marginTop: 10, paddingTop: 9, borderTop: `1px solid ${T.gray100}` }}>
              <div style={{ marginBottom: 5 }}>
                <span style={{ fontWeight: 600 }}>Today: </span>
                <span style={{ color: T.gray700, fontWeight: 500 }}>{todayLabel}</span>
              </div>
              <div style={{ marginBottom: 5 }}>
                Next visit: <span style={{ color: T.gray700, fontWeight: 500 }}>{nextVisitLabel}</span>
                <span style={{ fontSize: 10, color: T.gray500, display: "block", marginTop: 2 }}>
                  ({VISIT_OFFSET_DAYS} days from today)
                </span>
              </div>
              <div style={{ lineHeight: 1.4, fontSize: 12, color: T.gray600 }}>{patient.doctor}</div>
            </div>
          </div>
        </div>

        {/* Main panel */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0, overflow: "hidden" }}>
          {/* Module header */}
          <div style={{ background: "#fff", borderBottom: `1px solid ${T.gray200}`, padding: "12px 24px", display: "flex", alignItems: "center", gap: 12, flexShrink: 0 }}>
            <span style={{ fontSize: 18 }}>{TABS.find(t => t.id === activeTab)?.icon}</span>
            <div>
              <div style={{ fontSize: 15, fontWeight: 600, color: T.gray800 }}>{TABS.find(t => t.id === activeTab)?.label}</div>
              <div style={{ fontSize: 12, color: T.gray400 }}>
                {/* PRODUCTION: replace "Groq AI" with dynamic provider label from LLM_MODEL (e.g. "Claude Haiku") */}
                {activeTab === "chat" && "Real-time PA coaching, goal setting & whole-person support — Groq AI"}
                {activeTab === "checkin" && "Daily wellness, activity & medication check-in"}
                {activeTab === "eligibility" && "Automated program eligibility screening — STEP-OB-24"}
                {activeTab === "education" && "Evidence-based PA education & obesity medicine"}
                {activeTab === "history" && "Conversation logs, visit records & health notes"}
              </div>
            </div>
            <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: T.tealMid, marginTop: 4 }} />
              <span style={{ fontSize: 12, color: T.gray500 }}>AI coaching active</span>
            </div>
          </div>

          {/* Module content */}
          <div style={{ flex: 1, minHeight: 0, overflow: "hidden" }}>
            {activeTab === "chat" && <ChatModule {...moduleProps} />}
            {activeTab === "checkin" && <CheckInModule {...moduleProps} />}
            {activeTab === "eligibility" && <EligibilityModule {...moduleProps} />}
            {activeTab === "education" && <EducationModule {...moduleProps} />}
            {activeTab === "history" && <HistoryModule exportPrefix="all-profiles" />}
          </div>
        </div>
      </div>
    </div>
  );
}
