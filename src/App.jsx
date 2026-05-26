import { useState, useRef, useEffect, useCallback, useMemo } from "react";
 
const GROQ_API_KEY = import.meta.env.VITE_GROQ_API_SECRET_KEY;
if (!GROQ_API_KEY) {
  throw new Error("VITE_GROQ_API_SECRET_KEY is not set in .env");
}

async function callGroq(messages, systemPrompt, onChunk) {
  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${GROQ_API_KEY}`,
    },
    body: JSON.stringify({
      model: "llama-3.3-70b-versatile",
      max_tokens: 1000,
      temperature: 0.7,
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

function instrumentLogToCsv(log) {
  const headers = ["submittedAt", "participantId", "participantName", "instrumentKey", "instrumentLabel", "responsesJson"];
  const lines = [headers.join(",")];
  for (const row of log) {
    const responsesJson = JSON.stringify(row.responses ?? {});
    lines.push([
      csvEscapeCell(row.submittedAt),
      csvEscapeCell(row.participantId),
      csvEscapeCell(row.participantName),
      csvEscapeCell(row.instrumentKey),
      csvEscapeCell(row.instrumentLabel),
      csvEscapeCell(responsesJson),
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

async function persistInstrumentSubmission({ instrumentKey, instrumentLabel, responses }) {
  const record = {
    submittedAt: new Date().toISOString(),
    participantId: PATIENT.id,
    participantName: PATIENT.name,
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

// ─── Patient data (baselines; program day/week + self-report from state) ────
const PATIENT = {
  name: "Son Vu",
  id: "PT-0001",
  trial: "Pilot Trial",
  drug: "Semaglutide 2.4mg",
  totalWeeks: 24,
  bmi: { current: 34.2, baseline: 38.1 },
  weight: { current: 198, baseline: 207, unit: "lbs", goal: 177 },
  adherence: 94,
  doctor: "Confident Moves Obesity Care Team",
  conditions: ["Type 2 diabetes", "Hypertension"],
  medications: ["Metformin 500mg", "Lisinopril 10mg"],
  pa: {
    weeklyGoalMins: 150,
    goalDays: 5,
    topBarrier: "time constraints",
    favoriteActivity: "walking",
  },
};

const TOTAL_PROGRAM_DAYS = PATIENT.totalWeeks * 7;
const PROGRAM_STATE_KEY = "confidentMoves_program_v1";

const defaultProgramState = () => ({
  programDay: 1,
  currentWeight: PATIENT.weight.current,
  weekPaMins: 0,
  activeDaysThisWeek: 0,
  lastPaLogProgramDay: null,
  lastActiveMarkProgramDay: null,
});

function loadProgramState() {
  try {
    const raw = localStorage.getItem(PROGRAM_STATE_KEY);
    if (!raw) return null;
    const d = JSON.parse(raw);
    const base = defaultProgramState();
    if (typeof d.programDay !== "number" || d.programDay < 1) return null;
    return { ...base, ...d, programDay: d.programDay };
  } catch {
    return null;
  }
}

function estBmiFromWeight(weightLbs) {
  const w = Number(weightLbs);
  if (!Number.isFinite(w) || w <= 0) return PATIENT.bmi.baseline.toFixed(1);
  return ((PATIENT.bmi.baseline * w) / PATIENT.weight.baseline).toFixed(1);
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
function buildSystems(rt) {
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

  return {
  chat: `You are ObesityCare AI, a clinical support assistant embedded in an obesity management trial platform. Your coaching is grounded in social cognitive theory — specifically Bandura's self-efficacy — and in common elements of evidence-based behavior change techniques (BCTs) used in lifestyle trials (e.g., goal setting, action planning, problem solving, self-monitoring of behavior).

Patient: ${PATIENT.name}, Trial: ${PATIENT.trial}, Drug: ${PATIENT.drug}.
Program timeline: Day ${programDay} of ${TOTAL_PROGRAM_DAYS} (week ${programWeek} of ${PATIENT.totalWeeks}).
Self-report: weight ${currentWeight} lbs; estimated BMI ~${estBmi} (baseline BMI ${PATIENT.bmi.baseline}); this week's PA total ${weeklyPaMins} / ${weeklyPaGoal} min; active days with movement this week: ${activeDaysThisWeek} / ${activeDaysGoal}.
Comorbidities: ${PATIENT.conditions.join(", ")}.

THEORY-DRIVEN COMMUNICATION (self-efficacy)
Self-efficacy is the person's confidence that they can perform a behavior in a given context. In practice, support mastery, credible encouragement, context, and emotional safety — not generic praise.
1) Mastery experiences: Elicit past successes — even small (e.g., "What helped the last time you fit in movement?"). Tie next steps to those successes.
2) Vicarious experience: When fitting, normalize with cohort-appropriate language (trials like this; many people with busy schedules) — never compare one patient invidiously to another.
3) Verbal/social persuasion: Use authentic, specific encouragement tied to their own data or stated intent — avoid empty reassurance or pressure.
4) Physiological/affective states: Acknowledge fatigue, stress, side effects, mood. Reframe discomfort as information for a smaller or adjusted plan, not as failure. Escalate severe symptoms.

EVIDENCE-ALIGNED STRATEGIES (use when relevant; do not stack all in one reply)
- Collaborative goal setting: Patient-chosen priority; ask what feels "doable this week" before suggesting specifics.
- Action planning: When/where/how long; break into steps the patient agrees to (implementation intentions: "If [situation], then I will [micro-action]").
- Confidence / importance: Brief 0–10 check ("How confident are you that you can do that plan?"); if confidence is low, shrink the step until confidence rises.
- Problem solving: Identify barrier → brainstorm one or two options → patient picks; avoid solving for them.
- Motivational interviewing style: Open questions, affirm effort, reflect, summarize; no lecturing.

OPERATING RULES
- Never diagnose, prescribe, or change medication instructions. Defer medical decisions to the care team.
- Keep replies concise (2–5 sentences) unless the user asks for detail.
- Flag severe side effects, distress, self-harm, or safety concerns with [ESCALATE].
- You may discuss GLP-1 therapy, obesity, nutrition, PA, and the trial in general educational terms.`,

  checkin: `You are ObesityCare Confident Moves AI conducting a structured daily check-in for a clinical trial participant. Use brief, collaborative language consistent with motivational interviewing and self-efficacy support (acknowledge effort; ask one thing at a time; no judgment).
Patient: ${PATIENT.name}, program day ${programDay} (week ${programWeek}), Drug: ${PATIENT.drug}.
Self-reported weight ${currentWeight} lbs; weekly PA minutes so far ${weeklyPaMins} / ${weeklyPaGoal}.
Conduct a brief, empathetic check-in. Ask ONE question at a time about:
1. Hunger/appetite (1-10 scale)
2. Side effects (nausea, fatigue, injection site reactions)
3. Mood and energy
4. Medication adherence
5. Any concerns
Keep each question short. After 5 exchanges, summarize the check-in data in a JSON block like: [CHECKIN_DATA: {...}]`,

  eligibility: `You are ObesityCare AI screening a patient for trial eligibility.
Current trial: STEP-OB-24 (Semaglutide extended therapy)
Inclusion criteria: BMI ≥ 30 (or ≥ 27 with comorbidity), age 18-70, no prior GLP-1 therapy, willing to modify lifestyle.
Exclusion criteria: pregnancy, severe renal impairment, personal/family history of MTC, pancreatitis history, active eating disorder.
Ask ONE screening question at a time. Be clinical but friendly. After collecting enough info, give a clear ELIGIBLE / POTENTIALLY ELIGIBLE / NOT ELIGIBLE verdict with reasoning. Never give a definitive medical clearance — always say the care team will review.`,

  education: `You are ObesityCare AI, an educational assistant specializing in obesity medicine, GLP-1 therapy, nutrition, and lifestyle modification. Prefer clear, evidence-based statements; when citing mechanisms or guidelines, speak at a population level and avoid overstating certainty. When discussing behavior change, you may briefly reference well-supported ideas (e.g., realistic action planning, building self-efficacy through small successes) without claiming individualized treatment.
The participant is on program day ${programDay} of ${TOTAL_PROGRAM_DAYS}. Keep responses to 3-5 sentences unless the user asks for more detail.
Always end with an invitation to ask a follow-up question.`,

  // ── Study instruments: inst1 active; uncomment inst2–inst7 in SYSTEMS + modules + TABS + UI below ──
  inst1: `You are ObesityCare AI, administering a structured research questionnaire in a chat interface (not a formal diagnosis).

CONTEXT
- Participant: ${PATIENT.name}
- Trial: ${PATIENT.trial}
- Treatment context: ${PATIENT.drug}, program day ${programDay} (week ${programWeek} of ${PATIENT.totalWeeks})

INSTRUMENT ID (replace with your IRB / protocol name): Demographic Information.

ADMINISTRATION RULES
- Follow the numbered items below in order. Ask ONE item per message. Do not bundle multiple questions.
- When an item has response options, present them exactly as listed after you ask the question.
- If the user answers in their own words, map to the closest allowed option; if unclear, ask one short clarifying question.
- For the income item, accept a number or range; if they are unsure, offer "Prefer not to answer" only if your protocol allows.
- If the user skips a question, record "skipped" or "declined" only if allowed by protocol; otherwise explain once why the item matters and ask again.
- Stay neutral, respectful, and non-judgmental. Do not express surprise or opinions about answers.

SKIP LOGIC
- None unless you add protocol-specific rules here (e.g. "If answer to Q1 is X, skip Q3").

SCORING
- Categorical / descriptive only unless you add a scoring key: (e.g. "Sum items 2–5; higher = ...")

WHEN ALL ITEMS ARE COMPLETE
- Thank the participant briefly.
- Output one final line with a JSON object inside a tag, for research staff to copy, e.g.:
  [INSTRUMENT_DATA: {"q1_gender":"", "q2_household_income":"", "q3_ethnicity":"", "q4_race":"", "q5_education":"", "q6_marital_status":"", "q7_employment":""}]
- Use the exact keys above; fill string values from the conversation.

SAFETY
- If the user expresses self-harm, harm to others, severe distress, or urgent medical crisis, respond with empathy and include [ESCALATE] so the care team can follow up.
- Never diagnose, prescribe, or change medications; defer clinical decisions to the care team.

INSTRUMENT ITEMS (ask in this order)
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
  };
}

// inst2–inst7: add keys to buildSystems return, enable TABS + InstrumentModule2–7 when ready.

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

// ─── Chat Engine ──────────────────────────────────────────────
function ChatEngine({ systemKey, systems, placeholder, quickReplies = [], intro, persistInstrument }) {
  const [messages, setMessages] = useState(intro ? [{ role: "assistant", content: intro }] : []);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const send = useCallback(async (text) => {
    const userMsg = text.trim();
    if (!userMsg || loading) return;
    setInput("");
    const newMessages = [...messages, { role: "user", content: userMsg }];
    setMessages([...newMessages, { role: "assistant", content: "" }]);
    setLoading(true);

    const apiMessages = newMessages.map(m => ({ role: m.role, content: m.content }));
    let finalText = "";
    try {
      await callGroq(apiMessages, systems[systemKey], (chunk) => {
        finalText = chunk;
        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: "assistant", content: chunk };
          return updated;
        });
      });

      if (persistInstrument && finalText && !finalText.startsWith("⚠️")) {
        const parsed = extractInstrumentJson(finalText);
        if (parsed) {
          await persistInstrumentSubmission({
            instrumentKey: persistInstrument.key,
            instrumentLabel: persistInstrument.label ?? persistInstrument.key,
            responses: parsed,
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
            return updated;
          });
        }
      }
    } catch (e) {
      console.error("❌ Groq API error:", e);
      const errMsg = e?.message || String(e);
      let friendlyMsg = `⚠️ Error: ${errMsg}\n\nCheck the browser console (F12) for details.`;
      if (errMsg.includes("400")) friendlyMsg = "⚠️ API Error 400: Bad request — check your Groq API key and request payload.";
      if (errMsg.includes("403")) friendlyMsg = "⚠️ API Error 403: Permission denied — your Groq key may be invalid or expired.";
      if (errMsg.includes("429")) friendlyMsg = "⚠️ API Error 429: Rate limit hit. Wait a moment and try again.";
      if (errMsg.includes("Failed to fetch") || errMsg.includes("NetworkError")) friendlyMsg = "⚠️ Network error: Could not reach Groq API. Check your internet connection.";
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = { role: "assistant", content: friendlyMsg };
        return updated;
      });
    }
    setLoading(false);
  }, [messages, loading, systemKey, systems, persistInstrument]);

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input); }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      <div style={{ flex: 1, minHeight: 0, overflowY: "auto", overscrollBehavior: "contain", padding: "16px 20px 20px", display: "flex", flexDirection: "column", gap: 12 }}>
        {messages.map((m, i) => (
          <div key={i} style={{ display: "flex", gap: 10, alignItems: "flex-start", flexDirection: m.role === "user" ? "row-reverse" : "row", maxWidth: "90%", alignSelf: m.role === "user" ? "flex-end" : "flex-start" }}>
            <Avatar
              initials={m.role === "user" ? patientInitials(PATIENT.name) : "AI"}
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
            </div>
          </div>
        ))}
        {loading && messages[messages.length - 1]?.role !== "assistant" && (
          <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
            <Avatar initials="AI" color={T.chatAiAvatarFg} bg={T.chatAiAvatarBg} size={30} />
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

      <div style={{ padding: "10px 20px 16px", borderTop: `1px solid ${T.gray200}`, display: "flex", gap: 8, alignItems: "flex-end" }}>
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
    </div>
  );
}

// ─── Modules ──────────────────────────────────────────────────
function ChatModule({ systems, programDay, programWeek }) {
  return (
    <ChatEngine
      key={`chat-d${programDay}`}
      systems={systems}
      systemKey="chat"
      placeholder="Ask about activity goals, barriers, motivation, or your treatment..."
      intro={`Hello ${PATIENT.name}! I'm your Confident Moves PA coach. I'm here to help you build a sustainable, personalized physical activity routine that works with your body and your life.\n\nToday is program day ${programDay} (week ${programWeek}) — every step counts. We can work on setting goals, finding activities you enjoy, or problem-solving barriers like ${PATIENT.pa.topBarrier}.\n\nWhat's on your mind today?`}
      quickReplies={["Help me set a PA goal", "I'm struggling to stay motivated", "What activity suits my fitness level?", "How does exercise help with my weight loss?", "Had some side effects this week"]}
    />
  );
}

function CheckInModule({ systems, programDay, programWeek }) {
  return (
    <ChatEngine
      key={`checkin-d${programDay}`}
      systems={systems}
      systemKey="checkin"
      placeholder="Answer today's check-in questions..."
      intro={`Good morning ${PATIENT.name}! Time for your program day ${programDay} check-in (week ${programWeek}) — it takes about 2 minutes and helps your care team track your whole-person progress.\n\nI'll ask a few short questions covering activity, energy, appetite, and how you're feeling. Let's start: On a scale of 1–10, how would you rate your hunger and appetite today compared to before you started the program?`}
      quickReplies={["1–3 (much less hungry)", "4–6 (somewhat less hungry)", "7–10 (about the same)", "I forgot my medication today"]}
    />
  );
}

function EligibilityModule({ systems }) {
  return (
    <ChatEngine
      systems={systems}
      systemKey="eligibility"
      placeholder="Answer the screening questions..."
      intro={`Welcome to the STEP-OB-24 trial eligibility screener.\n\nThis brief questionnaire helps determine if you may qualify for our extended semaglutide therapy study. This is not a medical assessment — your care team will review and confirm any eligibility decision.\n\nLet's start with the basics: What is your current height and weight? (You can give approximate values)`}
      quickReplies={["I'm 5'6\", 210 lbs", "I'm 5'8\", 230 lbs", "I'm 5'4\", 195 lbs", "I'd rather answer questions one by one"]}
    />
  );
}

function EducationModule({ systems, programDay }) {
  return (
    <ChatEngine
      key={`edu-d${programDay}`}
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
    />
  );
}

function InstrumentModule1({ systems }) {
  return (
    <ChatEngine
      systems={systems}
      systemKey="inst1"
      placeholder="Answer each question as prompted..."
      intro={`Hi ${PATIENT.name}! This is a brief background survey — it helps the research team understand who is using the Confident Moves program so we can make it better for everyone.\n\nIt takes about 2 minutes and your answers are confidential. I'll ask one question at a time.\n\nReady to start?`}
      quickReplies={["Yes, let's start", "I have a question first", "What is this survey for?"]}
      persistInstrument={{ key: "inst1", label: "Demographics" }}
    />
  );
}

/*
function InstrumentModule2() {
  return (
    <ChatEngine
      systemKey="inst2"
      placeholder="Answer each question as prompted..."
      intro={`Hi ${PATIENT.name}! This module runs Instrument 2. Replace SYSTEMS.inst2 with your full instrument text, then edit this intro.\n\nWhen you're ready, reply with your first answer or type "start".`}
      quickReplies={["Start", "I have a question first", "Remind me what this is for"]}
    />
  );
}

function InstrumentModule3() {
  return (
    <ChatEngine
      systemKey="inst3"
      placeholder="Answer each question as prompted..."
      intro={`Hi ${PATIENT.name}! This module runs Instrument 3. Replace SYSTEMS.inst3 with your full instrument text, then edit this intro.\n\nWhen you're ready, reply with your first answer or type "start".`}
      quickReplies={["Start", "I have a question first", "Remind me what this is for"]}
    />
  );
}

function InstrumentModule4() {
  return (
    <ChatEngine
      systemKey="inst4"
      placeholder="Answer each question as prompted..."
      intro={`Hi ${PATIENT.name}! This module runs Instrument 4. Replace SYSTEMS.inst4 with your full instrument text, then edit this intro.\n\nWhen you're ready, reply with your first answer or type "start".`}
      quickReplies={["Start", "I have a question first", "Remind me what this is for"]}
    />
  );
}

function InstrumentModule5() {
  return (
    <ChatEngine
      systemKey="inst5"
      placeholder="Answer each question as prompted..."
      intro={`Hi ${PATIENT.name}! This module runs Instrument 5. Replace SYSTEMS.inst5 with your full instrument text, then edit this intro.\n\nWhen you're ready, reply with your first answer or type "start".`}
      quickReplies={["Start", "I have a question first", "Remind me what this is for"]}
    />
  );
}

function InstrumentModule6() {
  return (
    <ChatEngine
      systemKey="inst6"
      placeholder="Answer each question as prompted..."
      intro={`Hi ${PATIENT.name}! This module runs Instrument 6. Replace SYSTEMS.inst6 with your full instrument text, then edit this intro.\n\nWhen you're ready, reply with your first answer or type "start".`}
      quickReplies={["Start", "I have a question first", "Remind me what this is for"]}
    />
  );
}

function InstrumentModule7() {
  return (
    <ChatEngine
      systemKey="inst7"
      placeholder="Answer each question as prompted..."
      intro={`Hi ${PATIENT.name}! This module runs Instrument 7. Replace SYSTEMS.inst7 with your full instrument text, then edit this intro.\n\nWhen you're ready, reply with your first answer or type "start".`}
      quickReplies={["Start", "I have a question first", "Remind me what this is for"]}
    />
  );
}
*/

function HistoryModule() {
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
  const btnBase = {
    border: "none", cursor: exportLog.length ? "pointer" : "not-allowed", fontFamily: "'DM Sans', sans-serif",
    fontWeight: 600, fontSize: 12.5, padding: "8px 14px", borderRadius: 8, transition: "opacity .15s",
  };
  const dlJson = () => {
    const log = readInstrumentLog();
    if (!log.length) return;
    const name = `confident-moves-instruments-${PATIENT.id}-${Date.now()}.json`;
    triggerDownload(name, "application/json", JSON.stringify(log, null, 2));
  };
  const dlCsv = () => {
    const log = readInstrumentLog();
    if (!log.length) return;
    const name = `confident-moves-instruments-${PATIENT.id}-${Date.now()}.csv`;
    triggerDownload(name, "text/csv;charset=utf-8", instrumentLogToCsv(log));
  };

  return (
    <div style={{ padding: 20, overflowY: "auto", height: "100%" }}>
      <div style={{
        marginBottom: 24, padding: 16, background: "#fff", border: `1px solid ${T.gray200}`, borderRadius: 10,
      }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: T.gray500, textTransform: "uppercase", letterSpacing: ".05em", marginBottom: 8 }}>
          Assessment data export
        </div>
        <p style={{ fontSize: 12.5, color: T.gray600, lineHeight: 1.55, marginBottom: 14 }}>
          Download completed instrument submissions stored in this browser ({exportLog.length} record{exportLog.length !== 1 ? "s" : ""}).
          Data appears after the Demographics chat finishes and includes an <code style={{ fontSize: 11 }}>[INSTRUMENT_DATA: …]</code> line.
        </p>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
          <button type="button" disabled={exportLog.length === 0} onClick={dlJson} style={{
            ...btnBase, background: T.teal, color: "#fff", opacity: exportLog.length ? 1 : 0.45,
          }}>Download JSON</button>
          <button type="button" disabled={exportLog.length === 0} onClick={dlCsv} style={{
            ...btnBase, background: "#fff", color: T.tealDark, border: `1px solid ${T.teal}`, opacity: exportLog.length ? 1 : 0.45,
          }}>Download CSV</button>
        </div>
      </div>

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
// Instruments 2–7: uncomment matching rows in TABS, module header, and module content when you enable inst2–inst7 in SYSTEMS + InstrumentModule2–7.
const TABS = [
  { id: "chat", label: "PA Coach", icon: "🏃" },
  { id: "checkin", label: "Daily Check-in", icon: "✅" },
  { id: "eligibility", label: "Program Eligibility", icon: "🔬" },
  { id: "education", label: "Learn & Explore", icon: "📚" },
  { id: "inst1", label: "Demographics", icon: "📋" },
  // { id: "inst2", label: "Assessment 2", icon: "📋" },
  // { id: "inst3", label: "Assessment 3", icon: "📋" },
  // { id: "inst4", label: "Assessment 4", icon: "📋" },
  // { id: "inst5", label: "Assessment 5", icon: "📋" },
  // { id: "inst6", label: "Assessment 6", icon: "📋" },
  // { id: "inst7", label: "Assessment 7", icon: "📋" },
  { id: "history", label: "My Progress", icon: "📈" },
];

export default function App() {
  const saved = loadProgramState();
  const initial = saved ?? defaultProgramState();

  const [activeTab, setActiveTab] = useState("chat");
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

  const programWeek = Math.min(Math.ceil(programDay / 7), PATIENT.totalWeeks);
  const prevProgramWeekRef = useRef(programWeek);

  useEffect(() => {
    try {
      localStorage.setItem(
        PROGRAM_STATE_KEY,
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
  }, [programDay, currentWeight, weekPaMins, activeDaysThisWeek, lastPaLogProgramDay, lastActiveMarkProgramDay]);

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
    const weightOk = Number.isFinite(w) && w > 0 ? w : PATIENT.weight.current;
    return {
      programDay,
      programWeek,
      currentWeight: weightOk,
      estBmi: estBmiFromWeight(weightOk),
      weeklyPaMins: weekPaMins,
      weeklyPaGoal: PATIENT.pa.weeklyGoalMins,
      activeDaysThisWeek,
      activeDaysGoal: PATIENT.pa.goalDays,
    };
  }, [programDay, programWeek, currentWeight, weekPaMins, activeDaysThisWeek]);

  const systems = useMemo(() => buildSystems(runtime), [runtime]);

  const weightLoss = Math.max(0, PATIENT.weight.baseline - currentWeight);
  const toGoal = PATIENT.weight.baseline - PATIENT.weight.goal;
  const weightGoalPct = toGoal > 0 ? Math.min(100, Math.round((weightLoss / toGoal) * 100)) : 0;
  const trialPct = Math.min(100, Math.round((programDay / TOTAL_PROGRAM_DAYS) * 100));
  const paPct = Math.min(100, Math.round((weekPaMins / PATIENT.pa.weeklyGoalMins) * 100));
  const activeDaysPct = Math.min(100, Math.round((activeDaysThisWeek / PATIENT.pa.goalDays) * 100));

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
    setProgramDay((d) => Math.min(d + 1, TOTAL_PROGRAM_DAYS));
  };

  const resetProgramStart = () => {
    const d = defaultProgramState();
    setProgramDay(d.programDay);
    setCurrentWeight(d.currentWeight);
    setWeekPaMins(d.weekPaMins);
    setActiveDaysThisWeek(d.activeDaysThisWeek);
    setLastPaLogProgramDay(d.lastPaLogProgramDay);
    setLastActiveMarkProgramDay(d.lastActiveMarkProgramDay);
    setDraftWeight(String(d.currentWeight));
    setDraftTodayPa("");
    prevProgramWeekRef.current = 1;
  };

  const initials = patientInitials(PATIENT.name);

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
          <Badge children={`${PATIENT.trial}`} color={T.purple} bg={T.purpleLight} />
          <Badge children={`Day ${programDay} / ${TOTAL_PROGRAM_DAYS}`} color={T.tealDark} bg={T.tealLight} />
          <Badge children={`Week ${programWeek}/${PATIENT.totalWeeks}`} color={T.teal} bg={T.tealLight} />
          <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "5px 12px", border: `1px solid ${T.gray200}`, borderRadius: 20, background: T.gray50 }}>
            <Avatar initials={initials} size={22} />
            <span style={{ fontSize: 12, fontWeight: 500, color: T.gray700 }}>{PATIENT.name}</span>
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
              <button type="button" onClick={goNextProgramDay} disabled={programDay >= TOTAL_PROGRAM_DAYS} style={{
                flex: 1, fontFamily: "'DM Sans', sans-serif", fontWeight: 600, fontSize: 11.5,
                padding: "7px 7px", borderRadius: 8, border: `1px solid ${T.gray300}`, background: "#fff", color: T.gray800,
                cursor: programDay >= TOTAL_PROGRAM_DAYS ? "not-allowed" : "pointer",
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
              <MetricCard label="Weekly PA" value={`${weekPaMins}/${PATIENT.pa.weeklyGoalMins}′`} sub={`${paPct}%`} progress={paPct} compact sidebar />
              <MetricCard label="Active days" value={`${activeDaysThisWeek}/${PATIENT.pa.goalDays}`} sub="week" progress={activeDaysPct} compact sidebar />
              <MetricCard
                label="Weight lost"
                value={`−${weightLoss} lbs`}
                sub={`${weightGoalPct}% · BMI ${estBmiFromWeight(currentWeight)}`}
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
              <div style={{ lineHeight: 1.4, fontSize: 12, color: T.gray600 }}>{PATIENT.doctor}</div>
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
                {activeTab === "chat" && "Real-time PA coaching, goal setting & whole-person support — Groq AI"}
                {activeTab === "checkin" && "Daily wellness, activity & medication check-in"}
                {activeTab === "eligibility" && "Automated program eligibility screening — STEP-OB-24"}
                {activeTab === "education" && "Evidence-based PA education & obesity medicine"}
                {activeTab === "inst1" && "Assessment 1 of 7 — Demographic Information Survey"}
                {/*
                {activeTab === "inst2" && "Assessment 2 — edit SYSTEMS.inst2 & sidebar label"}
                {activeTab === "inst3" && "Assessment 3 — edit SYSTEMS.inst3 & sidebar label"}
                {activeTab === "inst4" && "Assessment 4 — edit SYSTEMS.inst4 & sidebar label"}
                {activeTab === "inst5" && "Assessment 5 — edit SYSTEMS.inst5 & sidebar label"}
                {activeTab === "inst6" && "Assessment 6 — edit SYSTEMS.inst6 & sidebar label"}
                {activeTab === "inst7" && "Assessment 7 — edit SYSTEMS.inst7 & sidebar label"}
                */}
                {activeTab === "history" && "Activity log, visit records & health notes"}
              </div>
            </div>
            <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: T.tealMid, marginTop: 4 }} />
              <span style={{ fontSize: 12, color: T.gray500 }}>AI coaching active</span>
            </div>
          </div>

          {/* Module content */}
          <div style={{ flex: 1, minHeight: 0, overflow: "hidden" }}>
            {activeTab === "chat" && <ChatModule systems={systems} programDay={programDay} programWeek={programWeek} />}
            {activeTab === "checkin" && <CheckInModule systems={systems} programDay={programDay} programWeek={programWeek} />}
            {activeTab === "eligibility" && <EligibilityModule systems={systems} />}
            {activeTab === "education" && <EducationModule systems={systems} programDay={programDay} />}
            {activeTab === "inst1" && <InstrumentModule1 systems={systems} />}
            {/*
            {activeTab === "inst2" && <InstrumentModule2 />}
            {activeTab === "inst3" && <InstrumentModule3 />}
            {activeTab === "inst4" && <InstrumentModule4 />}
            {activeTab === "inst5" && <InstrumentModule5 />}
            {activeTab === "inst6" && <InstrumentModule6 />}
            {activeTab === "inst7" && <InstrumentModule7 />}
            */}
            {activeTab === "history" && <HistoryModule />}
          </div>
        </div>
      </div>
    </div>
  );
}
