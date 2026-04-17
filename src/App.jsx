import { useState, useRef, useEffect, useCallback } from "react";
 
const GROQ_API_KEY = import.meta.env.VITE_GROQ_API_SECRET_KEY;
if (!GROQ_API_KEY) {
  throw new Error("GROQ_API_SECTET_KEY is not set");
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

// ─── Theme tokens ──────────────────────────────────────────────
const T = {
  teal: "#0d9373",
  tealLight: "#e0f5ef",
  tealMid: "#1aab87",
  tealDark: "#075c49",
  purple: "#6c47d6",
  purpleLight: "#ede9fb",
  amber: "#d97706",
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
};

// ─── Patient data ──────────────────────────────────────────────
const PATIENT = {
  name: "Jane Doe",
  id: "PT-0042",
  trial: "STEP-OB-24",
  drug: "Semaglutide 2.4mg",
  week: 6,
  totalWeeks: 24,
  bmi: { current: 34.2, baseline: 38.1 },
  weight: { current: 198, baseline: 207, unit: "lbs", goal: 177 },
  adherence: 94,
  nextVisit: "Apr 9, 2026",
  doctor: "Dr. Patel",
  conditions: ["Type 2 diabetes", "Hypertension"],
  medications: ["Metformin 500mg", "Lisinopril 10mg"],
};

// ─── System prompts ───────────────────────────────────────────
const SYSTEMS = {
  chat: `You are ObesityCare AI, a clinical support assistant embedded in an obesity management trial platform. 
Patient: ${PATIENT.name}, Trial: ${PATIENT.trial}, Drug: ${PATIENT.drug}, Week ${PATIENT.week}/${PATIENT.totalWeeks}, BMI: ${PATIENT.bmi.current} (baseline ${PATIENT.bmi.baseline}), Comorbidities: ${PATIENT.conditions.join(", ")}.
Rules:
- Never diagnose, prescribe, or change medication instructions. Always defer to care team for medical decisions.
- Use motivational interviewing: empathetic, open-ended, non-judgmental.
- Flag any severe side effects, distress, or safety concerns with [ESCALATE] tag.
- Keep responses concise (2-4 sentences). Be warm but clinical.
- You can answer general questions about GLP-1 therapy, obesity, nutrition, and the trial.`,

  checkin: `You are ObesityCare AI conducting a structured daily check-in for a clinical trial participant.
Patient: ${PATIENT.name}, Week ${PATIENT.week}, Drug: ${PATIENT.drug}.
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

  education: `You are ObesityCare AI, an educational assistant specializing in obesity medicine, GLP-1 therapy, nutrition, and lifestyle modification.
Be clear, evidence-based, and accessible. Use analogies when helpful. Keep responses to 3-5 sentences unless the user asks for more detail.
Always end with an invitation to ask a follow-up question.`,
};

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

function MetricCard({ label, value, sub, progress, color = T.teal }) {
  return (
    <div style={{
      background: "#fff", border: `1px solid ${T.gray200}`,
      borderRadius: 10, padding: "12px 14px", marginBottom: 8
    }}>
      <div style={{ fontSize: 11, color: T.gray500, marginBottom: 4, fontWeight: 500 }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 700, color: T.gray800, lineHeight: 1.2 }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color, marginTop: 3, fontWeight: 500 }}>{sub}</div>}
      {progress != null && (
        <div style={{ height: 5, background: T.gray200, borderRadius: 3, marginTop: 8, overflow: "hidden" }}>
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
function ChatEngine({ systemKey, placeholder, quickReplies = [], intro }) {
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
    setMessages(newMessages);
    setLoading(true);

    const apiMessages = newMessages.map(m => ({ role: m.role, content: m.content }));
    let streamText = "";
    setMessages(prev => [...prev, { role: "assistant", content: "" }]);

    try {
      await callGroq(apiMessages, SYSTEMS[systemKey], (chunk) => {
        streamText = chunk;
        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: "assistant", content: chunk };
          return updated;
        });
      });
    } catch (e) {
      console.error("❌ Groq API error:", e);
      const errMsg = e?.message || String(e);
      let friendlyMsg = `⚠️ Error: ${errMsg}\n\nCheck the browser console (F12) for details.`;
      if (errMsg.includes("400")) friendlyMsg = "⚠️ API Error 400: Bad request — check your GEMINI_API_KEY is valid and not the placeholder.";
      if (errMsg.includes("403")) friendlyMsg = "⚠️ API Error 403: Permission denied — your key may be invalid. Check aistudio.google.com.";
      if (errMsg.includes("429")) friendlyMsg = "⚠️ API Error 429: Rate limit hit. Wait a moment and try again.";
      if (errMsg.includes("Failed to fetch") || errMsg.includes("NetworkError")) friendlyMsg = "⚠️ Network error: Could not reach Groq API. Check your internet connection.";
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = { role: "assistant", content: friendlyMsg };
        return updated;
      });
    }
    setLoading(false);
  }, [messages, loading, systemKey]);

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input); }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ flex: 1, overflowY: "auto", padding: "16px 20px", display: "flex", flexDirection: "column", gap: 12 }}>
        {messages.map((m, i) => (
          <div key={i} style={{ display: "flex", gap: 10, alignItems: "flex-start", flexDirection: m.role === "user" ? "row-reverse" : "row", maxWidth: "90%", alignSelf: m.role === "user" ? "flex-end" : "flex-start" }}>
            <Avatar
              initials={m.role === "user" ? "JD" : "AI"}
              color={m.role === "user" ? T.purple : T.teal}
              bg={m.role === "user" ? T.purpleLight : T.tealLight}
              size={30}
            />
            <div style={{
              padding: "10px 14px", borderRadius: m.role === "user" ? "16px 4px 16px 16px" : "4px 16px 16px 16px",
              background: m.role === "user" ? T.purple : T.gray100,
              color: m.role === "user" ? "#fff" : T.gray800,
              fontSize: 13.5, lineHeight: 1.6, whiteSpace: "pre-wrap",
              minHeight: m.content === "" ? 40 : "auto"
            }}>
              {m.content === "" ? <TypingDots /> : m.content}
            </div>
          </div>
        ))}
        {loading && messages[messages.length - 1]?.role !== "assistant" && (
          <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
            <Avatar initials="AI" color={T.teal} bg={T.tealLight} size={30} />
            <div style={{ padding: "10px 14px", background: T.gray100, borderRadius: "4px 16px 16px 16px" }}>
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
function ChatModule() {
  return (
    <ChatEngine
      systemKey="chat"
      placeholder="Ask anything about your treatment, nutrition, or trial..."
      intro={`Hello ${PATIENT.name}! 👋 I'm your ObesityCare assistant. I can help you track symptoms, answer questions about your ${PATIENT.drug} trial, and support your journey. You're in week ${PATIENT.week} of ${PATIENT.totalWeeks} — great progress so far!\n\nHow are you feeling today?`}
      quickReplies={["I'm feeling good today", "Had some side effects", "Question about my medication", "How does semaglutide work?", "Tips for managing nausea"]}
    />
  );
}

function CheckInModule() {
  return (
    <ChatEngine
      systemKey="checkin"
      placeholder="Respond to check-in questions..."
      intro={`Good morning ${PATIENT.name}! Time for your Week ${PATIENT.week} daily check-in. This takes about 2 minutes and helps your care team at the clinical site monitor your progress.\n\nI'll ask a few short questions. Let's start: On a scale of 1–10, how would you rate your hunger and appetite today compared to before you started the trial?`}
      quickReplies={["1-3 (much less hungry)", "4-6 (somewhat less hungry)", "7-10 (about the same as before)", "I forgot to take my medication"]}
    />
  );
}

function EligibilityModule() {
  return (
    <ChatEngine
      systemKey="eligibility"
      placeholder="Answer the screening questions..."
      intro={`Welcome to the STEP-OB-24 trial eligibility screener.\n\nThis brief questionnaire helps determine if you may qualify for our extended semaglutide therapy study. This is not a medical assessment — your care team will review and confirm any eligibility decision.\n\nLet's start with the basics: What is your current height and weight? (You can give approximate values)`}
      quickReplies={["I'm 5'6\", 210 lbs", "I'm 5'8\", 230 lbs", "I'm 5'4\", 195 lbs", "I'd rather answer questions one by one"]}
    />
  );
}

function EducationModule() {
  return (
    <ChatEngine
      systemKey="education"
      placeholder="Ask any question about obesity, GLP-1, nutrition..."
      intro={`Welcome to the ObesityCare Education Hub! I can explain your treatment in plain language, share evidence-based nutrition tips, help you understand lab results, or answer any questions about obesity medicine.\n\nWhat would you like to learn about today?`}
      quickReplies={[
        "How does GLP-1 therapy work?",
        "What foods work best with semaglutide?",
        "Why do I feel nauseous?",
        "What is a healthy rate of weight loss?",
        "How does obesity affect my heart?",
      ]}
    />
  );
}

function HistoryModule() {
  const visits = [
    { date: "Mar 26, 2026", type: "Telehealth visit", note: "Week 4 assessment. Weight −6 lbs. Mild nausea reported. Dose maintained at 1.0mg.", status: "completed" },
    { date: "Mar 12, 2026", type: "Check-in call", note: "Week 2 follow-up. Tolerating medication well. Started dietary log.", status: "completed" },
    { date: "Feb 27, 2026", type: "Enrollment visit", note: "Baseline recorded. BMI 38.1. Consent signed. First injection administered.", status: "completed" },
    { date: "Apr 9, 2026", type: "Telehealth visit", note: "Week 8 assessment scheduled with Dr. Patel.", status: "upcoming" },
  ];

  const sideEffects = [
    { date: "Apr 5", effect: "Mild nausea", severity: "low", time: "2h post-injection" },
    { date: "Mar 28", effect: "Fatigue", severity: "low", time: "afternoon" },
    { date: "Mar 15", effect: "Nausea", severity: "medium", time: "morning" },
    { date: "Mar 3", effect: "Injection site redness", severity: "low", time: "day 1" },
  ];

  const severityColor = { low: T.teal, medium: T.amber, high: T.red };
  const severityBg = { low: T.tealLight, medium: T.amberLight, high: T.redLight };

  return (
    <div style={{ padding: 20, overflowY: "auto", height: "100%" }}>
      <div style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: T.gray500, textTransform: "uppercase", letterSpacing: ".05em", marginBottom: 12 }}>Visit history</div>
        {visits.map((v, i) => (
          <div key={i} style={{
            display: "flex", gap: 12, marginBottom: 12,
            background: v.status === "upcoming" ? T.purpleLight : "#fff",
            border: `1px solid ${v.status === "upcoming" ? "#c4b5fd" : T.gray200}`,
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
const TABS = [
  { id: "chat", label: "Chat", icon: "💬" },
  { id: "checkin", label: "Daily check-in", icon: "📋" },
  { id: "eligibility", label: "Trial eligibility", icon: "🔬" },
  { id: "education", label: "Education hub", icon: "📚" },
  { id: "history", label: "Visit history", icon: "🗂️" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("chat");

  const weightLoss = PATIENT.weight.baseline - PATIENT.weight.current;
  const weightGoalPct = Math.round((weightLoss / (PATIENT.weight.baseline - PATIENT.weight.goal)) * 100);
  const bmiDrop = (PATIENT.bmi.baseline - PATIENT.bmi.current).toFixed(1);
  const trialPct = Math.round((PATIENT.week / PATIENT.totalWeeks) * 100);

  return (
    <div style={{ fontFamily: "'DM Sans', 'Helvetica Neue', sans-serif", background: T.gray50, height: "100vh", display: "flex", flexDirection: "column" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        @keyframes bounce { 0%,80%,100%{transform:translateY(0)} 40%{transform:translateY(-5px)} }
        ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-thumb { background: ${T.gray300}; border-radius: 2px; }
        textarea { font-family: 'DM Sans', sans-serif !important; }
      `}</style>

      {/* Top bar */}
      <div style={{ background: "#fff", borderBottom: `1px solid ${T.gray200}`, padding: "0 24px", display: "flex", alignItems: "center", gap: 16, height: 56, flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 32, height: 32, borderRadius: 8, background: T.teal, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
            </svg>
          </div>
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: T.gray900, lineHeight: 1 }}>ObesityCare AI</div>
            <div style={{ fontSize: 11, color: T.gray400, lineHeight: 1.4 }}>Clinical trial platform</div>
          </div>
        </div>

        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 12 }}>
          <Badge children={`Trial ${PATIENT.trial}`} color={T.purple} bg={T.purpleLight} />
          <Badge children={`Week ${PATIENT.week}/${PATIENT.totalWeeks}`} color={T.teal} bg={T.tealLight} />
          <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "5px 12px", border: `1px solid ${T.gray200}`, borderRadius: 20, background: T.gray50 }}>
            <Avatar initials="JD" size={22} />
            <span style={{ fontSize: 12, fontWeight: 500, color: T.gray700 }}>{PATIENT.name}</span>
          </div>
        </div>
      </div>

      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        {/* Sidebar */}
        <div style={{ width: 220, background: "#fff", borderRight: `1px solid ${T.gray200}`, display: "flex", flexDirection: "column", flexShrink: 0 }}>
          <div style={{ padding: "16px 16px 8px" }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: T.gray400, textTransform: "uppercase", letterSpacing: ".06em", marginBottom: 6 }}>Modules</div>
          </div>

          {TABS.map(tab => (
            <button key={tab.id} onClick={() => setActiveTab(tab.id)} style={{
              display: "flex", alignItems: "center", gap: 10, padding: "10px 16px",
              background: activeTab === tab.id ? T.tealLight : "transparent",
              border: "none", borderLeft: activeTab === tab.id ? `3px solid ${T.teal}` : "3px solid transparent",
              cursor: "pointer", textAlign: "left", width: "100%", transition: "all .15s"
            }}>
              <span style={{ fontSize: 15 }}>{tab.icon}</span>
              <span style={{ fontSize: 13, fontWeight: activeTab === tab.id ? 600 : 400, color: activeTab === tab.id ? T.tealDark : T.gray600 }}>{tab.label}</span>
            </button>
          ))}

          {/* Patient metrics */}
          <div style={{ marginTop: "auto", padding: "16px 14px", borderTop: `1px solid ${T.gray200}` }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: T.gray400, textTransform: "uppercase", letterSpacing: ".06em", marginBottom: 10 }}>Patient metrics</div>
            <MetricCard label="Current BMI" value={PATIENT.bmi.current} sub={`↓ ${bmiDrop} from baseline`} progress={Math.round((bmiDrop / PATIENT.bmi.baseline) * 100 * 5)} />
            <MetricCard label="Weight loss" value={`−${weightLoss} lbs`} sub={`${weightGoalPct}% toward goal`} progress={weightGoalPct} />
            <MetricCard label="Trial progress" value={`Wk ${PATIENT.week}/${PATIENT.totalWeeks}`} sub={`${trialPct}% complete`} progress={trialPct} color={T.purple} />
            <MetricCard label="Adherence" value={`${PATIENT.adherence}%`} sub="last 30 days" progress={PATIENT.adherence} />
            <div style={{ fontSize: 11, color: T.gray400, marginTop: 8 }}>
              <div style={{ marginBottom: 2 }}>Next visit: <span style={{ color: T.gray600, fontWeight: 500 }}>{PATIENT.nextVisit}</span></div>
              <div>{PATIENT.doctor}</div>
            </div>
          </div>
        </div>

        {/* Main panel */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          {/* Module header */}
          <div style={{ background: "#fff", borderBottom: `1px solid ${T.gray200}`, padding: "12px 24px", display: "flex", alignItems: "center", gap: 12, flexShrink: 0 }}>
            <span style={{ fontSize: 18 }}>{TABS.find(t => t.id === activeTab)?.icon}</span>
            <div>
              <div style={{ fontSize: 15, fontWeight: 600, color: T.gray800 }}>{TABS.find(t => t.id === activeTab)?.label}</div>
              <div style={{ fontSize: 12, color: T.gray400 }}>
                {activeTab === "chat" && "General support & questions — powered by Groq AI"}
                {activeTab === "checkin" && "Structured PRO collection for clinical records"}
                {activeTab === "eligibility" && "Automated inclusion/exclusion screening — STEP-OB-24"}
                {activeTab === "education" && "Evidence-based obesity medicine education"}
                {activeTab === "history" && "Visit records, side effects & clinical notes"}
              </div>
            </div>
            <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#22c55e", marginTop: 4 }} />
              <span style={{ fontSize: 12, color: T.gray500 }}>Live AI connection</span>
            </div>
          </div>

          {/* Module content */}
          <div style={{ flex: 1, overflow: "hidden" }}>
            {activeTab === "chat" && <ChatModule />}
            {activeTab === "checkin" && <CheckInModule />}
            {activeTab === "eligibility" && <EligibilityModule />}
            {activeTab === "education" && <EducationModule />}
            {activeTab === "history" && <HistoryModule />}
          </div>
        </div>
      </div>
    </div>
  );
}
