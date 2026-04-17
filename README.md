# ObesityCare AI

A clinical AI chatbot built for obesity management clinical trials. It supports patients enrolled in GLP-1 therapy studies (e.g., STEP-OB-24 / Semaglutide) by providing AI-powered conversation, structured check-ins, and educational resources — all in a single web interface.

## Features

- **Chat** — General AI support for questions about treatment, nutrition, and the trial
- **Daily Check-in** — Structured symptom and adherence check-in for clinical records
- **Trial Eligibility Screener** — Conversational inclusion/exclusion screening for new patients
- **Education Hub** — Evidence-based Q&A on obesity medicine and GLP-1 therapy
- **Visit History** — Summary of past visits, upcoming appointments, and side effect log

All chat modules are powered by [Groq](https://groq.com) (Llama 3.3 70B) with streaming responses.

## Tech Stack

- [React 19](https://react.dev) + [Vite](https://vitejs.dev)
- Groq API (`llama-3.3-70b-versatile`) via streaming SSE
- Inline styles with a custom teal/purple clinical theme (no CSS framework)

## Getting Started

1. Clone the repo
2. Install dependencies:
   ```bash
   npm install
   ```
3. Create a `.env` file in the root with your Groq API key:
   ```
   VITE_GROQ_API_SECRET_KEY=your_groq_key_here
   ```
4. Start the dev server:
   ```bash
   npm run dev
   ```
5. Open [http://localhost:5173](http://localhost:5173)

## Deployment

To build for production:
```bash
npm run build
```

The `dist/` folder can be deployed to any static host (Vercel, Netlify, etc.). Make sure to add `VITE_GROQ_API_SECRET_KEY` as an environment variable in your hosting platform.

## Notes

- This tool is for research and demonstration purposes only. It is not a medical device and does not provide medical advice.
- Patient data in the current version is hardcoded as a demo (`PATIENT` object in `App.jsx`).
- The AI is instructed to always defer medical decisions to the care team and flag safety concerns with an `[ESCALATE]` tag.
