import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// PRODUCTION LLM Phase 1 — optional dev proxy so frontend can call /api/chat locally without CORS:
//   server: { proxy: { '/api': { target: 'http://localhost:3001', changeOrigin: true } } }
// Set VITE_API_BASE_URL="" in dev to use proxy path /api/chat; production sets full backend URL.
export default defineConfig({
  plugins: [react()],
})
