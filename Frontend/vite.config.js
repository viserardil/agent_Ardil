import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // /api istekleri backend'e yönlendirilir. Proxy sayesinde frontend göreli yol
    // kullanır ("/api/..."), böylece ajanın ürettiği görsellerin URL'leri
    // (/api/artifacts/x.png) <img src> içinde doğrudan çalışır ve CORS'a takılmaz.
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      // Canlı STT WebSocket'i (/ws/stt) da backend'e yönlendirilir.
      '/ws': {
        target: 'ws://127.0.0.1:8000',
        ws: true,
        changeOrigin: true,
      },
    },
  },
})
