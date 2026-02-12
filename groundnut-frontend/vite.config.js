import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'url'
import { dirname } from 'path'

const __dirname = dirname(fileURLToPath(import.meta.url))

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, __dirname, "");
  const apiBase = env.VITE_API_BASE || "http://127.0.0.1:5000";
  
  return {
    plugins: [react()],
    server: {
      host: true,
      port: 5173,
      strictPort: true,
      // Paling pasti untuk ngrok (DEMO/dev)
      allowedHosts: true,

      // Ngrok pakai https -> HMR perlu wss + port 443
      hmr: {
        protocol: "wss",
        clientPort: 443,
      },
      proxy: {
        "/api": {
          target: apiBase,
          changeOrigin: true,
        },
      },
    },
  }
});
