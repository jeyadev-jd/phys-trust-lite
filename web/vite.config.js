import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// dev: npm run dev (port 5173) proxies the API; prod: FastAPI serves web/dist on :8600
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8600",
      "/results": "http://localhost:8600",
      "/ws": { target: "ws://localhost:8600", ws: true },
    },
  },
});
