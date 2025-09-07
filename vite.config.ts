import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // String shorthand for simple proxy
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        // Rewrite the path: remove '/api' at the beginning
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
