import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
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
