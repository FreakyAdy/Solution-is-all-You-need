import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/ui/",
  server: {
    port: 5173,
    proxy: {
      "/v1": "http://127.0.0.1:11411",
      "/api": "http://127.0.0.1:11411",
      "/phantom": {
        target: "http://127.0.0.1:11411",
        ws: true,
      },
    },
  },
});
