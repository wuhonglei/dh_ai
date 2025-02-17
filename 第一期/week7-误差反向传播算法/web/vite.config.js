import { defineConfig } from "vite";
import watchPublic from "./watchPublic";

export default defineConfig({
  plugins: [watchPublic()],
  server: {
    proxy: {
      "/predict": "http://127.0.0.1:5000",
    },
  },
});
