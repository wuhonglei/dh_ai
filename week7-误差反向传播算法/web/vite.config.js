import { defineConfig } from "vite";
import watchPublic from "./watchPublic";

export default defineConfig({
  plugins: [watchPublic()],
  server: {
    proxy: {
      "/predict": "http://localhost:5000",
    },
  },
});
