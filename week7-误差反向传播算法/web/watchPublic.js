import fs from "fs";
import path from "path";

export default function watchPublic() {
  return {
    name: "watch-public",
    configureServer(server) {
      const publicDir = path.resolve(__dirname, "public");

      fs.watch(publicDir, { recursive: true }, (eventType, filename) => {
        if (filename) {
          server.ws.send({
            type: "full-reload",
            path: "*",
          });
        }
      });
    },
  };
}
