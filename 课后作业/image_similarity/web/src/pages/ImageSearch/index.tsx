import { useRequest } from "ahooks";
import { Image } from "antd";

import FileDropZone from "../../components/FileDropZone";
import { searchImages } from "../../service";
import { useEffect, useState } from "react";
import { useImageSrc } from "../../hooks";

import fallbackUrl from "./fallback.webp";

export default function ImageSearch() {
  const [file, setFile] = useState<File>();
  const { run, data: result } = useRequest(searchImages, {
    manual: true,
  });
  useEffect(() => file && run(file), [file, run]);
  const imageSrc = useImageSrc(file);

  console.info("result", result);

  return (
    <section className="py-4 flex flex-col gap-2">
      <div className="flex h-40 gap-1">
        <FileDropZone
          className=" flex-3"
          onBeforeUpload={(file) => setFile(file)}
        />
        <Image
          src={imageSrc}
          preview={false}
          className="flex-1"
          fallback={fallbackUrl}
          style={{ height: "100%" }}
        />
      </div>
    </section>
  );
}
