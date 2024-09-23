import { useRequest } from "ahooks";
import { Image } from "antd";

import FileDropZone from "../../components/FileDropZone";
import { searchImages } from "../../service";
import { useEffect, useMemo, useState } from "react";
import { useImageSrc } from "../../hooks";

import fallbackUrl from "./fallback.webp";
import { initialModel } from "../../components/ModelSelect/constant";
import ModelSelect from "../../components/ModelSelect";
import Gallery from "react-photo-gallery";
import { isEmpty } from "lodash-es";

export default function ImageSearch() {
  const [modelName, setModelName] = useState(initialModel);
  const [file, setFile] = useState<File>();
  const { run, data: result } = useRequest(searchImages, {
    manual: true,
  });
  useEffect(() => file && run(file, modelName), [file, run, modelName]);
  const imageSrc = useImageSrc(file);
  const photos = useMemo(() => {
    if (isEmpty(result) || !result) return [];
    return result.map((item) => ({
      src: "/images" + item.entity.filename,
    }));
  }, [result]);

  return (
    <section className="py-4 flex flex-col gap-2">
      <div>
        <ModelSelect value={modelName} onChange={setModelName} />
      </div>
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
      <Gallery photos={photos} />
    </section>
  );
}
