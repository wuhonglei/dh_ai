import { useRequest } from "ahooks";
import { Image, Select } from "antd";
import Gallery from "react-photo-gallery";
import { isEmpty } from "lodash-es";

import FileDropZone from "../../components/FileDropZone";
import { searchImages } from "../../service";
import { useEffect, useMemo, useState } from "react";
import { useImageSrc } from "../../hooks";

import fallbackUrl from "./fallback.webp";
import { initialModel } from "../../components/ModelSelect/constant";
import ModelSelect from "../../components/ModelSelect";

import { initialLimit, limitOptions } from "./constant";

export default function ImageSearch() {
  const [modelName, setModelName] = useState(initialModel);
  const [limit, setLimit] = useState(initialLimit);
  const [file, setFile] = useState<File>();
  const { run, data: result } = useRequest(searchImages, {
    manual: true,
  });
  useEffect(
    () => file && run(file, modelName, limit),
    [file, run, modelName, limit]
  );
  const imageSrc = useImageSrc(file);
  const photos = useMemo(() => {
    if (isEmpty(result) || !result) return [];
    return result.map((item) => ({
      src: "/images" + item.entity.filename,
      width: item.entity.width,
      height: item.entity.height,
    }));
  }, [result]);

  return (
    <section className="py-4 flex flex-col gap-2">
      <div className="flex gap-2">
        <ModelSelect value={modelName} onChange={setModelName} />
        <Select
          value={limit}
          options={limitOptions}
          onChange={(value) => setLimit(value)}
        />
      </div>
      <div className="flex h-40 gap-1">
        <FileDropZone
          useTarget={false}
          className=" flex-3"
          onBeforeUpload={(file) => setFile(file)}
        />
        <Image
          src={imageSrc}
          className="flex-1"
          fallback={fallbackUrl}
          style={{ height: "100%" }}
          preview={Boolean(imageSrc)}
        />
      </div>
      {!isEmpty(photos) && (
        <div className="mt-4">
          <Gallery photos={photos} direction="column" />
        </div>
      )}
    </section>
  );
}
