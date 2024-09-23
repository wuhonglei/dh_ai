import { useState } from "react";
import { Card, Image, Button, Typography } from "antd";

import { useRequest } from "ahooks";

import FileDropZone from "../../components/FileDropZone";

import { useImageSrc } from "../../hooks";
import { compareTwoImage } from "../../service";
import ModelSelect from "../../components/ModelSelect";
import { initialModel } from "../../components/ModelSelect/constant";

const { Title } = Typography;

export default function ImageCompare() {
  const [modelName, setModelName] = useState(initialModel);
  const [file1, setFile1] = useState<File | null>(null);
  const [file2, setFile2] = useState<File | null>(null);
  const {
    run,
    loading,
    data: similarity,
    mutate: setSimilarity,
  } = useRequest(compareTwoImage, {
    manual: true,
  });

  const src1 = useImageSrc(file1);
  const src2 = useImageSrc(file2);
  const allowCompare = src1 && src2;

  // 处理文件的逻辑
  function handleDrop(file: File, position: "one" | "two"): void {
    if (position === "one") {
      setFile1(file);
    } else {
      setFile2(file);
    }
    setSimilarity(undefined);
  }

  async function handleCompare(): Promise<void> {
    run(file1!, file2!, modelName);
  }

  return (
    <section className="py-4 flex flex-col gap-2">
      <div>
        <ModelSelect value={modelName} onChange={setModelName} />
      </div>
      <div className="flex gap-4">
        <Card title="Image 1" className="flex-1">
          <FileDropZone onBeforeUpload={(file) => handleDrop(file, "one")} />
          <Image src={src1} preview={false} />
        </Card>
        <div className="w-24 text-center">
          <Button
            type="primary"
            className="mt-4"
            loading={loading}
            onClick={handleCompare}
            disabled={!allowCompare}
          >
            比较
          </Button>
          {allowCompare && similarity && (
            <Title level={5} className="mt-2">
              相似度:
              <br />
              {similarity.toFixed(4)}
            </Title>
          )}
        </div>
        <Card title="Image 2" className="flex-1">
          <FileDropZone onBeforeUpload={(file) => handleDrop(file, "two")} />
          <Image src={src2} preview={false} />
        </Card>
      </div>
    </section>
  );
}
