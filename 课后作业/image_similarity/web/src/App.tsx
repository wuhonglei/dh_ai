import { Card, Image, Button, Typography } from "antd";

import FileDropZone from "./components/FileDropZone";

import { useState } from "react";
import { useImageSrc } from "./hooks";
import { compareTwoImage } from "./service";

import "./App.css";

const { Title } = Typography;

function App() {
  const [file1, setFile1] = useState<File | null>(null);
  const [file2, setFile2] = useState<File | null>(null);
  const [similarity, setSimilarity] = useState<number>();

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
    const similarity = await compareTwoImage(file1!, file2!);
    setSimilarity(similarity);
  }

  return (
    <main className="container mx-auto flex gap-4 p-4">
      <Card title="Image 1" className="flex-1">
        <FileDropZone onBeforeUpload={(file) => handleDrop(file, "one")} />
        <Image src={src1} preview={false} />
      </Card>
      <div>
        <Button
          type="primary"
          className="mt-4"
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
    </main>
  );
}

export default App;
