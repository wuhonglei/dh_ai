import type { UploadProps } from "antd";
import { Upload } from "antd";
import { InboxOutlined } from "@ant-design/icons";
import { useEventListener } from "ahooks";
import { useRef } from "react";
import { get } from "lodash-es";

const { Dragger } = Upload;

interface FileDropZoneProps {
  useTarget: boolean; // 是否限制 paste 事件的 target
  className?: string;
  onBeforeUpload: (file: File) => void;
}

const FileDropZone = (props: FileDropZoneProps) => {
  const { onBeforeUpload, className, useTarget = false } = props;
  const uploaderRef = useRef(null);

  const uploadProps: UploadProps = {
    name: "file",
    accept: "image/*",
    multiple: false,
    showUploadList: false,
    beforeUpload: (file) => {
      console.info("beforeUpload", file);
      onBeforeUpload(file);
      return false;
    },
  };

  useEventListener(
    "paste",
    (event: ClipboardEvent) => {
      console.info(event.target);
      const clipboardData = event.clipboardData;
      if (!clipboardData || !clipboardData.files.length) {
        return;
      }

      const files = clipboardData.files;
      const imgFile = [...files].find((file) => file.type.includes("image"));
      if (imgFile) {
        onBeforeUpload(imgFile);
      }
    },
    {
      target: useTarget
        ? get(uploaderRef, "current.nativeElement.parentElement")
        : document,
    }
  );

  return (
    <Dragger {...uploadProps} className={className} ref={uploaderRef}>
      <p className="ant-upload-drag-icon">
        <InboxOutlined />
      </p>
      <p className="ant-upload-text">点击或者拖拽文件到这个区域上传</p>
    </Dragger>
  );
};

export default FileDropZone;
