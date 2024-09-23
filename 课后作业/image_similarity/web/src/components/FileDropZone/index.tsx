import type { UploadProps } from "antd";
import { Upload } from "antd";
import { InboxOutlined } from "@ant-design/icons";

const { Dragger } = Upload;

interface FileDropZoneProps {
  onBeforeUpload: (file: File) => void;
  className?: string;
}

const FileDropZone = (props: FileDropZoneProps) => {
  const { onBeforeUpload, className } = props;
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

  return (
    <Dragger {...uploadProps} className={className}>
      <p className="ant-upload-drag-icon">
        <InboxOutlined />
      </p>
      <p className="ant-upload-text">点击或者拖拽文件到这个区域上传</p>
    </Dragger>
  );
};

export default FileDropZone;
