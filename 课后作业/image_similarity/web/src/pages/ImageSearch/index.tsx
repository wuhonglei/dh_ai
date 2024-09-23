import { useRequest } from "ahooks";

import FileDropZone from "../../components/FileDropZone";
import { searchImages } from "../../service";

export default function ImageSearch() {
  const { run, data: result } = useRequest(searchImages, {
    manual: true,
  });

  console.log(result);

  return (
    <section className="py-4 flex flex-col gap-2">
      <FileDropZone onBeforeUpload={(file) => run(file)} />
    </section>
  );
}
