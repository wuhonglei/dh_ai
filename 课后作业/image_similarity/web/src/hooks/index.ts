import { useMemo } from "react";

export function useImageSrc(file: File | null): string {
  return useMemo(() => {
    if (!file) {
      return "";
    }

    return URL.createObjectURL(file);
  }, [file]);
}
