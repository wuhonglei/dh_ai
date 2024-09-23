import { useMemo } from "react";
import { useLocation } from "react-router-dom";

export function useImageSrc(file: File | null | undefined): string {
  return useMemo(() => {
    if (!file) {
      return "";
    }

    return URL.createObjectURL(file);
  }, [file]);
}

export function useSelectedKeys(): string[] {
  const location = useLocation();
  return useMemo(() => {
    const pathname = location.pathname;
    if (pathname === "/") {
      return ["/image-compare"];
    }

    return [location.pathname];
  }, [location.pathname]);
}
