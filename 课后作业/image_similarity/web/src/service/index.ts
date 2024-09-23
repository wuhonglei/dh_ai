import { SimilarImageItem } from "../interface";

export async function compareTwoImage(
  image1: File,
  image2: File,
  modelName: string
): Promise<number> {
  const formData = new FormData();
  formData.append("image1", image1);
  formData.append("image2", image2);
  formData.append("model", modelName);
  const res = await fetch("/api/compare/images", {
    method: "POST",
    body: formData,
  });
  const data = await res.json();
  return data.similarity;
}

export async function searchImages(
  image: File,
  modelName: string,
  limit: number
): Promise<SimilarImageItem[]> {
  const formData = new FormData();
  formData.append("image", image);
  formData.append("model", modelName);
  formData.append("limit", String(limit));

  const res = await fetch("/api/search/images", {
    method: "POST",
    body: formData,
  });
  const data = await res.json();
  return data?.images || [];
}
