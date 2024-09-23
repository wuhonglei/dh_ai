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

export async function searchImages(image: File): Promise<string[]> {
  const formData = new FormData();
  formData.append("image", image);
  const res = await fetch("/api/search/images", {
    method: "POST",
    body: formData,
  });
  const data = await res.json();
  return data;
}
