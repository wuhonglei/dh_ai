export async function compareTwoImage(
  image1: File,
  image2: File
): Promise<number> {
  const formData = new FormData();
  formData.append("image1", image1);
  formData.append("image2", image2);
  const res = await fetch("/api/compare/images", {
    method: "POST",
    body: formData,
  });
  const data = await res.json();
  return data.similarity;
}
