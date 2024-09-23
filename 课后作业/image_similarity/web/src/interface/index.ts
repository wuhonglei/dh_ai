export interface SimilarImageItem {
  distance: number;
  entity: {
    width: number;
    height: number;
    filename: string; // 例如 /test/lynx/n02127052_258.JPEG
  };
  id: number;
}
