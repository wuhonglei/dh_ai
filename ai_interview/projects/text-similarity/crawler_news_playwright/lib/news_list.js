() => {
  const selector = [
    '[data-spm-type="resource"]:not(.swiper-slide):not(.TPLImageTextFeedItem)',
    '[data-spm-type="resource"]:not(.swiper-slide)',
  ].find((selector) => document.querySelector(selector));

  if (!selector) {
    return [];
  }

  const result = Array.from(document.querySelectorAll(selector))
    .filter((newContainer) => !newContainer.textContent.includes("广告"))
    .map((node) => node.querySelector("a"))
    .filter((a) => a && a.href)
    .map((a) => a.href);

  const hrefList = new Set([...result].slice(0, 10));

  return Array.from(hrefList);
};
