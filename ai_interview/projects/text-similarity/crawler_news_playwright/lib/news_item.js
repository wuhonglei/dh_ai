() => {
  const url = window.location.href;

  const titleDom = document.querySelector(".text-title > h1");
  const title = titleDom ? titleDom.textContent.trim() : "";

  const contentDom = document.querySelector("[data-spm='content'] .article");
  const content = contentDom ? contentDom.textContent.trim() : "";

  return { url, title, content };
};
