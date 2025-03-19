/**
 * 获取新浪财经新闻类目
 * @returns {Array} 类目列表
 */

() => {
  const categoryContainer = document.querySelector(".nav_header");
  const nodes = [...categoryContainer.childNodes].slice(1, -3);
  const result = nodes
    .filter((node) => node.nodeType == 1)
    .map((node) => node.querySelector("a"))
    .filter((a) => a && a.href)
    .map((a) => ({
      category: a.textContent.trim(),
      url: a.href,
    }))
    .filter(Boolean);

  return result;
};
