/*
 * 获取所有文本节点, 有效的文本节点需要同时满足以下条件:
 * 1. 文本节点的父元素没有子元素(即没有 DOM 元素节点)
 * 2. 文本节点的 nodeValue 去除首尾空白字符后不为空
 * 3. 文本节点不在 script 和 iframe 标签内
 */
function getTextNodes(node) {
  let textNodes = [];
  if (node.nodeType === Node.TEXT_NODE && node.nodeValue.trim() !== "") {
    if (node.parentElement.childElementCount === 0) {
      textNodes.push(node.parentElement);
    }

    // 处理 shopee PC 版本，商品标题前面包含一个 "Star+" 图片标记
    if (
      node.parentElement.childElementCount === 1 &&
      // img, svg, icon-font
      node.parentElement.firstElementChild.textContent.trim() === ""
    ) {
      textNodes.push(node.parentElement);
    }

    // 处理 jd PC 版本，商品标题前面包含一个 "自营" 文本标记
    if (
      node.parentElement.childElementCount === 1 &&
      // img, svg, icon-font
      node.parentElement.firstElementChild.textContent.trim().length <= 4
    ) {
      textNodes.push(node.parentElement);
    }
  } else if (!["SCRIPT", "IFRAME"].includes(node.tagName)) {
    let children = node.childNodes;
    for (let i = 0; i < children.length; i++) {
      textNodes = textNodes.concat(getTextNodes(children[i]));
    }
  }
  return textNodes;
}

let root = document.body;
let allTextNodes = getTextNodes(root);
// allTextNodes 去重
const nodeMap = new Map();
allTextNodes.forEach((node) => {
  nodeMap.set(node, true);
});
const uniqueTextNodes = Array.from(nodeMap.keys());
uniqueTextNodes.forEach((node) => {
  node.style.outline = "1px solid red";
  node.style.outlineOffset = "-2px";
});

const keywordSelector = {
  "www.pcworld.com": (node) => {
    if (node.tagName !== "A") {
      return false;
    }

    if (node.parentElement.tagName !== "H3") {
      return false;
    }

    return true;
  },
  "kinokuniya.com.sg": (node) => {
    if (node.tagName !== "H1") {
      return false;
    }

    const parentElement = node.parentElement;
    if (parentElement.tagName !== "DIV" || parentElement.className !== "info") {
      return false;
    }

    return true;
  },
  "www.amazon.sg": (node) => {
    if (node.tagName !== "SPAN" || !node.className.includes("a-text-normal")) {
      return false;
    }
    const parentElement = node.parentElement;
    if (
      parentElement.tagName !== "A" ||
      !parentElement.className.includes("a-link-normal")
    ) {
      return false;
    }

    return true;
  },
  "www.e-sentral.com": (node) => {
    if (
      node.tagName !== "DIV" ||
      !node.className.includes("font-bold line-clamp-1")
    ) {
      return false;
    }

    return true;
  },
  "www.klook.com": (node) => {
    if (node.tagName !== "A" || !node.href.includes("https://www.klook.com/")) {
      return false;
    }

    const parentElement = node.parentElement;
    if (parentElement.tagName !== "H3") {
      return false;
    }

    return true;
  },
  "www.popmart.com": (node) => {
    if (node.tagName !== "SPAN") {
      return false;
    }

    const parentElement = node.parentElement;
    if (
      parentElement.tagName !== "DIV" ||
      !parentElement.className.includes("index_title")
    ) {
      return false;
    }

    return true;
  },
  "tendencias.mercadolivre.com.br": (node) => {
    if (
      node.tagName !== "H3" ||
      !node.className.includes("ui-search-entry-keyword")
    ) {
      return false;
    }

    return true;
  },
};

const validateFn = keywordSelector[window.location.hostname];
uniqueTextNodes.forEach((node) => {
  if (!validateFn) {
    return;
  }

  const isKeywordTarget = validateFn(node);
  if (isKeywordTarget) {
    node.style.outline = "2px solid yellow";
  }
});
