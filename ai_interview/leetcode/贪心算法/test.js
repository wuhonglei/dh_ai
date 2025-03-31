/* "bcabc" */
var removeDuplicateLetters = function (s) {
  const vis = new Array(26).fill(0);

  function doesCharVis(ch) {
    return vis[ch.charCodeAt() - "a".charCodeAt()];
  }

  function setCharVis(ch) {
    vis[ch.charCodeAt() - "a".charCodeAt()] = 1;
  }

  function clearCharVis(ch) {
    vis[ch.charCodeAt() - "a".charCodeAt()] = 0;
  }

  const num = _.countBy(s);

  const sb = new Array();
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (!doesCharVis(ch)) {
      let sb_top = sb[sb.length - 1];
      // 栈顶元素大于当前元素，并且栈顶元素在后面还有，则出栈
      while (sb_top && sb_top > ch && num[sb_top] > 0) {
        clearCharVis(sb_top);
        sb.pop();
        sb_top = sb[sb.length - 1];
      }
      setCharVis(ch);
      sb.push(ch);
    }
    num[ch]--;
  }
  return sb.join("");
};
