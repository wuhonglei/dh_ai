from string import ascii_lowercase
from collections import Counter


class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        """
        ① 初始化字典顺序，降序排列
        ② 遍历 s 字符串，统计出现次数
        3️⃣ 对于重复的字母，比较不同保留位置时的字典序，确保保留的需求字母序最小
        """
        visited = {
            char: 0 for char in ascii_lowercase[::-1]
        }
        count = Counter(s)
        stack = []  # 单调递增栈
        for char in s:
            if visited[char] == 0:
                top_char = stack[-1] if stack else None
                while top_char and top_char > char and count[top_char] > 0:
                    visited[top_char] = 0  # 出栈
                    stack.pop()
                    top_char = stack[-1] if stack else None
                visited[char] = 1  # 入栈
                stack.append(char)
            count[char] -= 1
        return ''.join(stack)


s = Solution()
print(s.removeDuplicateLetters("bcabc"))
# print(s.removeDuplicateLetters("cbacdcbc"))
# print(s.removeDuplicateLetters("abacb"))
