class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        numStack = []
        
        # 构建单调递增的数字串
        for digit in num:
            while k and numStack and digit < numStack[-1]:
                numStack.pop()
                k -= 1
        
            numStack.append(digit)
        
        # 如果 K > 0，删除末尾的 K 个字符 1 2 3 4 5
        finalStack = numStack[:-k] if k else numStack
        
        # 抹去前导零
        return "".join(finalStack).lstrip('0') or "0"


s = Solution()
print(s.removeKdigits('1432219', 3))
print(s.removeKdigits('10200', 1))
print(s.removeKdigits('10', 2))
