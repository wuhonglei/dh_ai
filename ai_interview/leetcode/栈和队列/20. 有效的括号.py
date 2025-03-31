class Solution:
    def isValid(self, s: str) -> bool:
        if len(s) % 2 == 1:
            return False

        stack = []
        char_dict = {
            '}': '{',
            ']': '[',
            ')': '('
        }
        for char in s:
            if char not in char_dict:
                stack.append(char)
            else:
                if not stack or char_dict[char] != stack[-1]:
                    return False
                stack.pop()

        return len(stack) == 0


solution = Solution()
print(solution.isValid("()"))
print(solution.isValid('()[]{}'))
print(solution.isValid("(]"))
print(solution.isValid("([])"))
