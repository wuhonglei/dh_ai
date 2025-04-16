class Solution:
    char_map = {
        4: [(1000, 'M')], # 千位
        3: [(900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C')], # 百位
        2: [(90, 'XC'), (50, 'L'), (40, 'XL'), (10, 'X')], # 十位
        1: [(9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]
    }
    def intToRoman(self, num: int) -> str:
        n = len(str(num))
        ans = ''
        # 从最高位开始遍历
        while n > 0:
            char_list = Solution.char_map[n]
            for base, c in char_list:
                count = num // base
                if count > 0:
                    ans += c * count
                    num = num % base
                    if num == 0:
                        return ans
            n -= 1
        return ans

case_list = [
    (3749, "MMMDCCXLIX"),
    (58, "LVIII"),
    (1994, "MCMXCIV")
]

for case_item in case_list:
    s = Solution()
    element = s.intToRoman(case_item[0])
    assert element == case_item[1], f'expect {case_item[1]}, but get {element}'
    print('success')