class Solution:
    value_symbols = [
        (1000, 'M'),
        (900, 'CM'),
        (500, 'D'),
        (400, 'CD'),
        (100, 'C'), # 百位
        (90, 'XC'),
        (50, 'L'),
        (40, 'XL'),
        (10, 'X'), # 十位
        (9, 'IX'),
        (5, 'V'), 
        (4, 'IV'), 
        (1, 'I')
    ]

    def intToRoman(self, num: int) -> str:
        ans = ''
        for value, symbol in Solution.value_symbols:
            while num >= value:
                num -= value
                ans += symbol
            if num == 0:
                break
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