import random

class RandomizedSet:

    def __init__(self):
        self.nums = []
        self.indices = {}
        

    def insert(self, val: int) -> bool:
        if val in self.indices:
            return False
        
        self.indices[val] = len(self.nums)
        self.nums.append(val)
        return True
        

    def remove(self, val: int) -> bool:
        if val not in self.indices:
            return False
        
        id = self.indices[val]
        self.nums[id] = self.nums[-1]
        self.indices[self.nums[id]] = id
        self.nums.pop()
        del self.indices[val]
        return True
        

    def getRandom(self) -> int:
        return random.choice(self.nums)
        


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()