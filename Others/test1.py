# Subets
'''
class Solution:
    def run(self, nums):
        results = []
        nums.sort()
        self.dfs(nums, 0, [], results)
        return results
     
    def dfs(self, nums, k, num, results): 
        results.append(num[:])
        
        for i in range(k, len(nums)):
            num.append(nums[i])
            self.dfs(nums, i+1, num, results)
            del num[-1]
'''

# permutation
'''
class Solution:
    def run(self, nums):
        results = []
        visited = [False]*len(nums)
        self.dfs(nums, visited, [], results)
        return results

    def dfs(self, nums, visited, num, results):
        if len(num) == len(nums):
            results.append(list(num))
            return
        
        for i in range(len(nums)):
            if not visited[i]:
                visited[i] = True
                num.append(nums[i])
                self.dfs(nums, visited, num, results)
                num.pop()
                visited[i] = False
'''

# Combinations
class Solution:
    def run(self, n, k):
        results = []
        self.dfs(1, [], n, k, results)
        return results
    
    def dfs(self, pos, num, n, k, results):
        if len(num) == k:
            results.append(num[:])
            return
        
        if pos == n + 1:
            return
        
        if len(num) + n - pos + 1 < k:
            return
        
        num.append(pos)
        self.dfs(pos + 1, num, n, k, results)
        num.pop()
        
        self.dfs(pos + 1, num, n, k, results)    


sol = Solution()
nums = [1,2,3]
#result = sol.run(nums)
result = sol.run(4,2)
print(result)