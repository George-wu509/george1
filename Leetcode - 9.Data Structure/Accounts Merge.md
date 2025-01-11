
```
样例 1:
	输入:
	[
		["John", "johnsmith@mail.com", "john00@mail.com"],
		["John", "johnnybravo@mail.com"],
		["John", "johnsmith@mail.com", "john_newyork@mail.com"],
		["Mary", "mary@mail.com"]
	]
	
	输出: 
	[
		["John", 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com'],
		["John", "johnnybravo@mail.com"],
		["Mary", "mary@mail.com"]
	]

	解释: 
	第一个第三个John是同一个人的账户，因为这两个账户有相同的邮箱："johnsmith@mail.com".
	剩下的两个账户分别是不同的人。因为他们没有和别的账户有相同的邮箱。

	你可以以任意顺序返回结果。比如：
	
	[
		['Mary', 'mary@mail.com'],
		['John', 'johnnybravo@mail.com'],
		['John', 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com']
	]
	也是可以的。
```


```python
class Solution:
    def accountsMerge(self, accounts):
        self.initialize(len(accounts))
        email_to_ids = self.get_email_to_ids(accounts)
        
        # union
        for email, ids in email_to_ids.items():
            root_id = ids[0]
            for id in ids[1:]:
                self.union(id, root_id)
                
        id_to_email_set = self.get_id_to_email_set(accounts)
        
        merged_accounts = []
        for user_id, email_set in id_to_email_set.items():
            merged_accounts.append([
                accounts[user_id][0],
                *sorted(email_set),
            ])
        return merged_accounts
    
    def get_id_to_email_set(self, accounts):
        id_to_email_set = {}
        for user_id, account in enumerate(accounts):
            root_user_id = self.find(user_id)
            email_set = id_to_email_set.get(root_user_id, set())
            for email in account[1:]:
                email_set.add(email)
            id_to_email_set[root_user_id] = email_set
        return id_to_email_set
            
    def get_email_to_ids(self, accounts):
        email_to_ids = {}
        for i, account in enumerate(accounts):
            for email in account[1:]:
                email_to_ids[email] = email_to_ids.get(email, [])
                email_to_ids[email].append(i)
        return email_to_ids
        
    def initialize(self, n):
        self.father = {}
        for i in range(n):
            self.father[i] = i
            
    def union(self, id1, id2):
        self.father[self.find(id1)] = self.find(id2)

    def find(self, user_id):
        path = []
        while user_id != self.father[user_id]:
            path.append(user_id)
            user_id = self.father[user_id]
            
        for u in path:
            self.father[u] = user_id
            
        return user_id
```
pass