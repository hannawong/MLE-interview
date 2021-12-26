### 1. 层次遍历

```python
queue = []
max_list = []
queue.append(root)
while not len(queue) == 0:
	ans = []
    for i in range(len(queue)):
    	front = queue[0]
        ans.append(front.val)
        if front.left:
        	queue.append(front.left)
        if front.right:
            queue.append(front.right)
        queue = queue[1:]
```



### 2. 找到从根节点到叶子节点的所有路径

```python
self.num = []
self.num_list = []

def find_path(self,root):
        if root and not root.left and not root.right:
            self.num_list.append(self.num+[str(root.val)])
            return
        if not root:
            self.num_list.append(self.num)
            print(self.num)
            return
        self.num.append(str(root.val))
        self.find_path(root.left)
        self.num.pop()

        self.num.append(str(root.val))
        self.find_path(root.right)
        self.num.pop()
```



### 3. 中序后继

```python
def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode': ##O(logn)
        cur = root
        result = None
        while (cur) :
            if (cur.val > p.val):
                result = cur
                cur = cur.left
            else:
                cur = cur.right
            
        return result

```

