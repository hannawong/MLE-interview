# 剑指Offer36. 二叉搜索树与双向链表

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。



![img](https://assets.leetcode.com/uploads/2018/10/12/bstdlloriginalbst.png)





```python
class Solution:
    node_list = []
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        if not root:
            return None
        self.node_list = []
        self.inorder(root)
        print(self.node_list)
        for i in range(len(self.node_list)-1):
            self.node_list[i].right = self.node_list[i+1]
            self.node_list[i+1].left = self.node_list[i]
        self.node_list[0].left = self.node_list[-1]
        self.node_list[-1].right = self.node_list[0]
        return self.node_list[0]
    
    def inorder(self,root):
        if not root:
            return
        self.inorder(root.left)
        self.node_list.append(root)
        self.inorder(root.right)
```





