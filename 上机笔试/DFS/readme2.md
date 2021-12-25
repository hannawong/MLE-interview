#  全排列 / DFS

#### 1. 所有可能的全排列

Problem1. 全排列

给定一个不含重复数字的数组 `nums` ，返回其 **所有可能的全排列** 。你可以 **按任意顺序** 返回答案。

```python
class Solution:
    ans = []
    ans_list = []
    def permute(self, nums: List[int]) -> List[List[int]]:
        self.ans_list.clear()
        self.ans.clear()
        
        visited = [0]* len(nums)
        length = len(nums)-1
        self.DFS(nums,visited,0,length)
        return self.ans_list

    def DFS(self,nums, visited, start_idx, length):
        if start_idx == length + 1:
            self.ans_list.append(self.ans[:])
        for i in range(len(nums)):
            if not visited[i]:
                visited[i] = 1
                self.ans.append(nums[i])
                self.DFS(nums,visited,start_idx+1,length)
                self.ans.pop()
                visited[i] = 0

```

给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。
给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。



**Problem2. 和为某个数的组合（允许重复）**

给定一个无重复元素的正整数数组 candidates 和一个正整数 target ，找出 candidates 中所有可以使数字和为目标数 target 的唯一组合。

candidates 中的数字可以无限制重复被选取。如果至少一个所选数字数量不同，则两种组合是唯一的。 

对于给定的输入，保证和为 target 的唯一组合数少于 150 个。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/Ygoe9J

```python
class Solution:
    ans = []
    ans_list = []
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        self.ans.clear() #不要忘记清空缓存
        self.ans_list.clear()
        self.DFS(0,candidates,target)
        ans_dedup = []
        for item in self.ans_list:
            item = sorted(item)
            if item not in ans_dedup:
                ans_dedup.append(item)
        return ans_dedup
        


    def DFS(self,now, candidates,target):
        if now == target:
            self.ans_list.append(self.ans[:])
            return
        if now > target:
            return
        
        for i in range(len(candidates)):
            self.ans.append(candidates[i])
            self.DFS(now + candidates[i],candidates,target)
            self.ans.pop()
```



**Problem2-1. 和为某个数的组合（不允许重复）**

遇到不允许重复的情况，就用visited数组来记录是否被访问过了。



**Problem3. 有效的括号**

正整数 n 代表生成括号的对数，请设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。

 

示例 1：

输入：n = 3
输出：["((()))","(()())","(())()","()(())","()()()"]

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/IDBivT

每个位置不是放左括号就是放右括号，所以只要对每个位置都遍历一下即可。还要保证左括号数永远都>右括号。

```python
class Solution(object):
    ans = ""
    ans_list = []
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        self.ans = ""
        self.ans_list = []
        self.DFS(0,0,n)
        return self.ans_list
    
    def DFS(self,left,right,n):
        if left == n and right == n:
            self.ans_list.append(self.ans[:])
        
        if left < right:
            return 
        if left > n or right > n:
            return 
        
        self.ans += "("
        self.DFS(left+1,right,n)
        self.ans = self.ans[:-1]

        self.ans += ")"
        self.DFS(left,right+1,n)
        self.ans = self.ans[:-1]
        
```



Problem4. 恢复ip地址(字符切分)

给定一个只包含数字的字符串 s ，用以表示一个 IP 地址，返回所有可能从 s 获得的 有效 IP 地址 。你可以按任何顺序返回答案。

有效 IP 地址 正好由四个整数（每个整数位于 0 到 255 之间组成，且不能含有前导 0），整数之间用 '.' 分隔。

例如："0.1.2.201" 和 "192.168.1.1" 是 有效 IP 地址，但是 "0.011.255.245"、"192.168.1.312" 和 "192.168@1.1" 是 无效 IP 地址。

 

示例 1：

输入：s = "25525511135"
输出：["255.255.11.135","255.255.111.35"]

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/0on3uN

```python
class Solution(object):
    ans = []
    ans_list = []

    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        self.ans_list = []
        self.ans = []

        self.DFS(0,0,len(s),s)
        return self.ans_list

    def is_valid(self, s):
        if 0 > int(s) or int(s) > 255:
            return False
        if s[0] == '0':
            if len(s) > 1:
                return False
        return True

    def DFS(self,idx,ip_num,length,s):
        if ip_num == 4 and idx >= length:
            this_ans = ".".join(self.ans)
            if this_ans not in self.ans_list:
                self.ans_list.append(this_ans)
            return
        if ip_num > 4:
            return
        if ip_num < 4 and idx >= length:
            return
        for i in range(1,4):
            cut = s[idx:idx+i]
            #print(cut)
            if self.is_valid(cut):
                self.ans.append(cut)
                self.DFS(idx+i,ip_num+1,length,s)
                self.ans.pop()

```



## 2. 树结构的递归



#### 2.1 展平多级双向链表

```
多级双向链表中，除了指向下一个节点和前一个节点指针之外，它还有一个子链表指针，可能指向单独的双向链表。这些子列表也可能会有一个或多个自己的子项，依此类推，生成多级数据结构，如下面的示例所示。

给定位于列表第一级的头节点，请扁平化列表，即将这样的多级双向链表展平成普通的双向链表，使所有结点出现在单级双链表中。


示例 1：

输入：head = [1,2,3,4,5,6,null,null,null,7,8,9,10,null,null,11,12]
输出：[1,2,3,7,8,11,12,9,10,4,5,6]
解释：

输入的多级列表如下图所示：

```

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/multilevellinkedlist.png)

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/Qv1Da2



```c++
/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* prev;
    Node* next;
    Node* child;
};
*/

class Solution {
public:
    Node* flatten(Node* head){
        Node* newhead = DFS(head);
        for(Node* tmp = newhead; tmp;tmp = tmp->next){
            cout<<newhead->val<<" ";
        }
        return newhead;
    }
    Node* DFS(Node* head) {
        if(head == NULL){ //不必展平
            return head;
        }
        if(head->child == NULL){
            Node* flatten_right = flatten(head->next); //展平后面的
            head->next = flatten_right; //和现在相连
            if(flatten_right) flatten_right->prev = head;
            return head;
        }
        else{ //有孩子
            Node* flatten_child = flatten(head->child); //把child展平
            head->child = NULL; //没有child了！！！！
            Node* tmp;
            for(tmp = flatten_child; tmp->next; tmp = tmp->next){}
            //tmp 是child尾
            Node* nnext = head->next;
            head->next = flatten_child;
            flatten_child->prev = head;
            tmp->next = nnext;
            if (nnext) nnext->prev = tmp;
            return head;
        }
    }
};
```

