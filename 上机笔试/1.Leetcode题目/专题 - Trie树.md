# Trie树

![img](https://pic2.zhimg.com/80/v2-9d07fbd164fc0d737aabe428b4484bd1_1440w.png)

```python
class Trie:

    def __init__(self):
        self.children = [None] * 26
        self.isend = False

    def insert(self, word: str) -> None:
        tmp = self
        for c in word:
            idx = ord(c) - ord("a")
            if tmp.children[idx]: ###already has one
                tmp = tmp.children[idx]
            else: ###make one!
                tmp.children[idx] = Trie()
                tmp = tmp.children[idx]
        tmp.isend = True

    def search(self, word: str) -> bool:
        tmp = self
        for c in word:
            idx = ord(c) - ord("a")
            if tmp.children[idx]:
                tmp = tmp.children[idx]
            else:
                return False
        return tmp.isend


    def startsWith(self, prefix: str) -> bool:
        tmp = self
        for c in prefix:
            idx = ord(c) - ord("a")
            if tmp.children[idx]:
                tmp = tmp.children[idx]
            else:
                return False
        return True
```





#### [1233. 删除子文件夹](https://leetcode-cn.com/problems/remove-sub-folders-from-the-filesystem/)

你是一位系统管理员，手里有一份文件夹列表 `folder`，你的任务是要删除该列表中的所有 **子文件夹**，并以 **任意顺序** 返回剩下的文件夹。

如果文件夹 `folder[i]` 位于另一个文件夹 `folder[j]` 下，那么 `folder[i]` 就是 `folder[j]` 的 **子文件夹** 。

文件夹的「路径」是由一个或多个按以下格式串联形成的字符串：'/' 后跟一个或者多个小写英文字母。

- 例如，`"/leetcode"` 和 `"/leetcode/problems"` 都是有效的路径，而空字符串和 `"/"` 不是。

**示例 1：**

```
输入：folder = ["/a","/a/b","/c/d","/c/d/e","/c/f"]
输出：["/a","/c/d","/c/f"]
解释："/a/b/" 是 "/a" 的子文件夹，而 "/c/d/e" 是 "/c/d" 的子文件夹。
```

题解：

首先，按照字典序排序；然后，依次把文件名（split "/"）插入trie树，如果再某次插入的时候发现遇到了之前的end标志，说明这个是一个子文件，无需加入最终的结果中。

```python
class Trie:
    def __init__(self):
        self.children = {} 
        self.is_end = False

    def insert(self,folder):
        folder = folder.split("/")
        tmp = self
        for i in range(len(folder)):
            if folder[i]!="":
                if folder[i] not in tmp.children.keys(): ##还没有这个词语，建立一个！
                    tmp.children[folder[i]] = Trie()
                    tmp = tmp.children[folder[i]]
                else:
                    if tmp.children[folder[i]].is_end: ##之前有endwith这个词语的
                        return False
                    tmp = tmp.children[folder[i]]
        tmp.is_end = True
        return True

   
class Solution:
    def removeSubfolders(self, folder: List[str]) -> List[str]:
        trie = Trie()
        folder = sorted(folder)
        ans = []
        for i in range(len(folder)):
            res = trie.insert(folder[i])
            if res == True:
                ans.append(folder[i])
        return ans
```



#### [421. 数组中两个数的最大异或值](https://leetcode-cn.com/problems/maximum-xor-of-two-numbers-in-an-array/)

难度中等428

给你一个整数数组 `nums` ，返回 `nums[i] XOR nums[j]` 的最大运算结果，其中 `0 ≤ i ≤ j < n` 。

**进阶：**你可以在 `O(n)` 的时间解决这个问题吗？

**示例 1：**

```
输入：nums = [3,10,5,25,2,8]
输出：28
解释：最大运算结果是 5 XOR 25 = 28.
```

解法：首先，把所有的数字都表示成32位二进制数，然后依次插入Trie树中。然后，再遍历数组的每个元素，对于其二进制表示，**尽量向反方向去找**（为了使异或值更大）。如果实在没有反方向，那么就只能向同方向找。由于每个数都必定是32位，所以都不用去标注is_end了。

```python
class Trie:
    def __init__(self):
        self.children = [None] * 2   ##要么转0，要么转1
        self.is_end = False

    def insert(self, binary): ##插入一个32位二进制数字
        tmp = self
        for digit in binary:
            if not tmp.children[digit]:  ##还没有这个字符，建立一个！
                tmp.children[digit] = Trie()
                tmp = tmp.children[digit]
            else:
                tmp = tmp.children[digit]
        tmp.is_end = True

    def find_max_xor(self, number):  ##找到trie树中和number异或值最大的数
        tmp = self
        ans = []
        for digit in number:
            if tmp.children[1-digit]: ##能走相反，就走相反
                ans.append(1-digit)
                tmp = tmp.children[1-digit]
            else:
                ans.append(digit)
                tmp = tmp.children[digit]
        return ans

class Solution:
    def to_binary(self,number): ##转为32位二进制数
        binary = []
        while number:
            digit = number % 2
            number = number // 2
            binary.append(digit)
        return [0]*(32-len(binary))+binary[::-1]
    def binary2num(self,binary):
        ans = 0
        for i in range(len(binary)):
            ans += 2**(31-i)*binary[i]
        return ans


    def findMaximumXOR(self, nums) -> int:
        mmax = 0
        trie = Trie()
        for num in nums:
            trie.insert(self.to_binary(num))
        for i in range(len(nums)):
            mmax = max(mmax,self.binary2num(trie.find_max_xor(self.to_binary(nums[i]))) ^ nums[i])
        print(mmax)
        return mmax
```

【总结】

- 整数转二进制写法：

  - ```python
    while number:
    	digit = number % 2
    	number = number // 2
    ```

    但是不要忘记这样得到的digit是**倒序**的！！

- 二进制转int：

  - ```python
    for i in range(len(binary)):
    	ans += 2**(31-i)*binary[i]
    ```

    注意这里是`31-i`而不是`32-i`, 需要注意细节





#### [820. 单词的压缩编码](https://leetcode.cn/problems/short-encoding-of-words/)

难度中等273

单词数组 `words` 的 **有效编码** 由任意助记字符串 `s` 和下标数组 `indices` 组成，且满足：

- `words.length == indices.length`
- 助记字符串 `s` 以 `'#'` 字符结尾
- 对于每个下标 `indices[i]` ，`s` 的一个从 `indices[i]` 开始、到下一个 `'#'` 字符结束（但不包括 `'#'`）的 **子字符串** 恰好与 `words[i]` 相等

给你一个单词数组 `words` ，返回成功对 `words` 进行编码的最小助记字符串 `s` 的长度 。

 

**示例 1：**

```
输入：words = ["time", "me", "bell"]
输出：10
解释：一组有效编码为 s = "time#bell#" 和 indices = [0, 2, 5] 。
words[0] = "time" ，s 开始于 indices[0] = 0 到下一个 '#' 结束的子字符串，如加粗部分所示 "time#bell#"
words[1] = "me" ，s 开始于 indices[1] = 2 到下一个 '#' 结束的子字符串，如加粗部分所示 "time#bell#"
words[2] = "bell" ，s 开始于 indices[2] = 5 到下一个 '#' 结束的子字符串，如加粗部分所示 "time#bell#"
```



题解：

把word逆序，插入trie树之中。需要注意的是，**需要先把words数组按照长度逆序排序！！** 

**如果后面的word没有出现在trie树中，那么就需要记录全部长度 + “#”；** 如果已经出现在了trie树中，则不需要记录任何东西。(如`["time","atime","btime"]`, 虽然`atime`和`btime`前面有重合，但是由于不是完全重合，还是需要全部重新记录)

```python

class Trie:
            def __init__(self):
                self.children = [None]*26 ##26个字母
                self.is_end = False ##是否表示一个字符串
                self.ans = 0
            def insert(self,string):
                tmp = self
                find = True ##判断是不是已经出现在了trie树中
                for letter in string:
                    idx = ord(letter) - ord('a')
                    if tmp.children[idx]:
                        tmp = tmp.children[idx]
                    else: ##没有，需要创建一个
                        tmp.children[idx] = Trie()
                        tmp = tmp.children[idx]
                        find = False
                
                tmp.is_end = True
                if not find:##如果后面的word没有出现在trie树中，那么就需要记录全部长度 + “#”
                    self.ans += len(string)+1

class Solution:
    def minimumLengthEncoding(self, words: List[str]) -> int:
        trie = Trie()
        words.sort(key = lambda x: -len(x)) ##排序
        for word in words:
            trie.insert(word[::-1]) ##倒序插入
        return trie.ans
```

