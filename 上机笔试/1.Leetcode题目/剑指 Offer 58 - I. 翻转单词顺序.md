#### [剑指 Offer 58 - I. 翻转单词顺序](https://leetcode.cn/problems/fan-zhuan-dan-ci-shun-xu-lcof/)

难度简单225

输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student. "，则输出"student. a am I"。

 

**示例 1：**

```
输入: "the sky is blue"
输出: "blue is sky the"
```

```python
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        s = s.strip()
        s = [_ for _ in s]
        i = 0; j = 0; cnt = 0
        while i < len(s) and j < len(s):
            while j < len(s) and s[j] != " ":
                s[cnt] = s[j]
                cnt += 1
                j += 1
            i = j
            while j < len(s) and s[i] == s[j] == " ":
                j += 1
            print(i,j)
            if cnt < len(s):
                s[cnt] = " "
            cnt += 1
            i = j
        s = s[:cnt-1]
        print(s)
        def reverse(s,begin,end):
            while begin <= end:
                s[begin],s[end] = s[end],s[begin]
                begin += 1
                end -= 1
            return s
        s = reverse(s,0,len(s)-1)
        i = 0; j=0
        while j < len(s):
            while j < len(s) and s[j] != " ":
                j += 1
            print(i,j)
            s = reverse(s,i,j-1)
            i = j
            while i<len(s) and j < len(s) and s[i] == s[j] == " ":
                i += 1
                j += 1
        return "".join(s)
            
```

