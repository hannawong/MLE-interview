### 【典型题】连续子数组

解法一：滑动窗口

```c++
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int n = nums.size();
        int left_ptr = 0;
        int right_ptr = 0;
        int cnt = 0;
        int min_len = INT_MAX;

        while(right_ptr < n){
            cnt += nums[right_ptr];
            right_ptr++; //右窗口移动
            while(cnt >= target){ //符合条件了！
                min_len = min(min_len,right_ptr-left_ptr);
                cnt -= nums[left_ptr];
                left_ptr++;
            }
        }
        if(min_len == INT_MAX) return 0;
        return min_len;
    }
};
```

解法二：前缀和





### 【典型题】回文串

【判断一个字符串是否为回文串】的算法是O(n)的，一个指针指向头，一个指针指向尾，两者逐渐向中心靠拢。

【判断一个字符串每个区间是否是回文的】的算法是O(n^2), 用dp可以解决。而不用暴力的O(n^3)方法。



【中心扩展法】：计算【最长回文子串】，【列举所有的回文子串】

```c++
// 列举所有的回文子串

class Solution {
public:
    int expand(string s, int left,int right){
        while(left >= 0 && right <= s.length()){
            if(s[left] != s[right]){
                break;
            }
            else{
                left -- ;
                right ++;
            }
        }
        return right-left-1;
    }
    int countSubstrings(string s) {
        int len = s.length();
        int cnt = 0;
        for (int i = 0;i<len;i++)
            cnt += (expand(s,i,i)+1)/2;
        for(int i = 0;i<len-1;i++){
            cnt += expand(s,i,i+1)/2;
        }
        return cnt;
    }
};
```



```c++
//分割回文子串：暴力法判断是否为回文+递归（全排列方法）
//输入：s = "google"
//输出：[["g","o","o","g","l","e"],["g","oo","g","l","e"],["goog","l","e"]]

//链接：https://leetcode-cn.com/problems/M99OJA

class Solution:
    ans_list = []
    ans = []
    def ispandlindrome(self,s,begin,end):  ### 判断s的[begin,end]是否为回文串
        while(begin <= end):
            if s[begin] != s[end]:
                return False
            else:
                begin += 1
                end -= 1
        return True

    def partition(self, s: str) -> List[List[str]]:
        self.ans = []
        self.ans_list = []
        self.DFS(0,s)
        return self.ans_list
    
    def DFS(self,now_idx,s):
        if now_idx >= len(s):
            self.ans_list.append(self.ans[:])
        
        for i in range(now_idx,len(s)):
            if self.ispandlindrome(s,now_idx,i):
                tmp = now_idx
                now_idx = i+1
                self.ans.append(s[tmp:i+1])
                self.DFS(now_idx,s)
                self.ans.pop()
                now_idx = tmp
```

