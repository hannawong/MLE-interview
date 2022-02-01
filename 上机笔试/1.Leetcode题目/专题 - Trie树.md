# Trie树

![img](https://pic2.zhimg.com/80/v2-9d07fbd164fc0d737aabe428b4484bd1_1440w.png)

```python
class Trie:
    def __init__(self):
        self.children = [None]*26
        self.is_end = False

    def insert(self,word):
        tmp = self
        for letter in word:
            idx = ord(letter) - ord('a')
            if not tmp.children[idx]: ##还没有这个字符，建立一个！
                tmp.children[idx] = Trie()
                tmp = tmp.children[idx]
            else:
                tmp = tmp.children[idx]
        tmp.is_end = True

    def search(self,word): ##word是否在trie树中
        tmp = self
        for letter in word:
            idx = ord(letter) - ord('a')
            if not tmp.children[idx]:
                return False
            tmp = tmp.children[idx]
        return tmp.is_end
        
    def startsWith(self,prefix):
        tmp = self
        for letter in prefix:
            idx = ord(letter) - ord('a')
            if not tmp.children[idx]:
                return False
            tmp = tmp.children[idx]
        return True


```

