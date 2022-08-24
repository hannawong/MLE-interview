# Git Cheetsheet

#### 1. Setup

`git config --global user.name “[firstname lastname]”`

`git config --global user.email “[valid-email]”`

## 1. 提交

你的本地仓库由 git 维护的三棵“树”组成。第一个是你的 `工作目录`，它持有实际文件；第二个是 `暂存区（Index）`，它像个缓存区域，**临时保存你的改动**；最后是 `HEAD`，它指向你**最后一次commit的结果**。

当我们使用`git commit -m "代码提交信息"`之后，改动已经提交到了 **HEAD**，但是还没到你的远端仓库。执行`git push origin master`以将这些改动提交到远端仓库（可以把 *master* 换成你想要推送的任何分支）如果你还没有克隆现有仓库，并欲将你的仓库连接到某个远程服务器，你可以使用`git remote add origin <server>`添加。如此你就能够将你的改动推送到所添加的服务器上去了。就像下面这样：

```python
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/hannawong/test.git
git push -u origin main
```



- `git status`: show modified files in working directory, staged for your next commit
- `git rm --cached <file>...` : unstage
- `git diff`: diff of what is changed but not staged 
- `git diff --staged`: diff of what is staged but not yet commited
- `git log`: show all commits in the current branch’s history

## 2. 分支

分支是用来将特性开发绝缘开来的。在你创建仓库的时候，*master* 是“默认的”分支。在其他分支上进行开发，完成后再将它们合并到主分支上。

![img](https://rogerdudler.github.io/git-guide/img/branches.png)
`git checkout -b feature_x`: 创建一个叫做“feature_x”的分支，并切换过去：
切换回主分支：`git checkout master`
再把新建的分支删掉：`git branch -d feature_x`
除非你将分支推送到远端仓库，不然该分支就是 *不为他人所见的*：`git push origin <branch>`

要**合并**其他分支到你的当前分支（例如 master），执行：`git merge <branch>`。遗憾的是，这可能并非每次都成功，并可能出现*冲突（conflicts）*。 这时候就需要你手动修改这些文件来合并*冲突（conflicts）*。改完之后，你需要执行`git add <filename>`以将它们标记为合并成功。在合并改动之前，你可以使用如下命令预览差异：
`git diff <source_branch> <target_branch>`

- `git branch` list your branches, 并且可以看到当前处于什么分支之下
- `git diff main..new_branch`:show the diff of what is in branchA that is not in branchB
- `git merge [branch] `:merge the specified branch’s history into the current one



#### 3. 替换本地改动

- 假如你操作失误（当然，这最好永远不要发生），你可以使用`git checkout -- <filename>`替换掉本地改动。此命令会使用 HEAD 中的最新内容替换掉你的工作目录中的这个文件。已添加到暂存区的改动以及新文件都不会受到影响。
-  `git reset --hard b9edd2a6711583dda020f6cf935f0cae685bda1b`: clear staging area, rewrite working tree from specified commit