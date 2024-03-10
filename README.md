# multi-phase-simulation

一次正确规范的push流程：

1. 在开始写代码之前使用如下命令拉取其他人的更新：

   ```
   git pull
   ```

   如果发生冲突，一般是自己在pull之前就修改了某个文件，本地该文件内容与远端仓库中的不一致，因此需要由你决定该文件是远端内容覆盖本地内容还是保留本地内容（这一步请遇到冲突时自行chatgpt）

2. 使用如下命令创建一个新分支，这会创建并切换到新分支中，并把main中的内容拷贝到新分支中，然后你可以在新分支中开始写或改代码：

   ```
   git checkout -b <new-branch>
   ```

   \<new-branch>一般取这次的开发目标为名字，。如果之前已经存在`<new-branch>`分支，则用如下命令切换分支即可：

   ```
   git checkout <new-branch>
   ```

   注意经常用如下命令检查你当前处于哪个分支，**不要随意在main分支上commit内容**！

   ```
   git branch
   ```

3. 现在我已经切换到\<new-branch>分支并打算开发这一功能模块，在此期间我可以在\<new-branch>分支上随意commit（不要随意在main分支上commit！除非你确定即将commit的内容绝对正确）：

   ```bash
   git branch # 如下输出，确定没有在main分支上开发
     main
   * <new-branch>
   
   # 写代码......有了阶段性成果后可以
   
   git add . # .表示选择当前文件夹下所有文件，添加它们到待commit列表；.也可以换成具体的文件或文件夹
   git commit -m "我完成了<new-branch>的一部分，提交一下"
   git push
   ```

   如果远端仓库还没有\<new-branch>分支会报错：

   ```
   fatal: The current branch <new-branch> has no upstream branch.
   To push the current branch and set the remote as upstream, use
   
       git push --set-upstream origin <new-branch>
   ```

   执行它给出的这行代码就行了，这会在远端创建一个\<new-branch>分支。这样我们就push了新写的\<new-branch>的代码。注意此时全是在\<new-branch>分支下操作的，还没有对main分支做任何修改！如果此时你`git checkout main`会看到你刚刚写的代码都消失了，这是对的，因为我们就是不往main commit任何未完成的东西。此时再`git checkout <new-branch>`即可，东西会回来。

4. 当你把\<new-branch>功能开发完全，确定不需要修改后，再把它merge到main分支中：

   ```bash
   git add .
   git commit -m "<new-branch>开发完啦！"
   git push
   git checkout main # 切换到main分支
   git pull origin main # 拉取最新的main分支更新 (可选)
   git merge <new-branch> # 合并<new-branch>到main，将<new-branch>分支的更改合并到main分支
   ```

   建议以上命令行不要直接复制，而是一行一行执行，这样如果出问题才知道是哪句出问题。merge时如果遇到冲突请自行chatgpt解决。

5. （可选）删除本地和远端的\<new-branch>：

   ```bash
   # 首先确保你不在<new-branch>上
   git branch -d <new-branch> # 在本地删除<new-branch>分支
   git push origin --delete <new-branch> # 删除远程的<new-branch>分支
   ```

   一般如果是比较大的模块开发就不建议删除开发的分支了，因为它包括日志等重要信息。

这就是一个完整的拉取、开发、提交、合并的流程，自己新建一个分支开发可以保证main分支的简洁。如果大家都直接在main分支上开发会使得冲突的概率加大，且如果你提交了错误代码，在main分支上的回滚甚至会影响到其他人已经提交的正确代码，因此正确的做法是：**永远在非main分支上修改、提交，直到你确定新分支下的代码是绝对正确的，再合并到main分支**（因此记得多用git branch看自己在哪个分支！）。

另外，如果提交了错误的代码，甚至删错了文件也不要慌，可以使用`git reset + 哈希值`将本地回滚到之前某次commit时的状态，具体的遇到问题去问chatgpt即可。

但至少有一点，不用害怕，无论你做了什么事，把远端仓库弄成了什么样子，只要不是故意删库跑路，那总能通过命令将github远端仓库与本地仓库恢复到之前一个正确的版本的。