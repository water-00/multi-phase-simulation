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

   \<new-branch>一般取这次的开发目标为名字，比如我想写某一个模块的内容，就

   ```
   git checkout -b module1
   ```

   然后开始写。如果之前已经存在`module1`分支，则用如下命令切换：

   ```
   git checkout module1
   ```

   注意经常用如下命令检查你当前处于哪个分支，**不要随意在main分支上commit内容**！

   ```
   git branch
   ```

3. 现在我已经切换到module1分支并打算开发这一功能模块，在此期间我可以在module1分支上随意commit（不要随意在main分支上commit！除非你确定commit的内容已经绝对正确）：

   ```bash
   git branch # 确定没有在main分支上开发
     main
   * module1
   
   # 写代码......有了阶段性成果后可以
   
   git add . # .表示选择当前文件夹下所有文件，添加它们到待commit列表；.也可以换成具体的文件或文件夹
   git commit -m "我完成了module1的部分，提交一下"
   git push
   ```
   
   如果远端仓库还没有module1分支会报错：
   
   ```
   fatal: The current branch module1 has no upstream branch.
   To push the current branch and set the remote as upstream, use
   
       git push --set-upstream origin module1
   ```
   
   执行它给出的这行代码就行了，这会在远端创建一个module1分支。这样我们就push了新写的module1的代码。注意此时全是在module1分支下操作的，还没有对main分支做任何修改！如果此时你``git checkout main`会看到你刚刚写的代码都消失了，这是对的，因为我们就是不往main commit任何未完成的东西。此时再`git checkout module1`即可，东西会回来。
   
4. 当你把module1功能开发完全，确定不需要修改后，再把它merge到main分支中：

   ```bash
   git add .
   git commit -m "module1开发完啦！"
   git push
   git checkout main # 切换到main分支
   git pull origin main # 拉取最新的main分支更新 (可选)
   git merge module1 # 合并module1到main，将module1分支的更改合并到main分支
   ```

   