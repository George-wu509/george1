

|                           |     |
| ------------------------- | --- |
| [[#### Git 某一側未同步而要遺棄改動]] |     |
| [[#### Git 要使用到過去Commit]] |     |
|                           |     |
|                           |     |


#### Git 某一側未同步而要遺棄改動
```

git merge --abort
git fetch --all
git reset --hard origin/main

git reset --hard origin/dev

```



#### Git 要使用到過去Commit
```
切換到特定commit
[1]  git pull
[2]  git log origin/dev --oneline --decorate       (按q就會跳出了)
[3]  git switch --detach xxxxxnumber    or   git checkout xxxxxnumber 
[4]  切回舊版本   git switch dev     or    git checkout dev
[5]  git pull origin dev

[6]  git checkout -b temp_fix_from_old 56aac4f
[7]

把特定commit存在另一個folder
[1]  git pull
[2]  git log origin/dev --oneline --decorate
[3]  git worktree add “D:\folder”  xxxxxnumber 
git worktree remove - -force "D:\folder"

把特定commit直接複製到另一個folder
[1]  git clone D:\your_repo_path D:\compare_repo_copy
[2]  cd D:\compare_repo_copy
[3]  git checkout a1b2c3d

把舊的commit復原成最新的並push
[1]  git switch dev
[2]  git pull origin dev
[3]  git revert --no-commit 445335f..HEAD
[4]  git commit -m "Restore dev to state of commit 445335f"
[5]  git push origin dev

其他
[1] 查目前的remote repo and branch:
[2] 查目前最新的commit:
[3]
```