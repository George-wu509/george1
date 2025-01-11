
[https://github.com/George-wu509/WaveletSEG.git](https://github.com/George-wu509/WaveletSEG.git)

gh repo clone George-wu509/WaveletSEG

CREATE A NEW REPOSITORY ON THE COMMAND LINE

-------------------------------------

$ git init

$ git add README.md

$ git commit -m "first commit"

$ git remote add origin [https://github.com/George-wu509/WaveletSEG.git](https://github.com/George-wu509/WaveletSEG.git)

$ git push -u origin master

Upload file/folder to this NEW REPOSITORY ON THE COMMAND LINE

-------------------------------------

Open new empty  folder

$ git clone + repository address(for example: [https://github.com/George-wu509/Dinov2_foundation.git](https://github.com/George-wu509/Dinov2_foundation.git))

cd Dinob2_foundation

git init

git add .

git commit -m "message about the commit"

git push

PUSH AN EXISTING REPOSITORY FROM THE COMMAND LINE  把東西推上遠端的 Git 伺服

-------------------------------------

git remote add origin github.com/George-wu509/WaveletSEG.git

1. 需要設定一個端節的節點. (git remote 指令，顧名思義，主要是跟遠端有關的操作。add 指令是指要加入一個遠端的節點. 在這裡的 origin 是一個「代名詞」，指的是後面那串 GitHub 伺服器的位置。)    
2. 在慣例上，遠端的節點預設會使用 origin 這個名字。如果是從 Server 上 clone 下來的話，它的預設的遠端節點就會叫 origin。不過別擔心，這只是個慣例，不用這名字或是之後想要改也都可以，如果想改叫七龍珠 dragonball：$ git remote add dragonball [git@github.com:kaochenlong/practice-git.git](mailto:git@github.com:kaochenlong/practice-git.git)

git push -u origin master

 把 master 這個分支的內容，推向 origin 這個位置。在 origin 那個遠端 Server 上，如果 master 不存在，就建立一個叫做 master 的同名分支。但如果本來 Server 上就存在 master 分支，便會移動 Server 上 master 分支的位置，使它指到目前最新的進度上。設定 upstream，就是那個 -u 參數做的好事，這個稍候說明。

 LARGE FILE USING LFS

-------------------------------------

git lfs install

git lfs track "xxxxx.xx"

git remote add origin github.com/George-wu509/WaveletSEG.git

git push -u origin master

fatal: does not appear to a git repository Could not read from remote repository

-------------------------------------

git remote rm origin

git remote -v

Reference:

[https://gitbook.tw/chapters/github/push-to-github.html](https://gitbook.tw/chapters/github/push-to-github.html)

[https://gitbook.tw/chapters/using-git/add-to-git.html](https://gitbook.tw/chapters/using-git/add-to-git.html)