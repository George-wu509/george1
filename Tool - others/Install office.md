Youtube
https://www.youtube.com/watch?v=vRCV38wrl9s&t=65s

Step1: 在桌面或其他folder (Example: D:\office) 創建"office" folder
Step2: 記得uninstall 所有舊版的office
Step3: 用第一個link(1.office 软件部署工具)按Download. 會下載office 软件部署工具到桌面
Step4: 執行office 软件部署工具, 選擇安裝到這個"office"folder. 應該會有兩個file
(setup.exe, configuration-Office365-x64.xml)

Step5: 用第二個link(office 版本自定义工具)按照選項勾選
(64位元, Office LTSC Professional Plus 2024 - Volume License, 版本:最新, 安裝:CDN,
對使用者顯示安裝:開啟, 關閉正在執行的應用程式:開啟, 解除安裝所有 Office MSI 版本::開啟
Office LTSC Professional Plus 2024 - Volume License: KMS)

Step6: 匯出, 保留當前設置, 勾選我接受許可, 文件名: config, 匯出
Step7: 把config.xml放到"office"folder, 把configuration-Office365-x64.xml刪除
Step8: 用admin開啟cmd, cd到"office"folder
Step9: cmd執行setup /download config.xml
Step10: cmd執行setup /configure config.xml
Step11: cd C:\Program Files\Microsoft Office\Office16
Step12: cscript ospp.vbs /sethst:kms.03k.org 
Step13: cscript ospp.vbs /act


**1.office 软件部署工具：[https://www.microsoft.com/en-us/download/details.aspx?id=49117](https://www.microsoft.com/en-us/download/details.aspx?id=49117)**

**2.office 版本自定义工具：[https://config.office.com/deploymentsettings](https://config.office.com/deploymentsettings)**

**3.基于KMS的 GVLK：[https://learn.microsoft.com/zh-cn/deployoffice/vlactivation/gvlks](https://learn.microsoft.com/zh-cn/deployoffice/vlactivation/gvlks)**



original video explanations:
然后以管理员身份运行CMD 进入命令终端
**下载命令：**
setup /download config.xml
**安装命令:**
setup /configure config.xml
**最后启动命令:**
cd C:\Program Files\Microsoft Office\Office16
cscript ospp.vbs /sethst:kms.03k.org 
cscript ospp.vbs /act
注意：如果你安装的是32位版本，那么启动命令第一个要改成：cd C:\Program Files (x86)\Microsoft Office\Office16