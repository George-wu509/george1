
|     |     |
| --- | --- |
|     |     |
|     |     |
|     |     |
|     |     |
|     |     |



```
在新的template(也就是利用update_template_addnew.py升級了template)裡面應該有rotation、af_mode_id、use_liqud_af, hdr_mode這些新parameter. 那請幫我檢查在執行App main.py的standard capture or hdr capture的camera pipeline是否會讀取到af_mode_id、use_liqud_af, hdr_mode這些設定然後用這些設定下面的parameter values進行拍照? 以及在DB/templates_import/run_workflow_demo.py是否會完整把template的所有數值包括rotation、af_mode_id、use_liqud_af, hdr_mode這些新parameter都有擷取出來存到taml file? 以及template_yaml_sync.py code的作用是否是把yaml file的內容存回local DB的template? 所以實際上如何操作請詳細說明, 以及要如何確保把yaml存回local DB不會讓template的某些數值遺漏或改動?
```

我檢查完了。結論先講：**App main.py 的 standard capture / HDR capture pipeline 會把這些新參數一路傳到 camera driver，而且 driver 真的會用它們拍照**。但前提是你實際載入的 template/version 真的已經包含這些欄位。

我也直接掃了目前 local DB：只有 run_16613T(V3) 的 v2 是完整帶新欄位；很多 v1 template 還缺 rotation、af_mode_id、use_liqud_af，test_hdr1 v1 也缺 hdr_mode。

**Capture Pipeline**

- App/main.py (line 11381) 會讀：
    - use_liqud_af / use_liquid_af
    - rotation
    - af_mode_id
- App/main.py (line 11520) capture overlay 會把 capture-level 的 use_liqud_af、hdr_mode 套進每次 capture 的 hw_cfg_run。
- App/main.py (line 11452) 最後呼叫 execute_template_point(hw_cfg)。
- Controller/hardware_drivers/unified_driver.py (line 4832) 會讀 af_mode_id，並透過 config/af_mode_config.yaml 暫時覆寫 AF config。
- Controller/hardware_drivers/unified_driver.py (line 4891) 會讀 use_liqud_af；如果是 false，會跳過 liquid lens AF，必要時用 manual cammag。
- Controller/hardware_drivers/unified_driver.py (line 4867) 會讀 rotation，最後在 line 5552 (line 5552) 對影像做 0/90/180/270 旋轉。
- Controller/hardware_drivers/unified_driver.py (line 2704) HDR path 會用 hw_cfg["hdr_mode"] resolve config/hdr_config.yaml 裡對應的 mode。

設計上目前是：rotation、af_mode_id 屬於 watch point level；use_liqud_af 屬於 capture level；hdr_mode 屬於 HDR capture level。這符合你說的 point-level metadata 和 image-level metadata 分開。

**run_workflow_demo.py**  
DB/templates_import/run_workflow_demo.py (line 464) 的 Step 2 是從 local DB local_templates.data 直接抓整包 JSON，再 dump 成 YAML。所以只要 DB 裡有 rotation、af_mode_id、use_liqud_af、hdr_mode，它會完整輸出，不是白名單欄位輸出。

但不要直接跑整支 run_workflow_demo.py，因為目前 RUN_STEP_1/2/3/6/7/8 都是 True，直接執行會 import、寫 DB、同步 AWS、匯出全 DB 等等。只要匯出 template，建議只呼叫 Step 2。

**template_yaml_sync.py**  
是的，DB/template_yaml_sync.py (line 220) 的 apply 作用就是把 YAML 內容存回 local DB template。它會呼叫 DatabaseManager.save_local_template (line 496)，把整包 YAML 轉成 JSON 存進 local_templates.data。

重要：它不是 deep merge，是整包覆蓋該 (TemplateID, version) row。它會刻意正規化/改動：

- TemplateID
- templateID
- templatename 若缺失
- version
- seriesversion
- watchView.version
- updated_at，除非加 --preserve-updated-at

**建議操作**

1. 先升級舊 template，先 dry-run：

powershell

`python DB\DB_test\update_template_addnew.py --template-id "test_hdr1" --version v1`

2. 確認 additions 沒問題後才寫新版本：

powershell

`python DB\DB_test\update_template_addnew.py --template-id "test_hdr1" --version v1 --apply`

3. 只匯出 local DB template YAML：

powershell

`python -c "from DB.db_manager import DatabaseManager; from DB.templates_import.run_workflow_demo import LOCAL_DB_PATH, step_2_query_LocalDB_template_and_export; db=DatabaseManager(LOCAL_DB_PATH); step_2_query_LocalDB_template_and_export(db); db.close()"`

4. 編輯 DB/templates_import/local_template_exports/<TemplateID>_<version>.yaml。
    
5. 存回 DB 前先 dry-run：
    

powershell

`python DB\template_yaml_sync.py --dry-run apply --yaml "DB\templates_import\local_template_exports\test_hdr1_v2.yaml"`

6. 真正寫入時建議先 bump version，不覆蓋原本版本：

powershell

`python DB\template_yaml_sync.py apply --yaml "DB\templates_import\local_template_exports\test_hdr1_v2.yaml" --bump-version`

7. 再匯出一次，比對 re-export 的 YAML。除了 version / updated_at 這類正規化欄位外，不應該有其他值改動或消失。

最關鍵的安全原則：**永遠從 DB 匯出的完整 YAML 開始改，不要手寫一個只含局部欄位的 YAML 再 apply**，因為 template_yaml_sync.py apply 會整包覆蓋，局部 YAML 會造成沒寫到的欄位消失。