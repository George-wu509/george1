
```
那執行在git repo根目錄下執行 pip install -e . 主要就是執行setup.py,如果setup.py不在repo的根目錄而是其中一個子目錄, 可能就會Error?
```

沒錯，你理解得很對 👍

`pip install -e .` 的流程大致上是這樣：

1. **你所在的資料夾（.）** → 就是 `pip` 要嘗試安裝的專案根目錄。
    
2. `pip` 會檢查這個目錄下是否有 **`setup.py`** 或 **`pyproject.toml`**（新格式）。
    
3. 如果有，就執行對應的安裝流程：
    
    - `setup.py` → 執行其中的 `setup()`。
        
    - `pyproject.toml` → 用 PEP 517/518 的方式安裝。
        
4. 如果這些檔案不存在，就會出現錯誤，通常類似：
    
    `ERROR: File "setup.py" not found. Directory '.' is not installable.` 
    

---

### 回答你的情境

- **如果 `setup.py` 在 repo 根目錄** → `pip install -e .` 沒問題。
    
- **如果 `setup.py` 在子資料夾** → 在根目錄跑 `pip install -e .` 會失敗，因為 `pip` 找不到 `setup.py`。
    

---

### 解法有幾個：

1. 進到有 `setup.py` 的子目錄再執行：
    
    `cd subdir_with_setup pip install -e .`
    
2. 或者在 repo 根目錄執行，但指定子目錄：
    
    `pip install -e ./subdir_with_setup`
    
1. 如果專案作者本來就希望你在 repo 根目錄安裝，通常會在根目錄放一個 `setup.py` 或 `pyproject.toml`，再在裡面指定實際的程式碼位置。







## `pip install -e .` 常見錯誤與解法

|錯誤訊息 / 情況|可能原因|解法|
|---|---|---|
|`ERROR: File "setup.py" not found. Directory '.' is not installable.`|目前目錄沒有 `setup.py` 或 `pyproject.toml`|1. 進到有 `setup.py` 的資料夾再執行  <br>2. 或者指定子目錄：`pip install -e ./subdir`|
|`ERROR: Directory '.' is not installable. Neither 'setup.py' nor 'pyproject.toml' found.`|專案用錯資料夾當安裝入口|檢查 repo 說明文件，看應該在哪個目錄執行|
|`error: invalid command 'bdist_wheel'`|缺少 **wheel** 套件|安裝 wheel：`pip install wheel`|
|`ModuleNotFoundError` 在 import package 時發生|`setup.py` 的 `packages` 或 `py_modules` 沒設定正確，或資料夾缺少 `__init__.py`|修改 `setup.py` 的 `find_packages()` 或加上 `__init__.py`|
|安裝過程找不到需求的套件|`install_requires` 或 `requirements.txt` 沒被正確處理|手動安裝缺少的依賴：`pip install -r requirements.txt`|
|`pip` 執行但檔案沒有更新|用 `-e` 安裝後，程式碼改動應該立即生效，但可能路徑錯誤|確認在正確的 venv 環境內，並檢查 `pip show <package>` 的 `Location`|

---

## 📌 小技巧

1. **確認 setup/pyproject 是否存在**
    
    `ls | grep setup.py ls | grep pyproject.toml`
    
2. **測試 import 是否成功**
    
    `python -c "import mypackage; print(mypackage.__file__)"`
    
3. **快速移除重新安裝**
    
    `pip uninstall <package> pip install -e .`