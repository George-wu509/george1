
#### Python library 是否應該要有 `setup.py`?

**簡短回答**：在現代 Python 專案中，`pyproject.toml` 已經很大程度上取代了 `setup.py` 的必要性。對您的專案來說，**`pyproject.toml` 已經足夠**。

**詳細說明**： Python 的套件打包生態系統近年來有了很大的演進。`pyproject.toml` 是 PEP 518 中引入的標準，用於聲明專案的建構相依性 (build dependencies)。現在，它已經擴展為儲存專案元數據 (metadata)、相依性 (dependencies) 和工具設定的主要位置。

- **`pyproject.toml` (現代標準)**:
    
    - 這是一個靜態的、宣告式的設定檔。
        
    - 它告訴 `pip` 和其他建構工具 (如 `build`) 如何打包您的專案。
        
    - 幾乎所有的專案資訊，包括名稱、版本、相依性等，都可以定義在這裡。
        
- **`setup.py` (傳統方式)**:
    
    - 這是一個可執行的 Python 腳本。
        
    - 在過去，這是定義所有套件資訊的唯一方式。
        
    - 它的動態性（因為是程式碼）雖然靈活，但也可能導致建構過程不夠可靠。
        
    - 在某些非常複雜的情況下（例如，需要編譯 C++ 擴展且邏輯複雜），`setup.py` 仍然是必要的，但它會與 `pyproject.toml` 一起使用。
        

**結論**：對於 `FastTrack` 這個純 Python 函式庫，`pyproject.toml` 是更現代、更簡潔、更標準的選擇。您不需要額外建立 `setup.py` 檔案。

#### 2. `environment.yml` 是否應該是 `environment.yaml`?

兩者都是可以的。`.yml` 和 `.yaml` 都是 YAML 檔案的有效副檔名。

- **`.yml`**: 是更常見的縮寫，尤其在 Conda 和 Docker Compose 的社群中，大多數文件和範例都使用 `.yml`。它更簡潔。
    
- **`.yaml`**: 是官方推薦的副檔名，更具描述性。