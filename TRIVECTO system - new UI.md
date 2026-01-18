

|                             |     |
| --------------------------- | --- |
| [[#### PyDracula Github解釋]] |     |
| [[#### ui1-4介面的流程設定]]       |     |
|                             |     |
|                             |     |
|                             |     |
|                             |     |


#### PyDracula Github解釋
```
請幫我分析這個github https://github.com/Wanderson-Magalhaes/Modern_GUI_PyDracula_PySide6_or_PyQt6 的repo關於Qt designer的模板, 我想要更改UI的顏色, 以及想要做成類似我上傳這樣的UI, 我只知道基本的Qt desinger操作但不會使用themes, widgets. 應該要如何做請一步步從頭開始
```
下面我用「完全新手（只會基本 Qt Designer）」的角度，帶你**從 0 到 1**把 PyDracula 這個 repo 當成模板用，然後把 UI 改成你圖上那種「淺藍底 + 左側藍色側欄 + 大標題 + 輸入框」的風格。

> 先講重點：**在 PyDracula 這個專案裡，顏色/字體/圓角/hover…大多不是靠 Qt Designer 的 property 設定，而是靠 `.qss` 主題檔統一蓋過去。**  
> 所以你在 Designer 改顏色，跑起來常常「看起來沒變」，原因是 **QSS 覆蓋了 Designer 設定**。這在 repo 的討論裡也有人提到（最後是用 QSS 覆蓋才生效）。

---

## A. 先把 PyDracula 跑起來（確保環境 OK）

1. 到 repo：`Modern_GUI_PyDracula_PySide6_or_PyQt6` 下載 Code（ZIP）到本機解壓  
    repo 結構重點（README 有列）：
    

- `main.py`：程式入口
    
- `main.ui`：Qt Designer 檔
    
- `themes/`：主題 `.qss` 放這裡
    
- `modules/ui_main.py`：由 `main.ui` 轉出來的 Python UI 檔
    
- `resources.qrc` / `modules/resources_rc.py`：資源檔（圖示等）
    

2. 建議用 Python 3.9 + PySide6（README 有提醒版本相容性）
    
3. 建 venv + 安裝 PySide6（Windows PowerShell 範例）：
    

`cd 你的專案資料夾 py -3.9 -m venv .venv .\.venv\Scripts\activate pip install PySide6`

4. 跑起來：
    

`python main.py`

> 如果你是高 DPI 螢幕覺得字或圖怪怪的，README 有建議在 `main.py` 設 `QT_FONT_DPI=96` 的 workaround。

---

## B. 你需要知道的「PyDracula 變色」核心邏輯

### 1) 介面結構：Designer 管「佈局」，QSS 管「外觀」

- 你在 **Qt Designer** 主要做：
    
    - 左側欄/主畫面/按鈕/輸入框 的 **位置、大小、Layout**
        
    - widget 的 **objectName**（很重要，QSS 或程式會用它來套樣式/找元件）
        
- 你在 **QSS（themes/*.qss）** 主要做：
    
    - 背景色、按鈕顏色、hover/pressed、字體、圓角、邊框
        

### 2) `.ui` 改完要「重新轉出 Python」

README 明講：`main.ui` 需要用 `pyside6-uic` 轉成 `ui_main.py`。  
（很多人卡在：改了 main.ui 但程式還在用舊的 ui_main.py，所以看不到變化）

---

## C. 用 Qt Designer 做出你圖上的版面（一步一步）

你的圖大概是：

- 整個視窗背景：淺藍
    
- 左側 sidebar（固定寬）：藍色底 + 一堆按鈕（Watch Entry / View Images / Advanced / Engineering Mode / Reset / Exit）
    
- 主區塊：置中標題 “Check For Template” + 一個 QLineEdit
    

### Step C1 — 打開 `main.ui`

用 **Qt Designer（Qt6）** 打開 `main.ui`。

### Step C2 — 找到主容器（通常是 centralWidget 裡的 main frame）

PyDracula 通常會有：

- 左側 menu frame（可能叫 `left_menu_frame` / `leftMenu` 類似）
    
- 右側 content area（stackedWidget 或 pages）
    

如果你找不到，Designer 裡看 Object Inspector（左邊那棵樹），找：

- `QFrame`（左邊固定寬那個）
    
- `QStackedWidget`（右邊切頁用的）
    

### Step C3 — 做左側欄（sidebar）

在 Designer：

1. 選左側的 `QFrame`（sidebar）
    
2. 設定：
    
    - `minimumWidth` / `maximumWidth` 都設成同一個（例如 180 或 200），讓它固定寬
        
    - `objectName` 設成例如：`sidebar`
        
3. sidebar 裡放一個 `QVBoxLayout`
    
4. 依序放 `QPushButton`：
    
    - Watch Entry
        
    - View Images
        
    - Advanced
        
    - Engineering Mode
        
    - （加 stretch）
        
    - Reset System
        
    - Exit
        

> 小技巧：如果你希望按鈕高度一致，對每顆按鈕設 `minimumHeight`（例如 60）。

### Step C4 — 做右側主畫面

1. 在右側 content 的 page（可能是 stackedWidget 的其中一頁）放一個 container（例如 `QWidget` / `QFrame`）
    
2. 設 `objectName`：例如 `page_watch_entry`
    
3. 用 `QVBoxLayout`：
    
    - 上面放 `QLabel`（文字：Check For Template），置中
        
    - 中間放 `QLineEdit`，寬度可用 `maximumWidth`（例如 420）讓它不會撐滿
        

---

## D. 把顏色改成你圖上那套（最重要：做自己的 QSS 主題）

README 有說主題放在 `themes/`，而且 PyDracula 支援 multiple themes。

### Step D1 — 建一個新主題檔

在 `themes/` 新增一個檔案，例如：

- `themes/custom_lightblue.qss`
    

（你也可以先 copy 現有的 dark/light qss 再改，但就算你從空白開始也行）

### Step D2 — 寫最小可用的 QSS（先讓畫面長得像你圖）

先放這段（你可以直接貼到 `custom_lightblue.qss`）：

`/* ====== Global ====== */ QWidget {     background-color: #AFCFE8;   /* 淺藍底 */     color: #0B0B0B;     font-size: 18px; }  /* ====== Sidebar ====== */ QFrame#sidebar {     background-color: #36A9D4;   /* 左側欄藍色 */ }  QPushButton {     background-color: transparent;     border: none;     color: white;     padding: 14px 18px;     text-align: left;     font-size: 18px; }  QPushButton:hover {     background-color: rgba(255,255,255,0.18); }  QPushButton:pressed {     background-color: rgba(0,0,0,0.12); }  /* 你圖上 Watch Entry 那顆比較深色，可以用 objectName 指定 */ QPushButton#btn_watch_entry {     background-color: #0076B6;     font-weight: 600; }  /* ====== Main content widgets ====== */ QLabel#titleLabel {     font-size: 28px;     font-weight: 600;     color: #0B0B0B; }  QLineEdit {     background-color: white;     border: 2px solid rgba(0,0,0,0.20);     border-radius: 2px;     padding: 8px 10px;     font-size: 18px; }`

> 上面有用到 `#sidebar`、`#btn_watch_entry`、`#titleLabel`  
> 所以你要回到 Designer 把對應元件的 **objectName** 設成一樣，QSS 才會精準套到。

---

## E. 讓程式改用你的主題（custom_lightblue.qss）

你需要在程式裡找到「載入 QSS」的位置，通常會是：

- `main.py` 或 `modules/app_settings.py` 或 `modules/ui_functions.py` 之類（README 有提到 settings 與 ui functions 的位置）
    

**做法：**

1. 全專案搜尋字串：`.qss` 或 `themes/` 或 `setStyleSheet`
    
2. 你會看到類似：
    
    - 讀取 `themes/py_dracula_dark.qss`
        
3. 把路徑改成：
    
    - `themes/custom_lightblue.qss`
        

> 有人在 issue 提到打包時只吃到某個 qss，代表專案確實是靠「指定的 themes\xxx.qss」在跑。

---

## F. 你改了 main.ui 但沒反應？（99% 都是這個原因）

你在 Designer 改完 `main.ui` 後，記得重新輸出 Python UI 檔。README 給了指令範例：

`pyside6-uic main.ui > modules/ui_main.py`

另外如果你動到 `resources.qrc`（加圖示/圖片），要重新編 `resources_rc.py`：

`pyside6-rcc resources.qrc -o modules/resources_rc.py`

---

## G. 你接下來最穩的工作流（建議照這個走）

1. **先決定版面**：只在 Qt Designer 拖拉 Layout、設定 objectName
    
2. **再決定風格**：只在 `themes/custom_lightblue.qss` 調顏色、hover、字體
    
3. 每次改完 `.ui`：
    
    - 跑 `pyside6-uic` 重新生成 `modules/ui_main.py`
        
4. 每次改完 `.qss`：
    
    - 直接重跑 `python main.py` 就會看到變化
        

---

如果你願意，我也可以下一步直接幫你做「對照表」：  
把你這張 UI 圖拆成 **每個 widget 的 objectName、建議 Layout 結構、以及對應的 QSS selector**（你照抄就會 1:1 很像）。你只要告訴我：你想用 PyDracula 原本的左側 menu 動畫/折疊效果，還是你想要像圖那樣簡單固定側欄就好




#### ui1-4介面的流程設定
```
我上傳的四張image(ui1,ui2,ui3,ui4)是同一個ui的流程, 我主要想詢問的就是如何使用PyDracula能做到ui以及如下的功能. 從u1到u2代表左側的一排按鍵((u1圖裡左側))是可以往左縮回成一排小按鍵(u2圖裡左側), 而由u1,u2到u3是在中間空格填入文字click之後就進到下一個介面圖(u3圖)可供輸入Account跟Password, 而在輸入Account跟Password之後則進到下一個介面圖(u4圖)在上方多了一排按鍵. 所以這一系列動作進到下一個介面圖, 及像左側的可縮放的一排按鍵等等, 是如何由PyDracula能做到? 我只知道基本的Qt desinger操作但不會使用themes, widgets. 請非常詳細一步步從頭開始解釋每一步驟做成這樣的介面
```
你這 4 張圖其實就是典型的 PyDracula「三件套」：

1. **左側 Sidebar 可伸縮（u1 ⇄ u2）**
    
2. **主畫面用「頁面切換」（u1/u2 → u3 → u4）**
    
3. **u4 上方那排按鍵＝另一層「tab / page 切換」**
    

在 PyDracula 裡，這三件事分別對應到：

- **Qt Designer**：把「框架/頁面/按鈕/stackedWidget」畫好（管佈局）
    
- **ui_functions.py / app_functions.py（Python）**：寫「動畫、切頁、事件綁定」（管互動）
    
- **themes/*.qss（QSS）**：管顏色、hover、字體、圓角（管外觀；類似 CSS）
    

下面我用「你只會基本 Qt Designer」的方式，**一步一步從頭做出你 u1→u2→u3→u4 的流程**（而且是用 PyDracula 的架構做，不是從零硬寫）。

---

## 0. 先理解 PyDracula 專案你會碰到的檔案

PyDracula repo 的分工（README 直接寫）大概是：

- `main.ui`：Designer 檔（你改 UI 的主要地方）
    
- `modules/ui_main.py`：由 `main.ui` 轉出來的 Python（不要手改，改 ui 就重新轉）
    
- `modules/ui_functions.py`：**只放「UI 互動」**（像 toggle sidebar、換頁、動畫）
    
- `modules/app_functions.py`：放「你 app 的功能邏輯」（例如登入驗證、跑流程）
    
- `themes/`：放 `.qss` 主題（顏色幾乎都在這改）
    

你要做的 u1~u4，其實就是：

- main.ui 畫出：Sidebar + stackedWidget(頁面1/頁面2/頁面3) + u4 的 tabs
    
- ui_functions.py 寫：sidebar 寬度動畫 + setCurrentWidget 換頁
    
- theme qss 寫：你想要的藍色/淺藍色 UI
    

---

## 1) 用 Qt Designer 做「頁面切換」骨架（u1/u3/u4 都靠它）

你需要一個 `QStackedWidget` 來裝不同頁面（page）。  
Qt 官方/社群對 `QStackedWidget` 的切換方式就是 `setCurrentIndex()` 或 `setCurrentWidget()`。

### 1-1. 在 main.ui 放一個 `QStackedWidget`

在 Designer：

1. 找到右邊主內容區（content area）
    
2. 拖一個 `QStackedWidget`
    
3. 取 objectName：`stackedWidget`（或沿用 PyDracula 原本的名字也行）
    

### 1-2. 建 3 個頁面

在 `stackedWidget` 裡建立三個 page（右鍵 Add Page）：

- `page_template`（對應 u1/u2：Check For Template）
    
- `page_login`（對應 u3：Account / Password）
    
- `page_run`（對應 u4：上方 tabs + Run 按鈕）
    

每個 page 內照你的圖擺：

- u1/u2：一個大標題 QLabel + 一個 QLineEdit + 一個 Next/Check QPushButton（你圖上是輸入後 click 進下一頁，建議放一顆按鈕）
    
- u3：Account/Password 兩個 QLineEdit + Login 按鈕
    
- u4：上面 tabs + 中間 Run 按鈕
    

> 你 u1 圖看起來沒有 “Next” 按鈕，但你描述「填文字 click 之後進下一頁」，所以 UI 上一定有某個可 click 的 widget（按鈕、或輸入框的 returnPressed）。最簡單就是加一顆 Next。

---

## 2) 在 Designer 做左側 Sidebar（並準備伸縮 u1⇄u2）

PyDracula 的可伸縮 sidebar，原理幾乎都是：

- sidebar 是一個 `QFrame`
    
- 按一下「≪」按鈕
    
- 用 `QPropertyAnimation` 去 animate sidebar 的 `minimumWidth / maximumWidth`（寬度從 200 → 60，或反過來）  
    這是 Qt 常見動畫用法。
    

### 2-1. Sidebar 你要兩種狀態

- **展開狀態（u1）**：icon + 文字（Watch Entry / View Images…）
    
- **縮起狀態（u2）**：只剩 icon（或很短）
    

最實務、最好維護的方法是「一個 sidebar frame + 兩套子容器」：

- `frame_left_menu`（整個 sidebar 容器）
    
    - `widget_menu_expanded`（u1：有文字）
        
    - `widget_menu_compact`（u2：只有 icon）
        

切換時：

- animate `frame_left_menu` 寬度
    
- 同時 `expanded` / `compact` 其中一個 `setVisible(True/False)`
    

（你也可以只做一套按鈕、改文字透明/隱藏，但新手比較容易踩 QSS/大小問題）

### 2-2. 在 Designer 擺好

1. 左側放 `QFrame`，objectName：`frame_left_menu`
    
2. 設定展開寬度例如 180~220（你可照圖）
    
3. `frame_left_menu` 裡放 `QStackedWidget` 或兩個 `QWidget`
    
    - `widget_menu_expanded`：放你的文字按鈕（u1）
        
    - `widget_menu_compact`：放 icon-only 按鈕（u2）
        
4. 最上面放一顆 toggle 按鈕（≪ / ≫），objectName：`btn_toggle_menu`
    

> u1/u2 兩套按鈕最好保持「功能對應一致」：  
> 例如 expanded 的 `btn_watch_entry` 對應 compact 的 `btn_watch_entry_icon`。

---

## 3) 你要的「u4 上方 tabs」怎麼做？

u4 那排 “Front / Back / Open Back …” 最簡單有兩種做法：

### 做法 A：用 `QTabWidget`（最快）

Designer 直接拖 `QTabWidget` 到 `page_run` 上方  
每個 tab 內放你要的內容（或先留空）

### 做法 B：用「按鈕列 + 另一個 stackedWidget」（更像你圖的 “button tab”）

你圖像是「按鈕列」不是原生 tab 外觀，所以更推薦 B：

在 `page_run` 裡：

- 上方：`frame_top_tabs`（QHBoxLayout 放 6 顆 QPushButton）
    
- 下方：`stackedWidget_run_tabs`（6 個頁面：front/back/…）
    

按下哪個 tab button，就：

- `stackedWidget_run_tabs.setCurrentWidget(page_front)`
    
- 並把按鈕設成 checked 狀態（用 `QButtonGroup` 管理互斥）
    

---

## 4) 關鍵：把按鈕「連到」切頁與伸縮（Python：ui_functions / main.py）

到這裡 Designer 只完成了「畫面長相」，但 u1→u3→u4、u1⇄u2 都還不會動。  
下一步就是 PyDracula 的核心：**signals/slots + helper functions**。

> PyDracula 的 README 就說 `ui_functions.py` 是放 UI functions 的地方。  
> 另外 repo issue 也直接提到 ui_functions.py 會 `from main import *`（代表它確實是 UI 行為的集中處）。

下面我給你一個「最小可用」的寫法（你可以直接照抄再對照自己的 objectName 改）。

### 4-1. 在 main.py（或 MainWindow init）綁定事件

概念是：

- toggle 按鈕 → sidebar 動畫
    
- template next → 切到 login page
    
- login 成功 → 切到 run page
    
- u4 tabs → 切 run_tabs 的 stackedWidget
    

`# main.py (或你的主視窗類別初始化後) from PySide6.QtCore import QPropertyAnimation, QEasingCurve from PySide6.QtWidgets import QButtonGroup  class MainWindow(QMainWindow):     def __init__(self):         super().__init__()         self.ui = Ui_MainWindow()         self.ui.setupUi(self)          # sidebar 狀態         self.menu_expanded_width = 200         self.menu_collapsed_width = 60         self.is_menu_collapsed = False          # 1) 左側縮放         self.ui.btn_toggle_menu.clicked.connect(self.toggle_left_menu)          # 2) u1/u2 -> u3         self.ui.btn_template_next.clicked.connect(self.go_to_login_page)         # 也可以：self.ui.lineEdit_template.returnPressed.connect(self.go_to_login_page)          # 3) u3 -> u4         self.ui.btn_login.clicked.connect(self.do_login)          # 4) u4 上方 tabs（按鈕列 + stackedWidget）         self.tabs_group = QButtonGroup(self)         self.tabs_group.setExclusive(True)         self.tabs_group.addButton(self.ui.btn_tab_front, 0)         self.tabs_group.addButton(self.ui.btn_tab_back, 1)         self.tabs_group.addButton(self.ui.btn_tab_open_back, 2)         self.tabs_group.addButton(self.ui.btn_tab_open_back_crown, 3)         self.tabs_group.addButton(self.ui.btn_tab_strap_right, 4)         self.tabs_group.addButton(self.ui.btn_tab_material, 5)         self.tabs_group.idClicked.connect(self.switch_run_tab)          # 預設顯示 u1         self.ui.stackedWidget.setCurrentWidget(self.ui.page_template)      def toggle_left_menu(self):         # 目標寬度         start = self.ui.frame_left_menu.width()         end = self.menu_collapsed_width if not self.is_menu_collapsed else self.menu_expanded_width          # 顯示哪一套 menu（expanded / compact）         collapsing = (end == self.menu_collapsed_width)         self.ui.widget_menu_expanded.setVisible(not collapsing)         self.ui.widget_menu_compact.setVisible(collapsing)          # 動畫         self.anim = QPropertyAnimation(self.ui.frame_left_menu, b"minimumWidth")         self.anim.setDuration(220)         self.anim.setStartValue(start)         self.anim.setEndValue(end)         self.anim.setEasingCurve(QEasingCurve.InOutCubic)         self.anim.start()          self.is_menu_collapsed = collapsing      def go_to_login_page(self):         template_id = self.ui.lineEdit_template.text().strip()         if not template_id:             # 你可以彈 QMessageBox             return         self.ui.stackedWidget.setCurrentWidget(self.ui.page_login)      def do_login(self):         account = self.ui.lineEdit_account.text().strip()         password = self.ui.lineEdit_password.text().strip()          # TODO: 實際驗證邏輯（先用簡單判斷）         if not account or not password:             return          self.ui.stackedWidget.setCurrentWidget(self.ui.page_run)      def switch_run_tab(self, tab_id: int):         self.ui.stackedWidget_run_tabs.setCurrentIndex(tab_id)`

> 這段程式碼的核心就是：
> 
> - sidebar：用 QPropertyAnimation 動最小寬度（常見做法）
>     
> - 換頁：用 stackedWidget setCurrentWidget/setCurrentIndex（QStackedWidget 標準用法）
>     

### 4-2. 你在 Designer 必須配合的 objectName

你需要在 Designer 裡把這些 objectName 設好（名字你可以改，但 Python 要一致）：

**Sidebar**

- `frame_left_menu`
    
- `btn_toggle_menu`
    
- `widget_menu_expanded`
    
- `widget_menu_compact`
    

**u1/u2**

- `page_template`
    
- `lineEdit_template`
    
- `btn_template_next`
    

**u3**

- `page_login`
    
- `lineEdit_account`
    
- `lineEdit_password`
    
- `btn_login`
    

**u4**

- `page_run`
    
- `stackedWidget_run_tabs`
    
- `btn_tab_front`, `btn_tab_back`, ...
    

---

## 5) 顏色/主題（讓它長得像你圖的淺藍 + 藍色 sidebar）

PyDracula 的主題就是 `.qss`，放在 `themes/`，README 明講可以放多主題。  
Qt 官方也建議整體 styling 用 QSS（比逐個 widget 設定省事）。

### 5-1. 建一個你的主題檔

在 `themes/` 新增 `light_blue.qss`：

`QWidget {     background: #AFCFE8;     color: #0b0b0b;     font-size: 18px; }  QFrame#frame_left_menu {     background: #36A9D4; }  QPushButton {     background: transparent;     border: none;     color: white;     padding: 14px 18px;     text-align: left; }  QPushButton:hover {     background: rgba(255,255,255,0.18); }  QLineEdit {     background: white;     border: 2px solid rgba(0,0,0,0.2);     padding: 8px 10px; }`

### 5-2. 在程式啟動時載入 QSS

你在 `main.py`（初始化後）讀檔套用：

`def apply_theme(self, qss_path: str):     with open(qss_path, "r", encoding="utf-8") as f:         self.setStyleSheet(f.read())  # __init__ 末尾呼叫 self.apply_theme("themes/light_blue.qss")`

---

## 6) 每次改 main.ui 記得「重新輸出 ui_main.py」

PyDracula README 有給指令： `pyside6-uic main.ui > ui_main.py`  
不做這步，你 Designer 改半天，程式可能還在用舊 UI。

`pyside6-uic main.ui > modules/ui_main.py`

---

## 7) 你問的「PyDracula 怎麼做到？」一句話總結

- **u1 ⇄ u2（sidebar 縮放）**  
    = `QPropertyAnimation` 去 animate sidebar frame 的寬度 + 顯示/隱藏 expanded/compact menu
    
- **u1/u2 → u3 → u4（流程進到下一介面）**  
    = `QStackedWidget` 的 `setCurrentWidget()` / `setCurrentIndex()` 切 page
    
- **u4 上方那排 tabs**  
    = 另一個 `QStackedWidget` + `QButtonGroup` 互斥按鈕，或用 `QTabWidget`（較省事）
    

---

## 你如果要我更「對位」到你現在的 PyDracula 專案（最快成功的方法）

你把你目前專案的這兩個東西貼上來（不用全 repo）我就能把上面步驟「變成完全對應你檔案/元件名稱的版本」：

1. 你現在的 `main.ui`（或截圖 Object Inspector 樹狀結構也可以）
    
2. 你現在 `main.py` 裡面初始化 UI（`Ui_MainWindow().setupUi()`）那一段
    

我會直接告訴你：

- 哪些 objectName 應該叫什麼（跟 PyDracula 原本結構相容）
    
- 事件該寫在哪個檔（ui_functions / app_functions / main）
    
- sidebar toggle 應該動畫哪個 frame（避免你 animate 到錯的容器導致 layout 崩掉）

