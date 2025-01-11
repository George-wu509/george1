

| IMAGE DATA            |                                         |
| --------------------- | --------------------------------------- |
| ==imageDatastore==    | 專門用於處理大規模圖像數據集，能夠逐步讀取和處理圖像文件。           |
| ==Blocked Images==    | 是一種處理大規模影像的技術，它通過將影像分成更小的區塊來進行讀寫和處理     |
| Bigimage              | 專門用來處理超大規模影像的數據類型類似於 blockedImage       |
| imref2d,  imref3d     | 這些類別提供對影像參考系的支持                         |
| ==niftiread==         | 用於讀取 NIfTI 格式的醫學影像數據文件                  |
| ==dicomread==         | 讀取 DICOM 格式的醫學影像，這種格式通常用於放射影像           |
| **DATA STORE**        |                                         |
| fileDatastore         | 用於讀取任意大型二進制文件，提供逐步讀取和處理文件內容的功能。         |
| tabularTextDatastore  | 用於處理大規模的表格數據文件，如 CSV 或文本文件，逐步讀取和處理每行數據。 |
| binaryFileDatastore   | 針對二進制數據文件進行處理，適合自定義格式的大數據。              |
| **OTHER DATA**        |                                         |
| MapReduce             | 是一個分佈式計算框架，允許你在多個節點上並行處理大數據。            |
| Large MAT-Files       | 大型 MAT 文件使用 `-v7.3` 格式存儲，支持超過 2 GB 的數據  |
| Parquet Files Parquet | 是一種專為大數據分析而設計的列式存儲格式。它能有效壓縮數據並支持快速讀寫    |
| ==Memory Mapping==    | 將大型文件映射到內存中，使得你可以像操作內存中的數組一樣訪問大文件中的數據   |
| ==Tall Arrays==       | 用於處理超出內存限制的數據。它可以表示非常大的數據集              |

ref: [Data Import and Export](https://www.mathworks.com/help/matlab/data-import-and-export.html?s_tid=CRUX_lftnav)


在 MATLAB 中，處理大數據或大規模影像的情況下，常用的數據類型和技術包括 Datastore、Tall Arrays、MapReduce、大型 MAT 文件、Parquet 文件、內存映射等。這些工具幫助你有效地處理大數據或影像數據，而不會耗盡內存。接下來，我會詳細介紹這些技術及其適用場景，並提供相應的示例代碼。

### 1. <mark style="background: #FF5582A6;">imageDatastore (數據存儲)</mark>

#### 簡介：

`Datastore` 是一個用於處理大量數據的工具。它允許你逐步從文件中讀取數據，而不需要一次性將整個數據集加載到內存中。它特別適合處理大數據集，如圖像集、大型文本文件或分佈式文件系統中的數據。datastore包含<mark style="background: #FFB86CA6;">ImageDatastore</mark>(imread圖像文件), TallDatastore(tall點數組), TabularTextDatastore(CSV 文件), SpreadsheetDatastore(Excel®格式), KeyValueDatastore(鍵值對資料 mapreduce), FileDatastore(自訂文件格式), ParquetDatastore(Parquet 檔案)

#### 適用場景：

- 處理大型圖像數據集、CSV 文件、文本文件或分佈式數據。
- 逐步讀取數據，適合處理無法一次性載入內存的數據集。

% 創建一個用於處理圖像文件的 Datastore
ds = <mark style="background: #FFB86CA6;">imageDatastore</mark>('path_to_large_image_folder', 'FileExtensions', '.jpg');

% 逐步讀取和處理影像
while <mark style="background: #FFB86CA6;">hasdata</mark>(ds)
    img = read(ds); % 讀取下一張影像
    imshow(img); % 顯示影像
    % 這裡可以添加影像處理邏輯
end

### 2. <mark style="background: #FF5582A6;">Blocked Images </mark>

#### 什麼是 Blocked Images？

`Blocked Images` 是一種處理大規模影像的技術，它通過將影像分成更小的區塊來進行讀寫和處理，這樣可以有效減少內存的佔用。這些小區塊稱為“塊（block）”。MATLAB 使用 `blockedImage` 類來表示這類影像數據結構，並提供多解析度（multi-resolution）的處理能力，這對於需要進行大範圍縮放或逐級細化處理的應用非常有用。

#### 適用場景：

- **超大影像：** 當影像尺寸非常大時，無法一次性將整個影像載入內存，此時可以使用 Blocked Images 進行分塊處理。
- **多解析度影像處理：** 適合需要在不同解析度下進行影像分析或處理的情況，尤其是醫學影像或顯微影像。
- **分佈式處理：** 當你需要將大影像數據分佈在多個處理節點時，可以將影像分成塊進行分佈式計算。

### Blocked Images 常用函數

##### blockedImage()
用途： 創建一個 blockedImage 對象，將影像存儲為分塊的格式，可以來自不同的影像文件格式（如 TIFF、PNG 等）。
使用場景： 當你有一個超大影像文件，需要進行分塊處理時使用。
bim = <mark style="background: #FFB86CA6;">blockedImage</mark>('large_image.tiff');

##### apply()
用途： 在 blockedImage 的每個塊上應用指定的函數。這是一個重要的函數，用來實現針對每個塊的分佈式操作。
使用場景： 當你需要對大影像的每一塊進行同樣的操作，如圖像過濾、邊緣檢測等。
edgeBim = apply(bim, @(block) edge(block.Data, 'sobel'));

##### getBlock()
用途： 獲取指定位置的單個塊，用於檢查或處理特定區域的數據。
使用場景： 當你只想處理或檢查大影像中的某個特定區域時。
block = <mark style="background: #FFB86CA6;">getBlock</mark>(bim, [1, 1]);

##### regionApply()
用途： 在 blockedImage 的特定區域（如 ROI）應用指定的函數。
使用場景： 當你只需要處理影像的某個特定區域時，可以使用這個函數。
roi = [100, 100, 200, 200];  % [X, Y, Width, Height]
regionBim = regionApply(bim, @(block) imadjust(block.Data), roi);

##### show()
用途： 顯示 blockedImage 的多解析度影像。這允許你快速瀏覽影像的不同解析度層次。
使用場景： 當你需要檢視影像的全貌並在不同解析度之間切換時使用。
show(bim);

##### transform()
用途： 對 blockedImage 進行全局性轉換操作，例如旋轉、翻轉等操作，這些操作會應用到每一個塊。
使用場景： 當你需要對整個大影像進行某種變換（如旋轉或幾何變換）時。
rotatedBim = transform(bim, @imrotate, 90);

##### write()
用途： 將處理後的 blockedImage 寫入文件，支持 TIFF 等格式。
使用場景： 當你處理完大影像後需要將其保存為文件時使用。
write(rotatedBim, 'output_image.tiff');

##### blockLocation()
用途： 獲取某個塊的位置信息，如其在影像中的位置和尺寸。
使用場景： 當你需要了解特定塊的位置信息以進行更精細的處理時。
loc = blockLocation(bim, [1, 1]);
disp(loc);


### 2. <mark style="background: #FF5582A6;">Tall Arrays (大陣列)</mark>

#### 簡介：

`Tall Arrays` 用於處理超出內存限制的數據。它可以表示非常大的數據集，並允許進行像標準 MATLAB 陣列那樣的操作。背後使用了懶加載技術，計算只有在真正需要時才會被執行。

#### 適用場景：

- 處理比內存大得多的數據，例如大數據表格、數值矩陣。
- 進行數據分析、數據科學操作和數據轉換。

% 創建一個 tall array 來處理大型 CSV 文件
ds = datastore('large_data.csv');
tt = <mark style="background: #FFB86CA6;">tall</mark>(ds);

% 執行一些操作，這些操作將是懶加載的
meanValue = mean(tt.Var1);

% 將結果收集回內存
gather(meanValue)

`MATLAB` 中的 `tall` 函數和 `tall` 陣列主要是為了處理超過記憶體大小的數據集（即大數據集）。你遇到的「Starting parallel pool (parpool) using the 'local' profile」訊息，與 MATLAB 背後運行 `tall` 陣列時啟動的並行計算有關。讓我逐步解釋：

### `tall` 函數與 `tall` 陣列

1. **`tall` 陣列簡介**：
    
    - `tall` 陣列是一種特別設計來處理非常大的數據集的資料型別，這些數據集可能無法完全載入記憶體。
    - 它的特點是**逐塊處理**數據，並且使用 MATLAB 的並行計算功能來優化大數據的操作和計算。
    - 與常規陣列不同，`tall` 陣列的計算是延遲執行的，也就是說，當你執行對 `tall` 陣列的操作時，這些操作不會立即進行，而是被記錄下來，直到需要的時候（比如呼叫 `gather` 函數來獲取結果時）才進行計算。
2. **`tall` 函數**：
    
    - `tall(ds)` 是用來將一個資料存儲對象（如 `datastore`）轉換為 `tall` 陣列。`datastore` 代表一個外部的資料集（如檔案、數據庫、雲存儲等），而 `tall` 函數允許你以分塊方式操作這些超大數據。
    - 一般用於讀取大文件或表格數據，例如 `.xlsx` 文件。
3. **並行計算與並行池（Parallel Pool）**：
    
    - 當你在 MATLAB 中使用 `tall` 陣列時，為了提高效率，MATLAB 會自動使用並行計算來處理數據（這通常是針對多核處理器的計算）。
    - 在你執行 `tall(ds)` 時，MATLAB 會檢查是否有並行計算池（Parallel Pool）可用。如果沒有，它會自動啟動一個並行池，這就是為什麼你會看到「Starting parallel pool (parpool) using the 'local' profile」的訊息。
    - 並行池是 MATLAB 背後的工作者進程，它們會分攤計算任務，以加速數據處理，尤其是處理大型數據集的時候。

### `tall` 陣列的工作原理

1. **分塊處理**：`tall` 陣列的數據並不全部存儲在記憶體中，而是分成多個塊來處理。這樣做是為了避免一次性佔用過多記憶體，並讓 MATLAB 可以處理大數據集。
    
2. **延遲執行**：當你對 `tall` 陣列進行操作時，這些操作會被「記錄」下來，並不會立即執行。這是為了優化整個數據處理流程，最終在調用 `gather` 或類似函數時才會實際計算。
    
3. **多核加速**：通過並行計算，MATLAB 可以同時處理多個數據塊，這可以顯著減少處理時間。
    

### 總結

- `tall` 陣列允許你處理大於記憶體的數據，透過分塊處理、延遲執行以及並行計算來實現高效數據處理。
- 啟動並行池的原因是 MATLAB 嘗試在多核上分擔運算工作，以加速大數據的計算。
- 在大數據處理工作流中，`tall` 陣列是一個非常有用的工具，特別是當你的數據量很大並且記憶體不足以一次性載入所有數據時。

`MATLAB` 的 `tall` 函數最初是為了處理存儲在 `datastore` 中的超大數據集設計的，所以它的典型應用是與 `datastore` 配合使用。然而，它並不僅限於特定的數據類型。理論上，如果你的數據可以以 `datastore` 形式封裝或表示，你可以將其轉換為 `tall` 陣列來進行操作。

### `tall` 函數的應用範圍

1. **主要應用在 `datastore` 上**：`tall` 函數的設計是用來處理存儲在 `datastore` 中的數據，因為 `datastore` 是 MATLAB 的大數據架構，允許逐步讀取超大數據而不需要一次性載入整個數據集到記憶體中。
    
2. **Blocked Images 是否適用？**：
    
    - `Blocked Images` 是 MATLAB 為處理非常大的圖像（通常是多分辨率或多切片圖像）而設計的一個框架。`Blocked Images` 使用的機制與 `tall` 函數相似，因為它們都依賴於 **Lazy Evaluation**（延遲求值）來進行大數據的處理。
    - 不過，`Blocked Images` 是專門為圖像設計的，因此它有針對性地提供了一些與圖像處理有關的特定函數和功能。雖然 `Blocked Images` 本身不會直接使用 `tall` 函數，但它們在處理大數據時的原理有些相似。

### Lazy Evaluation 機制的聯繫

- **Lazy Evaluation**（延遲求值）：無論是 `tall` 陣列還是 `Blocked Images`，它們都依賴於 Lazy Evaluation 機制來處理大量數據。這意味著操作（如計算、轉換等）並不會立即執行，而是等到需要最終結果（如輸出或保存時）才會觸發計算。這有助於節省記憶體資源並優化性能。
    
- **兩者的不同應用場景**：
    
    - **`tall` 陣列**：更廣泛地應用於大數據分析，包括表格數據、數據庫、文本文件、時間序列數據等。
    - **Blocked Images**：專注於超大圖像的處理，尤其是那些需要多分辨率表示或切片操作的圖像。

### 兩者的關聯

儘管 `tall` 陣列和 `Blocked Images` 在運行機制上共享 Lazy Evaluation 的特點，它們的應用場景不同。`tall` 更偏向於數據分析和處理，而 `Blocked Images` 專門處理超大圖像。這兩個工具在 MATLAB 大數據處理框架下共同為不同類型的大數據提供了解決方案，但在底層實現上它們並沒有直接的關聯。

總結來說，`tall` 函數主要用於 `datastore` 類型的數據，而 `Blocked Images` 則是專門為處理大圖像設計的。它們都利用了 Lazy Evaluation 機制，但應用場景和處理方式不同。

### 3. <mark style="background: #FF5582A6;">MapReduce</mark>

#### 簡介：

`MapReduce` 是一個分佈式計算框架，允許你在多個節點上並行處理大數據。它將工作分成兩個階段：`map` 階段處理並過濾數據，`reduce` 階段總結和聚合數據。`MapReduce` 特別適合大規模分佈式數據處理。

#### 適用場景：

- 大規模分佈式數據集，例如來自多個文件的日誌數據或數據庫。
- 必須在分佈式系統上執行的數據處理和聚合。

% 創建 datastore
ds = datastore('big_data.csv', 'TreatAsMissing', 'NA');

% 定義 map 函數
function mapFunc(data, info, intermKVStore)
    % 計算每行的平均值並保存中間結果
    avg = mean(data.Var1);
    add(intermKVStore, 'Avg', avg);
end

% 定義 reduce 函數
function reduceFunc(intermKey, intermValIter, outKVStore)
    % 聚合所有中間結果
    totalSum = 0;
    count = 0;
    while hasnext(intermValIter)
        totalSum = totalSum + getnext(intermValIter);
        count = count + 1;
    end
    add(outKVStore, 'TotalAvg', totalSum / count);
end

% 使用 MapReduce
result = <mark style="background: #FFB86CA6;">mapreduce</mark>(ds, @mapFunc, @reduceFunc);

`MATLAB` 的 `reduce` 函數其實與 MapReduce 編程模型有關，這是一種專門設計用來處理大規模數據的分佈式運算模式。`reduce` 主要的功能是將已經經過初步處理的數據進行合併，適用於大數據環境中逐塊處理結果的聚合。讓我詳細解釋 `reduce` 的原理及其在圖像數據處理中的應用潛力。

### `reduce` 函數原理

1. **核心概念**：
    
    - `reduce` 這個詞源自 MapReduce 架構，它本質上是將中間結果聚合的過程。`reduce` 函數用於將多個計算結果合併成一個結果，這些結果可能來自於大數據集的不同部分或區塊。
    - 在 MapReduce 的過程中，數據通常會先通過 `map` 函數進行初步處理（如篩選、轉換等），然後 `reduce` 函數會將這些分散的結果匯總，最終得到一個單一的輸出。
2. **Reduce 的工作流程**：
    
    - 首先，數據被分成多個塊或部分。
    - 對每個塊使用 `map` 操作進行初步處理。
    - 之後將所有這些處理過的塊通過 `reduce` 函數進行匯總。這種聚合的方式可以是多種多樣的，例如：累加、計數、取平均值、拼接等。

### `reduce` 在圖像處理中的應用

1. **圖像數據的處理**：
    
    - 雖然 `reduce` 函數不直接針對圖像數據處理而設計，但可以在一些特定的圖像處理任務中使用。比如：
        - **圖像的分塊處理**：如果你需要對超大圖像進行處理，你可以先將圖像分成多個小塊，然後對每個小塊進行處理（類似於 `map` 階段），最後使用 `reduce` 函數來將結果進行合併（例如，合成完整的圖像，或累計像素強度、統計信息等）。
        - **特徵聚合**：在多圖像的處理流程中，可以對每個圖像提取特徵，然後使用 `reduce` 函數來合併這些特徵，形成最終的特徵向量，用於後續的分類或識別。
2. **處理超大圖像**：
    
    - 在處理超大圖像（如醫學影像、衛星影像）時，整張圖像可能無法一次性載入到記憶體中。這時可以將圖像切分成若干塊進行分塊處理，並且在完成每個塊的處理之後，通過 `reduce` 函數來進行結果的合併。
    - `Blocked Images` 也是一種處理超大圖像的方式，這裡也可以類似使用 `reduce` 機制來整合每個塊的處理結果。

### `reduce` 的優點

- **可擴展性**：`reduce` 可以處理非常大的數據集，因為它不需要一次性將整個數據集載入記憶體，而是通過分塊和聚合來處理數據。
- **靈活性**：`reduce` 可以用來進行多種聚合操作，如累加、計算最大值或最小值、統計分佈等。

### 總結

- `reduce` 函數主要用於將多個處理結果合併為一個結果，這在大數據環境中特別有用。
- 雖然它不專門針對圖像處理，但可以靈活應用於超大圖像的分塊處理、特徵提取、以及圖像數據的聚合上。
- 在處理大數據或超大圖像時，`reduce` 可以作為一個有效的工具來整合分塊處理的結果。

希望這樣的解釋能幫助你了解 `reduce` 在 MATLAB 及其在圖像處理中的潛在應用。

### 4. <mark style="background: #FF5582A6;">Large MAT-Files (大型 MAT 文件)</mark>

#### 簡介：

MATLAB 的 MAT 文件可以存儲大量數據。大型 MAT 文件使用 `-v7.3` 格式存儲，支持超過 2 GB 的數據。它是處理大數據的有效方式，因為數據存儲在磁盤上，而不會佔用內存。

#### 適用場景：

- 存儲和讀取大規模數據集，如圖像數據或大矩陣。
- 需要多次存取的持久性大數據

% 將大矩陣保存到 MAT 文件中
largeData = rand(1e5, 1e4);
save('large_data.mat', 'largeData', '-v7.3');

% 從 MAT 文件中讀取數據
loadedData = matfile('large_data.mat');
subsetData = loadedData.largeData(1:1000, :);

### 5. <mark style="background: #FF5582A6;">Parquet Files (Parquet 文件)</mark>

#### 簡介：

Parquet 是一種專為大數據分析而設計的列式存儲格式。它能有效壓縮數據並支持快速讀寫。MATLAB 可以讀取和寫入 Parquet 文件，非常適合處理來自大數據系統的數據。

#### 適用場景：

- 大數據系統中常見的數據交換格式。
- 高效讀寫和分析大數據集。


% 寫入 Parquet 文件
data = table(rand(1000, 1), randi(100, 1000, 1), 'VariableNames', {'Var1', 'Var2'});
parquetwrite('large_data.parquet', data);

% 從 Parquet 文件讀取數據
tbl = parquetread('large_data.parquet');

### 6. <mark style="background: #FF5582A6;">Memory Mapping (內存映射)</mark>

#### 簡介：

`Memory Mapping` 將大型文件映射到內存中，使得你可以像操作內存中的數組一樣訪問大文件中的數據。這非常適合處理超過內存限制的大文件，而不需要將文件一次性加載到內存中。

#### 適用場景：

- 操作非常大的二進制文件，如圖像數據或測量數據。
- 需要快速隨機訪問大文件的應用。

% 創建一個二進制文件
fid = fopen('large_binary_file.dat', 'w');
fwrite(fid, rand(1e6, 1), 'double');
fclose(fid);

% 將文件映射到內存中
m = <mark style="background: #FFB86CA6;">memmapfile</mark>('large_binary_file.dat', 'Format', 'double');

% 訪問映射文件中的數據
data = m.Data(1:1000);  % 訪問前 1000 個數據


在 MATLAB 中，`Blocked Images` 是用來處理超大規模影像的數據結構，特別適合處理無法一次性載入內存的影像數據。這種數據結構將影像分成若干塊進行處理，並支持多解析度影像的操作，非常適合於醫學影像、遙感影像等需要高效存儲和多解析度處理的應用。

`MATLAB` 的 memory mapping (記憶體映射) 技術是一種有效的方式來處理非常大的數據文件，比如超大型圖像數據，當這些數據無法完全載入到記憶體中時，使用記憶體映射能夠提供一種靈活且高效的解決方案。

### 記憶體映射 (Memory Mapping) 的原理

1. **核心概念**：
    
    - 記憶體映射允許將一個大文件的一部分直接映射到內存中，而不是將整個文件一次性讀入內存。這種技術使得可以以文件為基礎進行數據處理，而不需要消耗大量內存。
    - 當你使用 memory mapping 時，數據並不會被完全讀入內存，而是根據需要動態地將文件中的特定部分加載到內存中，進行處理後再釋放。
2. **工作原理**：
    
    - Memory mapping 的核心是在不將文件全部載入到內存中的情況下，通過操作系統將文件的一部分映射到進程的虛擬內存空間中。
    - 當 MATLAB 程序訪問映射區域時，操作系統會自動從磁盤讀取相應的數據進入內存，並進行所需的處理。
    - 如果進程不再使用這些數據，或者需要新的數據，映射的區域可以被替換，從而實現對大數據的有效處理。
3. **效益**：
    
    - 記憶體映射提供了一種處理大數據的靈活方式，它能夠減少內存的使用，並且在處理特定區域的數據時不需要每次都進行文件的重新讀寫，從而加快了數據處理的速度。

### 在 MATLAB 圖像處理中的應用

1. **應用場景**：
    
    - **超大圖像處理**：當處理非常大的圖像文件時，特別是在科學計算、醫學成像和遙感數據等應用中，圖像文件可能非常大，無法一次性載入內存。此時，可以使用 memory mapping 來逐塊處理這些圖像數據。
    - **圖像數據的逐塊處理**：記憶體映射允許你對大圖像進行分塊處理，這與 Blocked Images 或 tall arrays 處理大數據的方式類似。記憶體映射的優勢在於它提供了一種高效的機制來讀取和操作大型二進制圖像數據文件。
    - 

### 適用於大數據或大影像的其他數據類型

1. **`imageDatastore`**: 專門用於處理大規模圖像數據集，能夠逐步讀取和處理圖像文件。
2. **`fileDatastore`**: 用於讀取任意大型二進制文件，提供逐步讀取和處理文件內容的功能。
3. **`tabularTextDatastore`**: 用於處理大規模的表格數據文件，如 CSV 或文本文件，逐步讀取和處理每行數據。
4. **`binaryFileDatastore`**: 針對二進制數據文件進行處理，適合自定義格式的大數據。

### 其他適用於大影像和多解析度影像的數據類型

除了 `blockedImage` 外，MATLAB 還提供了幾種適合大影像和多解析度影像處理的數據類型：

##### <mark style="background: #BBFABBA6;">imageDatastore</mark>:
用途： 用於處理大型圖像數據集的 Datastore。它可以逐步讀取和處理大量圖像文件，適合影像分類、訓練神經網絡等應用。
imds = <mark style="background: #FFB86CA6;">imageDatastore</mark>('path_to_large_image_folder');

##### <mark style="background: #BBFABBA6;">bigimage</mark>:  (The bigimage object will be removed in a future release. Use the blockedImage object instead. For more information)
用途： 專門用來處理超大規模影像的數據類型，允許進行影像切片處理和多解析度處理，類似於 blockedImage，但更適合處理醫學影像或超高解析度的顯微影像。
% 創建一個大影像對象
bim = <mark style="background: #FFB86CA6;">bigimage</mark>('large_image.tiff');

##### <mark style="background: #BBFABBA6;">imref2d</mark> 和 <mark style="background: #BBFABBA6;">imref3d</mark>:
用途： 這些類別提供對影像參考系的支持，適合處理具有空間坐標和多分辨率的影像數據。
% 創建2D影像參考系
R = imref2d([1024, 1024], [-1 1], [-1 1]);

##### <mark style="background: #BBFABBA6;">niftiread</mark>:
用途： 用於讀取 NIfTI 格式的醫學影像數據文件，這種格式通常用於 MRI 或 CT 影像。
% 讀取 NIfTI 醫學影像
niftiImage = niftiread('brain_scan.nii');

##### <mark style="background: #BBFABBA6;">dicomread</mark>:
用途： 讀取 DICOM 格式的醫學影像，這種格式通常用於放射影像（如 X 光片）。
% 讀取 DICOM 影像
dicomImage = dicomread('xray.dcm');

總結
Blocked Images 是 MATLAB 中非常強大的工具，適合處理超大規模的影像數據，特別是在醫學影像和顯微影像中應用廣泛。通過將影像分塊處理，blockedImage 能夠在內存限制下高效處理大影像。同時，MATLAB 也提供了其他多種處理大影像的數據類型和函數，如 imageDatastore、bigimage 等，這些工具共同提供了靈活且高效的大數據影像處理能力。

在醫學圖像處理中，常見的 2D 和 3D 圖像格式包括 DICOM 和 NIfTI，它們是處理醫學影像數據的標準格式。以下是它們的詳細解釋及在 MATLAB 中的使用方法和常用函數。

### 1. **DICOM (Digital Imaging and Communications in Medicine)**

**概述**：

- DICOM 是醫學成像中最廣泛使用的格式之一，主要用於存儲和傳輸 CT、MRI、X 射線等醫學影像數據。
- 除了圖像數據本身，DICOM 文件還包含大量的元數據（例如病人資訊、成像參數、掃描日期等），這些信息對臨床應用和圖像處理非常重要。

**MATLAB 中使用 DICOM**： MATLAB 提供了內建的工具箱來讀取、顯示和處理 DICOM 文件。常用的 DICOM 函數如下：

- **`dicomread`**：讀取 DICOM 文件。
    
    matlab
    
    複製程式碼
    
    `img = dicomread('image.dcm');`
    
- **`dicominfo`**：提取 DICOM 文件的元數據信息。
    
    matlab
    
    複製程式碼
    
    `info = dicominfo('image.dcm');`
    
- **`imshow`** 和 **`montage`**：顯示 DICOM 圖像。
    
    matlab
    
    複製程式碼
    
    `imshow(img, []);`
    
- **`dicomwrite`**：將數據保存為 DICOM 文件。
    
    matlab
    
    複製程式碼
    
    `dicomwrite(img, 'output.dcm', info);`
    
- **`dicomdict`**：設置或讀取 DICOM 字典（標準標籤）。
    

**應用場景**： DICOM 主要用於醫療診斷影像的處理和存儲。它提供了詳細的圖像元數據，可以進行影像的診斷分析與臨床資料整合。

### 2. **NIfTI (Neuroimaging Informatics Technology Initiative)**

**概述**：

- NIfTI 是一種專門用於神經影像（例如 MRI 和功能磁共振成像 fMRI）的格式，主要用於 3D 或 4D 醫學影像數據。
- NIfTI 文件的擴展名通常為 `.nii` 或 `.nii.gz`，後者為壓縮格式。
- NIfTI 文件格式相對於 DICOM 更加簡單，沒有那麼多的標籤和元數據，主要關注圖像數據本身和一些基本的空間信息。

**MATLAB 中使用 NIfTI**： MATLAB 對 NIfTI 格式的支持通常依賴於附加的工具包（如 `niftitools` 或 `spm` 工具箱）。使用 NIfTI 工具包可以讀取、寫入和處理 NIfTI 文件。

- **`niftiread`**：讀取 NIfTI 文件。
    
    matlab
    
    複製程式碼
    
    `img = niftiread('image.nii');`
    
- **`niftiinfo`**：獲取 NIfTI 文件的元數據。
    
    matlab
    
    複製程式碼
    
    `info = niftiinfo('image.nii');`
    
- **`niftiwrite`**：將數據寫入 NIfTI 文件。
    
    matlab
    
    複製程式碼
    
    `niftiwrite(img, 'output.nii');`
    
- **`volshow`**：顯示 3D 體積數據。
    
    matlab
    
    複製程式碼
    
    `volshow(img);`
    

**應用場景**： NIfTI 主要用於神經影像學研究，尤其在腦科學中非常流行，例如處理 fMRI、DTI（擴散張量成像）數據時，NIfTI 是標準格式。

### DICOM 和 NIfTI 的比較

|**特徵**|**DICOM**|**NIfTI**|
|---|---|---|
|**應用領域**|廣泛的醫學影像（CT、MRI、X光等）|主要用於神經影像（MRI、fMRI、DTI等）|
|**文件結構**|複雜，包含詳細的病患資訊和圖像元數據|簡單，專注於圖像和空間信息|
|**文件格式**|每個切片為一個單獨的文件或多切片組合|整個影像數據存儲在一個單一的文件中|
|**維度支持**|2D 和 3D 圖像|3D 和 4D 圖像|
|**MATLAB 支持**|原生支持|通常需要第三方工具或工具箱|
|**典型應用**|臨床醫學診斷影像處理|神經影像學研究|

### 結論

- **DICOM** 適用於廣泛的醫學影像，特別是臨床環境中的診斷應用，它的強大之處在於豐富的元數據記錄。
- **NIfTI** 則更適合神經科學的研究，特別是處理 3D 或 4D 的 MRI 或功能成像數據。它的格式簡單，更加適合數據分析和研究用途。

兩者在 MATLAB 中都有良好的支持，不過 DICOM 具有更強的內建支持，而 NIfTI 通常需要附加的工具來進行處理。

DICOM = header + data element + data element ....

header: 
   preamble(128b) - abstract of DICOM
   prefix(4b) - t/f DICOM format

data element:
   Data element tag(Tag,4b) - 2000多個tag標示element數據內容
   values representations(VR,2b) - 27種VR, data format
   value length(2b) - 
   value field() - 

mainbody
   patient info
   study info
   series info
   equipment info
   image info


