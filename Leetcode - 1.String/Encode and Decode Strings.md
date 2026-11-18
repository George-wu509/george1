
LeetCode 271 / LintCode 659 請設計一個演算法，將一個字串陣列（`List[str]`）編碼成單一字串（`str`）。編碼後的字串會透過網路傳送，並解碼還原回原本的字串陣列。

請實作 `encode` 與 `decode` 兩個函式，且**不可使用內建的序列化函式（如 `eval`、`json`、`pickle`）**，演算法必須是無狀態（Stateless）的，且字串中可能包含任意 ASCII 字元（包括空格、數字、`#`、換行符號等）甚至空字串 `""`。

**样例 1：** 输入：

```Python
["lint", "code", "love", "you"]
```

输出：

```Python
["lint", "code", "love", "you"]
```

解释： 一種可能的編碼格式為 `"4#lint4#code4#love3#you"`，解碼後必須完全還原為原陣列。

**样例 2：** 输入：
```Python
["we", "say", ":", "yes", "!@#$%^&*()"]
```

输出：
```Python
["we", "say", ":", "yes", "!@#$%^&*()"]
```

解释： 即使字串中包含特殊符號（如 `#`、`:`）或數字，解碼器仍能準確切分出 5 個獨立字串。

**样例 3：** 输入：
```Python
[""]
```

输出：
```Python
[""]
```

解释： 空字串編碼為 `"0#"`，解碼時讀取長度為 `0`，還原為包含一個空字串的陣列 `[""]`。


```Python
class Codec:
    def encode(self, strs: list[str]) -> str:
        res = []
        for s in strs:
            res.append(f"{len(s)}#{s}")
        return "".join(res)

    def decode(self, s: str) -> list[str]:
        res = []
        i = 0
        while i < len(s):
            j = i
            while s[j] != "#":
                j += 1
            length = int(s[i:j])
            res.append(s[j + 1 : j + 1 + length])
            i = j + 1 + length
        return res
```


```Python
# 另一種寫法：使用 str.find() 定位分隔符 '#'，可讀性更高
class Codec:
    def encode(self, strs: list[str]) -> str:
        return "".join(f"{len(s)}#{s}" for s in strs)

    def decode(self, s: str) -> list[str]:
        res = []
        i = 0
        while i < len(s):
            j = s.find("#", i)
            length = int(s[i:j])
            start = j + 1
            end = start + length
            res.append(s[start:end])
            i = end
        return res
```

解釋: 
step1: 核心協議為 `[字串長度] + [#] + [原始字串]`（類似 HTTP Chunked Transfer Encoding） 
step2: 編碼時，將每個字串的長度轉為字串並加上 `#` 作為標頭（Header），例如 `["ab", "3#c"]` 會被編碼為 `"2#ab3#3#c"` step3: 解碼時，指標 `i` 永遠停在「下一個字串長度數字」的起點，一開始 `i = 0` 
step4: 第一輪迴圈：`j` 從 `0` 往右找到第一個 `#`（在索引 `1`），取出長度 `int(s[0:1]) = 2` 
step5: 直接從 `j + 1`（索引 `2`）往後擷取 `2` 個字元，得到 `"ab"`，並將 `i` 直接跳轉到 `2 + 2 = 4` 
step6: 第二輪迴圈：`i = 4`，`j` 找到索引 `5` 的 `#`，取出長度 `int(s[4:5]) = 3`，直接擷取索引 `6` 到 `9` 的 `"3#c"`（完全不會被字串內部的 `#` 干擾），`i` 跳轉到 `9` 結束迴圈
step7: 所以答案 = `["ab", "3#c"]`

解釋: 為什麼不能只用單純的分隔符（例如 `","` 或 `"#"`）？因為原始字串本身可能就包含這些符號。

透過「**長度前綴 + 分隔符**」，解碼器在讀取完長度後，會**直接按字元數量跳過內容區段**，因此內容區段裡出現任何符號或數字都不會被誤判為分隔符。

### **LeetCode 271 / LintCode 659 - Encode and Decode Strings（字串的編碼與解碼）**

這道題目是 **字串設計與序列化（String Design & Serialization）** 的經典代表題（Blind 75 與 NeetCode 150 必考題）。問題核心在於設計一套無歧義的通訊協議，讓任意字元組成的字串陣列在扁平化（Flatten）成單一字串後，仍能 100% 精準還原。

## **解法解析**

這道題可以轉化為 **自定義資料封包格式（Header + Payload）** 來求解。

### **思路**

1. **觀察痛點**
    
    - 若直接用特殊符號拼接（如 `",".join(strs)`）：當輸入為 `["a,b", "c"]` 時，解碼會錯誤切成 `["a", "b", "c"]`。
        
    - 若只用長度前綴而不加分隔符（如 `"12abcdefghijkl"`）：當字串長度大於等於 10 時，解碼器無法分辨長度是 `1` 還是 `12`。
        
    - 因此，必須同時具備 **「長度資訊」** 與 **「標記長度結束的分隔符」**。
        
2. **發現編碼協議（Length-Prefix Framing）**
    
    - 每個字串獨立封裝成：`str(len(s)) + "#" + s`
        
    - 這個格式有兩個關鍵保證：
        
        - **長度欄位只包含純數字 `0-9`**：從指標 `i` 開始往右掃描時，遇到第一個 `#` 必然是長度的結尾，絕不可能是字串內容裡的 `#`。
            
        - **內容欄位由長度決定邊界**：一旦取得整數 `length`，程式直接用切片 `s[j+1 : j+1+length]` 取出完整字串，並將指標 `i` 直接移動到下一個封包的開頭，完全跳過中間所有字元。
            
3. **代碼執行流程**
    
    - **`encode(strs)`**：遍歷 `strs`，將每個字串格式化為 `f"{len(s)}#{s}"`，最後用 `"".join()` 合併為一個字串（避免在迴圈中使用 `+=` 拼接字串導致 $O(N^2)$ 時間複雜度）。
        
    - **`decode(s)`**：維護指標 `i = 0`，每次尋找從 `i` 開始的第一個 `#`（位置 `j`），解析 `length = int(s[i:j])`，擷取子字串後令 `i = j + 1 + length`，重複直到 `i == len(s)`。
        

## **具體舉例**

假設編碼後的字串為 `s = "4#lint4#code3#1#a"`，解碼時的變數變化如下：

|**輪次**|**起始 i**|**找到 # 的 j**|**解析 length**|**擷取範圍 s[j+1 : j+1+length]**|**加入 res 的字串**|**更新後 i**|
|---|---|---|---|---|---|---|
|第 1 輪|`0`|`1`|`int("4") = 4`|`s[2:6]`|`"lint"`|`6`|
|第 2 輪|`6`|`7`|`int("4") = 4`|`s[8:12]`|`"code"`|`12`|
|第 3 輪|`12`|`13`|`int("3") = 3`|`s[14:17]`|`"1#a"`|`17`（結束）|

最後返回 `["lint", "code", "1#a"]`，即使第三個字串內部包含 `"1#"`，也因為指標直接跳躍而完全不受影響。

## **時間與空間複雜度分析**

假設陣列中有 $M$ 個字串，所有字串的總字元數為 $N$：

- **時間複雜度：$O(N)$**
    
    - `encode`：計算每個字串長度為 $O(1)$，拼接所有字元需要遍歷總長度 $N$，故為 $O(N)$。
        
    - `decode`：指標 `i` 與 `j` 都只會從左到右單向移動，每個字元最多被訪問常數次，切片總長度為 $N$，故為 $O(N)$。
        
- **空間複雜度：$O(N)$**（若不計入回傳結果所需的輸出空間，額外輔助空間為 $O(1)$）
    
    - `encode` 需要建立長度約為 $N + M \log(\text{max\_len})$ 的輸出字串。
        
    - `decode` 除了儲存回傳陣列 `res`（總字元數 $N$）外，僅使用 `i`、`j`、`length` 等常數級變數。
        

## **其他解法**

### **1. 轉義字元法（Escaping Delimiter）**

- **思路**：
    
    - 選定一個分隔符（例如 `#`）和轉義符（例如 `/`）。
        
    - 編碼時，將原字串中的 `/` 替換為 `//`，將 `#` 替換為 `/#`，最後用 `#` 連接所有字串。
        
    - 解碼時逐字元掃描，遇到 `/` 就讀取下一個字元作為普通字元，遇到單獨的 `#` 則切分字串。
        
- **時間複雜度**：$O(N)$
    
- **空間複雜度**：$O(N)$
    
- **缺點**：需要逐字元掃描與替換，實作較繁瑣且編碼後字串會膨脹（若充滿 `#` 則長度翻倍）。
    

### **2. 非 ASCII 分隔符法（Non-ASCII Delimiter）**

- **思路**：
    
    - 若題目保證輸入僅限標準 ASCII 字元（`0–127`），可直接選用超出 ASCII 範圍的 Unicode 字元（如 `chr(257)` 即 `'ā'`）作為分隔符進行 `join` 與 `split`。
        
    - **注意**：需特別處理空陣列 `[]` 與單一空字串 `[""]` 的邊界差異。
        
- **時間複雜度**：$O(N)$
    
- **空間複雜度**：$O(N)$
    
- **缺點**：在面試中通常屬於「取巧解法」，若輸入包含任意 Unicode 或二進位位元組則會失效，面試官通常會追問如何支援**任意字元集**（即回到「長度前綴法」）。
    

## **總結**

- **面試標準最佳解（Standard Optimal）**：**長度前綴法（`len#str`）**，時間 $O(N)$，額外空間 $O(1)$，支援包含 Unicode、空字串、控制字元在內的**任意字元集**，且實作簡潔不易出錯。
    
- **工程實務連結**：此概念與網路協定中的 **TLV（Type-Length-Value）**、**HTTP Chunked Encoding** 以及 **Protobuf / Redis RESP 協定**完全一致，是後端與系統設計面試的高頻基礎知識。
    

## **LeetCode / LintCode 字串序列化與設計類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 271 / LintCode 659**|Encode and Decode Strings（字串的編碼與解碼）|**字串陣列序列化與反序列化**|**長度前綴 + 分隔符（`len#str`）**，$O(N)$ 雙指標切片|
|**LeetCode 297 / LintCode 7**|Serialize and Deserialize Binary Tree（二元樹的序列化與反序列化）|**樹狀結構轉字串並還原**|**前序遍歷 DFS 或層序遍歷 BFS** + 空節點標記（`#` 或 `null`）|
|**LeetCode 449**|Serialize and Deserialize BST（二元搜尋樹的序列化與反序列化）|**BST 緊湊序列化（不存 null）**|**前序遍歷 + 數值上下界（`min, max`）** 還原樹結構|
|**LeetCode 38**|Count and Say（外觀數列）|**遊程編碼（Run-Length Encoding, RLE）**|**雙指標統計連續相同字元個數**，轉為 `count + char`|
|**LeetCode 443**|String Compression（壓縮字串）|**原地（In-place）字串遊程編碼**|**讀寫雙指標（Read / Write Pointers）**，$O(1)$ 額外空間|
|**LeetCode 394 / LintCode 575**|Decode String（字串解碼）|**巢狀括號與重複次數展開（如 `3[a2[c]]`）**|**雙堆疊（數字堆疊 + 字串堆疊）** 或遞迴下降解析|
|**LeetCode 535 / LintCode 522**|Encode and Decode TinyURL（短網址編解碼）|**系統設計與雙向映射**|**雜湊表（Hash Map）+ Base62 編碼** 生成短碼|
|**LeetCode 604**|Design Compressed String Iterator（設計壓縮字串迭代器）|**串流解碼（Lazy Evaluation）**|**按需解析數字與字元**，避免一次性展開導致記憶體爆炸|

### **進階變形題：如果字串可能包含多位元組（UTF-8 Bytes）或要求直接操作 Byte Stream，該如何調整？**

在真實網路傳輸（如 Socket / TCP 封包）中，字元長度（`len(str)`）與位元組長度（`len(bytes)`）可能不同（例如一個中文字在 UTF-8 佔 3 bytes）。因此必須先將字串編碼為 `bytes`，再以 **位元組長度** 進行封裝：

Python

```
class ByteCodec:
    def encode(self, strs: list[str]) -> bytes:
        res = []
        for s in strs:
            raw_bytes = s.encode("utf-8")
            # 使用位元組長度而非字元長度，並以 b"#" 分隔
            header = f"{len(raw_bytes)}#".encode("utf-8")
            res.append(header + raw_bytes)
        return b"".join(res)

    def decode(self, data: bytes) -> list[str]:
        res = []
        i = 0
        while i < len(data):
            j = data.find(b"#", i)
            byte_len = int(data[i:j])
            start = j + 1
            end = start + byte_len
            res.append(data[start:end].decode("utf-8"))
            i = end
        return res
```