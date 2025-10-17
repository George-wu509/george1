

```cpp
#include <iostream>
using namespace std;

int main ()
{
  int firstvalue = 5, secondvalue = 15;
  int * p1,* p2;

  p1 = &firstvalue;  // p1 = address of firstvalue
  p2 = &secondvalue; // p2 = address of secondvalue
  *p1 = 10;          // value pointed to by p1 = 10
  *p2 = *p1;         // value pointed to by p2 = value pointed to by p1
  p1 = p2;           // p1 = p2 (value of pointer is copied)
  *p1 = 20;          // value pointed to by p1 = 20

  cout << "firstvalue is " << firstvalue << '\n';
  cout << "secondvalue is " << secondvalue << '\n';
  return 0;
}
```
這段程式碼是學習 C++ 指標 (pointer) 非常經典的範例。它完美地展示了**指標本身的值（記憶體位址）**和**指標所指向的值（儲存的資料）**之間的區別。

我將逐行解釋，並追蹤每個變數在每一步的變化。

---

### 變數介紹

在開始之前，我們先了解一下程式中宣告的四個變數：

- `int firstvalue`: 一個整數 (integer) 變數。你可以把它想像成一個貼有「firstvalue」標籤的盒子，裡面裝著一個數字。
    
- `int secondvalue`: 另一個整數變數，是另一個貼有「secondvalue」標籤的盒子。
    
- `int * p1`: 一個**指向整數的指標** (pointer to integer)。它不是一個裝數字的盒子，而是一張紙條，上面寫著**另一個盒子的地址**。星號 `*` 代表它是一個指標。
    
- `int * p2`: 另一個指向整數的指標，也是一張寫地址的紙條。
    

---

### 程式碼逐行解析

#### **Step 1: 初始化變數**

C++

```
int firstvalue = 5, secondvalue = 15;
int * p1, * p2;
```

- **`int firstvalue = 5, secondvalue = 15;`**
    
    - **解釋**: 宣告兩個整數變數。
        
    - **`firstvalue` 的性質**: 型別為 `int`，初始值為 **5**。
        
    - **`secondvalue` 的性質**: 型別為 `int`，初始值為 **15**。
        
- **`int * p1, * p2;`**
    
    - **解釋**: 宣告兩個指向整數的指標。此時它們還沒有指向任何有效的記憶體位址，裡面存的是亂碼（未初始化的垃圾值）。
        
    - **`p1` 的性質**: 型別為 `int*`，值為**未知位址**。
        
    - **`p2` 的性質**: 型別為 `int*`，值為**未知位址**。
        

|變數|`firstvalue`|`secondvalue`|`p1`|`p2`|
|---|---|---|---|---|
|**值**|5|15|未知|未知|

匯出到試算表

---

#### **Step 2: 指標指向變數**

C++

```
p1 = &firstvalue;  // p1 = address of firstvalue
p2 = &secondvalue; // p2 = address of secondvalue
```

- **`p1 = &firstvalue;`**
    
    - **解釋**: `&` 是「取址運算子」。這行程式碼的意思是「取得 `firstvalue` 這個盒子的記憶體位址，然後將這個位址寫入 `p1` 這張紙條」。現在，`p1` 指向了 `firstvalue`。
        
    - **`p1` 的值**: `firstvalue` 的記憶體位址。
        
- **`p2 = &secondvalue;`**
    
    - **解釋**: 同理，將 `secondvalue` 的記憶體位址存入 `p2`。現在，`p2` 指向了 `secondvalue`。
        
    - **`p2` 的值**: `secondvalue` 的記憶體位址。
        

|變數|`firstvalue`|`secondvalue`|`p1`|`p2`|
|---|---|---|---|---|
|**值**|5|15|`&firstvalue`|`&secondvalue`|

匯出到試算表

---

#### **Step 3: 透過指標修改值**

C++

```
*p1 = 10;          // value pointed to by p1 = 10
```

- **解釋**: 星號 `*` 在這裡作為「取值運算子 (dereference operator)」。`*p1` 的意思是「`p1` 這張紙條上地址所對應的那個盒子」。因為 `p1` 指向 `firstvalue`，所以 `*p1` 就是指 `firstvalue` 這個變數本身。
    
- 這行程式碼等同於 `firstvalue = 10;`。
    
- **`firstvalue` 的值**被修改為 **10**。
    

|變數|`firstvalue`|`secondvalue`|`p1`|`p2`|
|---|---|---|---|---|
|**值**|**10**|15|`&firstvalue`|`&secondvalue`|

匯出到試算表

---

#### **Step 4: 透過指標賦值**

C++

```
*p2 = *p1;         // value pointed to by p2 = value pointed to by p1
```

- **解釋**:
    
    - `*p1` 是 `p1` 指向的值，也就是 `firstvalue` 的值，此刻為 **10**。
        
    - `*p2` 是 `p2` 指向的值，也就是 `secondvalue`。
        
    - 整句話的意思是：把 `*p1` 的值（10）賦給 `*p2`。這等同於 `secondvalue = firstvalue;`。
        
- **`secondvalue` 的值**被修改為 **10**。
    

|變數|`firstvalue`|`secondvalue`|`p1`|`p2`|
|---|---|---|---|---|
|**值**|10|**10**|`&firstvalue`|`&secondvalue`|

匯出到試算表

---

#### **Step 5: 修改指標本身**

C++

```
p1 = p2;           // p1 = p2 (value of pointer is copied)
```

- **解釋**: **這是最關鍵的一步！** 這裡沒有星號 `*`，所以我們操作的是指標本身（紙條），而不是它指向的盒子。
    
- 這句話的意思是：「把 `p2` 紙條上寫的地址，複製到 `p1` 紙條上」。
    
- 原本 `p1` 指向 `firstvalue`，`p2` 指向 `secondvalue`。執行後，`p1` 和 `p2` **都指向** `secondvalue`。`firstvalue` 從此「失寵」了，沒有指標指向它了。
    
- **`p1` 的值**被修改為 `secondvalue` 的記憶體位址。
    

|變數|`firstvalue`|`secondvalue`|`p1`|`p2`|
|---|---|---|---|---|
|**值**|10|10|**`&secondvalue`**|`&secondvalue`|

匯出到試算表

---

#### **Step 6: 再次透過指標修改值**

C++

```
*p1 = 20;          // value pointed to by p1 = 20
```

- **解釋**: `*p1` 意思是「`p1` 指向的那個盒子」。因為上一步 `p1` 已經被修改為指向 `secondvalue`，所以 `*p1` 現在指的就是 `secondvalue`。
    
- 這行程式碼等同於 `secondvalue = 20;`。
    
- **`secondvalue` 的值**被修改為 **20**。`firstvalue` 的值完全不受影響。
    

|變數|`firstvalue`|`secondvalue`|`p1`|`p2`|
|---|---|---|---|---|
|**值**|10|**20**|`&secondvalue`|`&secondvalue`|

匯出到試算表

---

### **Step 7: 輸出結果**

C++

```
cout << "firstvalue is " << firstvalue << '\n';
cout << "secondvalue is " << secondvalue << '\n';
```

- **第一行輸出**: `firstvalue` 的最終值是 **10**。
    
- **第二行輸出**: `secondvalue` 的最終值是 **20**。
    

所以，程式執行後的螢幕輸出會是：

```
firstvalue is 10
secondvalue is 20
```





```cpp
#include <iostream>
using namespace std;

int main ()
{
  int numbers[5];
  int * p;
  p = numbers;      *p = 10;
  p++;              *p = 20;
  p = &numbers[2];  *p = 30;
  p = numbers + 3;  *p = 40;
  p = numbers;      *(p+4) = 50;
  for (int n=0; n<5; n++)
    cout << numbers[n] << ", ";
  return 0;
}
```
這段程式碼是理解 C++ 中**陣列 (array)** 和**指標 (pointer)** 之間緊密關係的絕佳範例。它們在記憶體中的運作方式幾乎是相同的。

我將會逐行解析，並以表格形式追蹤陣列 `numbers` 和指標 `p` 的狀態變化。

---

### 變數介紹

- **`int numbers[5];`**
    
    - **性質**: 一個**整數陣列**。它在記憶體中是一塊連續的空間，足夠存放 5 個 `int` 型別的整數。你可以把它想像成一排相連的五個盒子，索引分別為 0, 1, 2, 3, 4。
        
- **`int * p;`**
    
    - **性質**: 一個**指向整數的指標**。它是一個變數，專門用來儲存某個整數的**記憶體位址**。
        

---

### 程式碼逐行解析

#### **Step 1: 宣告**

C++

```
int numbers[5];
int * p;
```

- **`int numbers[5];`**: 宣告陣列 `numbers`。此時，這 5 個整數盒子的內容是未知的（裡面是記憶體中的垃圾值）。
    
- **`int * p;`**: 宣告指標 `p`。此時，`p` 還沒有指向任何地方，它的值也是未知的。
    

**初始狀態:**

|變數/狀態|`numbers[0]`|`numbers[1]`|`numbers[2]`|`numbers[3]`|`numbers[4]`|`p` (指向哪裡)|
|---|---|---|---|---|---|---|
|**值**|?|?|?|?|?|未知|

匯出到試算表

---

#### **Step 2: 指標指向陣列開頭並賦值**

C++

```
p = numbers;
*p = 10;
```

- **`p = numbers;`**
    
    - **解釋**: 在 C++ 中，當你使用陣列的名稱（如此處的 `numbers`），它會自動「退化 (decay)」成一個指向其**第一個元素**的指標。所以這行程式碼等同於 `p = &numbers[0];`。
        
    - **`p` 的值**: 現在儲存了 `numbers[0]` 的記憶體位址。
        
- **`*p = 10;`**
    
    - **解釋**: `*p` 是「取出 `p` 所指向位址的值」。因為 `p` 現在指向 `numbers[0]`，所以這行程式碼等同於 `numbers[0] = 10;`。
        
    - **`numbers[0]` 的值**: 被更新為 **10**。
        

**狀態更新:**

|變數/狀態|`numbers[0]`|`numbers[1]`|`numbers[2]`|`numbers[3]`|`numbers[4]`|`p` (指向哪裡)|
|---|---|---|---|---|---|---|
|**值**|**10**|?|?|?|?|`numbers[0]`|

匯出到試算表

---

#### **Step 3: 移動指標並賦值**

C++

```
p++;
*p = 20;
```

- **`p++;`**
    
    - **解釋**: 這是**指標算術 (pointer arithmetic)**。對一個指標進行 `++` 操作，不是將其位址值加 1，而是讓它指向陣列中的**下一個元素**。它會自動加上一個 `int` 所佔的位元組數（通常是 4 bytes）。
        
    - **`p` 的值**: 從指向 `numbers[0]` 變為指向 `numbers[1]`。
        
- **`*p = 20;`**
    
    - **解釋**: 因為 `p` 現在指向 `numbers[1]`，所以這行程式碼等同於 `numbers[1] = 20;`。
        
    - **`numbers[1]` 的值**: 被更新為 **20**。
        

**狀態更新:**

|變數/狀態|`numbers[0]`|`numbers[1]`|`numbers[2]`|`numbers[3]`|`numbers[4]`|`p` (指向哪裡)|
|---|---|---|---|---|---|---|
|**值**|10|**20**|?|?|?|`numbers[1]`|

匯出到試算表

---

#### **Step 4: 指標跳轉並賦值**

C++

```
p = &numbers[2];
*p = 30;
```

- **`p = &numbers[2];`**
    
    - **解釋**: `&` 是「取址運算子」。這行程式碼直接將 `numbers[2]`（第三個元素）的記憶體位址賦給 `p`。指標 `p` 直接「跳」到了陣列的第三個位置。
        
    - **`p` 的值**: 現在儲存了 `numbers[2]` 的記憶體位址。
        
- **`*p = 30;`**
    
    - **解釋**: 因為 `p` 現在指向 `numbers[2]`，所以這行程式碼等同於 `numbers[2] = 30;`。
        
    - **`numbers[2]` 的值**: 被更新為 **30**。
        

**狀態更新:**

|變數/狀態|`numbers[0]`|`numbers[1]`|`numbers[2]`|`numbers[3]`|`numbers[4]`|`p` (指向哪裡)|
|---|---|---|---|---|---|---|
|**值**|10|20|**30**|?|?|`numbers[2]`|

匯出到試算表

---

#### **Step 5: 使用指標算術跳轉並賦值**

C++

```
p = numbers + 3;
*p = 40;
```

- **`p = numbers + 3;`**
    
    - **解釋**: 這也是指標算術。`numbers` 代表陣列開頭的位址，`numbers + 3` 計算出的就是「從陣列開頭算起的第 4 個元素（索引為 3）的位址」。這行程式碼等同於 `p = &numbers[3];`。
        
    - **`p` 的值**: 現在儲存了 `numbers[3]` 的記憶體位址。
        
- **`*p = 40;`**
    
    - **解釋**: 因為 `p` 現在指向 `numbers[3]`，所以這行程式碼等同於 `numbers[3] = 40;`。
        
    - **`numbers[3]` 的值**: 被更新為 **40**。
        

**狀態更新:**

|變數/狀態|`numbers[0]`|`numbers[1]`|`numbers[2]`|`numbers[3]`|`numbers[4]`|`p` (指向哪裡)|
|---|---|---|---|---|---|---|
|**值**|10|20|30|**40**|?|`numbers[3]`|

匯出到試算表

---

#### **Step 6: 重設指標並透過位移賦值**

C++

```
p = numbers;
*(p+4) = 50;
```

- **`p = numbers;`**
    
    - **解釋**: 將指標 `p` 重新設定，讓它再次指向陣列的開頭 `numbers[0]`。
        
    - **`p` 的值**: 重新變為 `numbers[0]` 的記憶體位址。
        
- **`*(p+4) = 50;`**
    
    - **解釋**: 這是一個很重要的語法。`p+4` 計算出一個**臨時的位址**（`p` 指向的位址 + 4 個元素的距離），也就是 `numbers[4]` 的位址。然後 `*` 對這個臨時位址取值並賦值為 50。
        
    - **重要的是**: `p` 本身的值**沒有改變**，它仍然指向 `numbers[0]`。
        
    - 這行程式碼等同於 `numbers[4] = 50;`。
        
    - **`numbers[4]` 的值**: 被更新為 **50**。
        

**最終狀態:**

|變數/狀態|`numbers[0]`|`numbers[1]`|`numbers[2]`|`numbers[3]`|`numbers[4]`|`p` (指向哪裡)|
|---|---|---|---|---|---|---|
|**值**|10|20|30|40|**50**|`numbers[0]`|

匯出到試算表

---

#### **Step 7: 輸出結果**

C++

```
for (int n=0; n<5; n++)
  cout << numbers[n] << ", ";
```

- **解釋**: 這是一個簡單的迴圈，依序 (`n` 從 0 到 4) 印出 `numbers` 陣列中每一個元素的值，並在每個值後面加上 ", "。
    
- **輸出內容**: 根據我們追蹤的最終狀態，程式會印出陣列的所有內容。
    

最終，螢幕上的輸出會是：

```
10, 20, 30, 40, 50, 
```
