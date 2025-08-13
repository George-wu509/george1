

|                                            |                                                                                                             |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| [[Transformer and ViT - architecture]]     |                                                                                                             |
|                                            | [[Transformer and ViT - positional encoding]]                                                               |
| [[Transformer and ViT - encoder &decoder]] |                                                                                                             |
| Encoder                                    | [[Transformer and ViT - ViT Encoder block]]<br><br>Multi-head self attention<br>Feed Forward neural network |
|                                            | [[Transformer and ViT - Attention]]                                                                         |
|                                            |                                                                                                             |
| Decoder                                    | Masked Multi-head                                                                                           |
|                                            |                                                                                                             |
| [[### QA list]]                            |                                                                                                             |


|                                  |     |
| -------------------------------- | --- |
| [[### 兩種 Normalization 配置]]      |     |
| [[### QA list]]                  |     |
| [[#### 舉例說明Encoder-Decoder的QKV]] |     |




![[transformer.webp]]
![[Pasted image 20250521144001.png]]
(現在常用)在ViT的Encoder block裡面的內部順序是 (1) Layer normalization, (2) Multi-head self attention, (3) Layer normalization, (4) Feed-Forward Network(FFN) .
Skip connection連接Layer normalization之前的輸入和Multi-head self attention的輸出.

總結來說，ViT Encoder Block 的結構是：**LayerNorm -> Multi-Head Self-Attention + Skip Connection -> LayerNorm -> Feed-Forward Network + Skip Connection**。並且所有的Normalization都使用的是Layer Normalization。

在Transformer的架構中，Normalization 層的位置確實有兩種常見的配置，分別稱為 **Post-LN (Post-Normalization)** 和 **Pre-LN (Pre-Normalization)**。

### 兩種 Normalization 配置

1. **Post-LN (Post-Normalization)**:
    
    - 這是在原始的 **"Attention Is All You Need"** 論文中採用的配置。
    - **順序**:
        
        ```
        Input ----> Multi-Head Self-Attention ----> Add (Skip Connection) ----> LayerNorm ----> Feed-Forward Network ----> Add (Skip Connection) ----> LayerNorm ----> Output
          ^                                       ^
          |_______________________________________|
                                                  ^
                                                  |_______________________________________|
        ```
        
    - **在每個子層（MSA 和 FFN）之後**進行歸一化。
2. **Pre-LN (Pre-Normalization)**:
    
    - 這是一種更現代、更常用的配置，特別是在較深的 Transformer 模型中。您在之前的問題中描述的順序就是這種 Pre-LN。
    - **順序**:
        
        ```
        Input ----> LayerNorm ----> Multi-Head Self-Attention ----> Add (Skip Connection) ----> LayerNorm ----> Feed-Forward Network ----> Add (Skip Connection) ----> Output
          ^                               ^                                                    ^
          |_______________________________|                                                    |_______________________________|
        ```
        
    - **在每個子層（MSA 和 FFN）之前**進行歸一化。





詳細解釋一下Vision Transformer (ViT) 在圖像分割中全局注意力的意義，以及Q, K, V 在圖像 Patch 中的角色。

**1. 圖像分 Patch 與 Token**

在 Vision Transformer 中，一張圖像首先會被分割成一系列固定大小的小圖像塊 (Patches)。例如，一張 224x224 像素的圖像，如果 Patch 大小是 16x16 像素，那麼就會得到 (224/16) * (224/16) = 14 * 14 = 196 個 Patches。

每個 Patch 接著會被展平 (flatten) 並通過一個線性投射層 (Linear Projection Layer) 轉換成一個固定維度的向量，這個向量就被視為一個 "Token"，類似於自然語言處理 (NLP) 中的詞彙 Token。此外，還會加入位置編碼 (Positional Encoding) 來保留 Patch 之間的空間位置信息。

**2. 两两计算全局注意力的真实意义是什么？是指两个图片 Patch 之间的相似性吗？**

是的，可以將全局注意力計算的過程理解為**衡量任意兩個圖像 Patch 之間在特定上下文中的“相關性”或“相似性”**。但這裡的“相似性”並非簡單的像素級別的直接相似，而是更高維度、更抽象的語義或特徵層面的相似性。

- **真實意義：** 全局注意力的核心思想是讓模型中的每個 Patch (Token) 都能夠“看到”並“評估”圖像中的所有其他 Patches，從而決定哪些 Patches 對於理解當前 Patch 的內容和上下文最為重要。它允許模型捕捉圖像中長距離的依賴關係。 對於一個特定的 Patch A，全局注意力機制會計算 Patch A 與圖像中所有其他 Patches (包括它自己) 的注意力分數。這個分數越高，代表其他 Patches 對於更新 Patch A 的表徵 (representation) 越重要。
    
- **如何計算（簡化版）：** 對於一個查詢 Patch Q (Query)，它會和所有其他 Patches 的鍵 K (Key) 進行比較（通常是點積運算）。這個比較結果會經過一個 Softmax 函數，轉換成一組權重。這些權重就代表了 Q 對於所有其他 Patches 的注意力分佈。權重越大的 K 所對應的 Patch，就被認為與 Q 更“相關”或“相似”。
    

**舉個簡單例子：** 想像一張圖裡有一隻貓和一個球。

- 一個包含貓耳朵的 Patch，在計算注意力時，可能會對包含貓頭部其他部分的 Patches、貓身體的 Patches 產生較高的注意力分數，因為它們在語義上高度相關（都屬於“貓”這個物體）。
- 它對遠處那個球的 Patch 可能注意力分數較低，除非在特定任務中，貓和球的互動關係很重要。
- 它對背景草地的 Patch 注意力分數可能更低。

所以，這種“相似性”是模型在學習過程中動態學習到的，服務於當前任務（如圖像分割）的特徵層面的相關性。

**3. 这跟 Image Segmentation 有什么关系？**

全局注意力機制對於圖像分割任務非常有幫助，主要體現在以下幾點：

- **捕捉長距離依賴關係 (Long-Range Dependencies)：** 圖像分割的目標是為圖像中的每個像素（或每個 Patch）分配一個類別標籤。很多時候，同一個物體的不同部分可能在圖像中相距較遠（例如，一條蛇的頭和尾巴）。傳統的卷積神經網絡 (CNN) 由於其局部感受野的限制，需要堆疊很多層才能捕捉到這種長距離關係。而全局注意力機制允許任何兩個 Patch 直接交互，無論它們在圖像中的距離有多遠。這使得模型能更好地理解物體的完整結構，即使物體被部分遮擋或形狀不規則。
    
- **理解全局上下文 (Global Context)：** 一個 Patch 的語義不僅取決於其自身內容，還取決於它在整個圖像中的上下文。例如，一個棕色的圓形 Patch，如果周圍的 Patches 是樹幹和樹葉，它可能被識別為樹木的一部分；如果周圍的 Patches 是車輪和車窗，它可能被識別為汽車的輪胎。全局注意力允許每個 Patch 整合來自整個圖像的信息，從而做出更準確的判斷。
    
- **提升分割一致性：** 通過讓屬於同一物體的不同 Patches 之間產生強烈的相互注意力，模型可以學習到這些 Patches 應該被賦予相同的分割標籤，從而提高分割結果的內部一致性和平滑性。例如，如果模型確定某個 Patch 是“汽車”，那麼與之高度相關的其他 Patches（如車門、車窗、車輪）也更有可能被正確標記為“汽車”。
    
- **區分相似但不同類別的物體：** 有時，不同的物體可能在局部看起來很相似。全局上下文可以幫助區分它們。例如，一個灰色的 Patch 可能是路面，也可能是建築物的牆壁。通過觀察與這個 Patch 相關聯的其他 Patches（是天空還是其他車輛？），模型可以更準確地判斷其類別。
    

**4. Multi-Head Self-Attention 的 Q, K, V 在图片 Patch 的意义是什么？**

在自注意力機制 (Self-Attention) 中，每個輸入的 Patch Token 會生成三個不同的向量：查詢 (Query, Q)，鍵 (Key, K)，和值 (Value, V)。這三個向量是通過將原始的 Patch Embedding 乘以三個不同的、可學習的權重矩陣 (W_q, W_k, W_v) 得到的。

- **Query (Q) - 查詢向量：**
    
    - **意義：** 代表當前 P<mark style="background: #FFF3A3A6;">atch 正在“尋找什麼信息”或“對什麼感興趣</mark>”。可以把它想像成當前 Patch 提出的一個問題或一個查詢請求。
    - **例子：** 一個包含貓眼睛的 Patch，它的 Q 向量可能編碼了“<mark style="background: #FFF3A3A6;">我是一個物體的關鍵部分，請告訴我與我相似的其他部分在哪裡</mark>，或者與我構成同一物體的其他部分在哪裡”這樣的信息。它想知道其他 Patches 中有哪些特徵與“貓眼”這個概念相關。
- **Key (K) - 鍵向量：**
    
    - **意義：** 代表該 Patch “<mark style="background: #ABF7F7A6;">擁有什麼樣的信息”或“能提供什麼樣的標識特徵</mark>”。它是用來和其他 Patches 的 Q 向量進行匹配的。
    - **例子：** 還是那個貓眼睛的 Patch，它的 K 向量可能編碼了“<mark style="background: #ABF7F7A6;">我包含貓眼特有的紋理、形狀和顏色</mark>”這樣的信息。對於一個包含貓耳朵的 Patch，它的 K 向量則會編碼貓耳朵的特徵。
- **Value (V) - 值向量：**
    
    - **意義：** 代表該 Patch “<mark style="background: #FFB86CA6;">實際攜帶的內容或信息</mark>”。一旦 Q 和 K 匹配成功（即注意力權重較高），那麼對應的 V 向量就會被用來更新查詢 Patch 的表徵。它是實際被提取和聚合的信息。
    - **例子：** 貓眼睛 Patch 的 V 向量攜帶了關於貓眼睛的豐富特徵表示。如果貓耳朵 Patch 的 Q 與貓眼睛 Patch 的 K 高度匹配，那麼貓眼睛 Patch 的 V 向量就會在很大程度上貢獻給貓耳朵 Patch 更新後的特徵表示。

**計算過程簡述：**

1. 對於一個特定的 Patch (稱為 Patch_i)，生成其 Q_i。
2. 對於圖像中的所有 Patches (包括 Patch_i 自身，稱為 Patch_j)，生成它們的 K_j 和 V_j。
3. 計算 Q_i 與所有 K_j 的點積，得到“注意力分數”（`score_ij = Q_i · K_j`）。這個分數衡量了 Patch_i 的查詢與 Patch_j 的鍵之間的匹配程度。
4. 將這些分數進行縮放（通常除以 Q、K 向量維度的平方根）並通過 Softmax 函數歸一化，得到注意力權重 (attention weights, `alpha_ij`)。`alpha_ij` 表示 Patch_i 應該對 Patch_j 的信息 V_j 投入多少注意力。
5. 用這些注意力權重 `alpha_ij` 對所有 V_j 進行加權求和，得到 Patch_i 的新表徵：`Output_i = Σ (alpha_ij * V_j)`。

**Multi-Head Self-Attention (多頭自注意力)：** 多頭機制不是只有一組 Q, K, V 的權重矩陣，而是有多組（例如 8 組或 12 組）。每一組（稱為一個“頭”）都獨立地執行上述的自注意力計算。

- **意義：**
    
    - **從不同子空間學習信息：** 不同的頭可以學會關注 Patch 之間不同類型的關係或特徵。例如，一個頭可能專注於紋理相似性，另一個頭可能專注於顏色相似性，還有一個頭可能專注於空間位置關係或者物體的組成部分關係。
    - **更豐富的表徵能力：** 將多個頭的輸出拼接起來（Concat）然後再進行一次線性變換，可以讓模型從多個角度、多個方面捕捉 Patch 之間複雜的依賴關係，從而得到更豐富、更魯棒的特徵表徵。
- **例子（續貓和球）：**
    
    - **頭1 (紋理頭)：** 貓耳朵 Patch 的 Q 可能在尋找相似毛髮紋理。它會高度關注貓身體其他毛髮 Patch 的 K，並聚合它們的 V。
    - **頭2 (形狀頭)：** 貓耳朵 Patch 的 Q 可能在尋找符合貓科動物輪廓的形狀。
    - **頭3 (顏色頭)：** 如果貓是特定顏色的，這個頭可能關注顏色一致性。
    - **頭4 (長距離關係頭)：** 可能會學習到“頭部”和“尾部”雖然形態不同，但經常共同出現，屬於同一物體。

最終，通過這種全局多頭自注意力機制，每個 Patch 的表徵都融合了來自圖像中所有其他 Patches 的、在多個語義層面上的相關信息。這些經過豐富和優化的 Patch 表徵隨後被送入後續的網絡層（通常是前饋神經網絡），並最終用於圖像分割任務中的像素級（或 Patch 級）分類。

總結來說，Vision Transformer 中的全局注意力機制通過計算 Patch 之間的動態相關性，使得模型能夠理解圖像的整體結構和上下文信息，這對於需要精確識別和分割物體的圖像分割任務至關重要。而 Q, K, V 的引入，則為這種相關性計算提供了一個靈活且強大的框架，多頭機制進一步增強了其從不同維度捕捉信息的能力。

Reference:
【动手学深度学习】一文详解Transformer架构及其代码实现 - 薯条算法的文章 - 知乎
https://zhuanlan.zhihu.com/p/1936167984377890005

### QA list

| Q                                                                       | Ans                                                                                                                                                                                                            |
| ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ViT将图像分割为固定大小的patch后，如何将这些patch转换为适合Transformer处理的序列？能否详细描述预处理流程中的关键步骤？ | 对patch进行线性embedding，维度变成模型hidden dims，并且与1D可学习的位置编码直接相加（element-wise add）。<br><br>预处理过程中要考虑恰当的patch大小，经过embedding之后，为了和NLP中的架构尽可能相似，会额外添加一个 `class token`与 `patch embeddings`进行拼接，最后和1D可学习的位置编码参数直接相加作为编码层的输入。 |
| ViT的位置编码与传统NLP中的位置编码有何不同？如果不使用位置编码，会对模型性能产生什么影响？                        | Transformer中使用的位置编码是根据位置信息计算正余弦值，而ViT中选择1D位置编码参数，是可学习的；论文中对不使用位置编码的情况进行了对比实验，会导致模型性能稍有下降（大约下降3个百分点）。                                                                                                           |
| ViT的"分类token"（class token）在模型中的作用是什么？为什么需要将它添加到patch序列中？                | 分类token在前向计算过程中会融合图像的全局信息，最终作为图像特征输入到分类头中，完成分类任务。<br><br>原论文中提到为了和NLP（BERT模型）中的设置尽可能相似，将其与patches进行拼接，作为编码器的输入，不过实验证明，如果不使用 `class token`、而是为分类头输入编码器的输出，也可得到相近的性能。                                            |
| ViT与CNN的核心区别是什么？为什么ViT在大规模数据集上表现更好，但在小数据集上容易过拟合？                        | CNN中依赖卷积操作提取特征，其中有局部性、平移不变性等先验知识，而ViT中仅引入了非常少的图像先验知识，完全依靠模型学习图像patch之间的关系。                                                                                                                                     |
| 自注意力机制在图像处理中可能面临哪些计算效率问题？ViT如何通过patch划分缓解这一问题？                          | 如果将图像中的每个像素看作token输入到模型中，QKV的计算时会面临大矩阵乘法问题，导致计算效率降低，而ViT通过划分patch、将每个patch作为token，减少了token的序列长度，从而缓解了上面的大矩阵乘法问题。<br><br>当输入图像分辨率与预训练模型不匹配时，ViT需要如何调整？试解释位置编码插值的实现逻辑<br><br>在原始的预训练得到的位置编码的基础上，根据图像位置的关系进行2D插值。 |
| 从ViT的实验结果来看，为什么它在ImageNet-21k等大规模数据集上的表现优于CNN？这种优势是否能迁移到小规模数据集？         | ViT通过全局自注意力机制，可以在不引入先验知识的情况下，学习到图像的通用特征，因此在大规模数据上性能优于CNN；同时如果使用少样本微调，ViT的性能也好过CNN。<br><br>在工业场景（如医学影像分析或自动驾驶）中部署ViT时，可能面临哪些实际挑战？列举三种优化策略并说明原理                                                                  |
| self attention和 cross attention 的区别                                     |                                                                                                                                                                                                                |
| LoRA的实现原理                                                               |                                                                                                                                                                                                                |
| 了解 Transformer 吗，编码器和解码器的注意力有什么区别，在计算注意力中时除以 \sqrt{d_k} 的原因是什么          |                                                                                                                                                                                                                |
| 简单介绍下Transformer                                                        |                                                                                                                                                                                                                |
| 大概讲一下Transformer的Encoder模块？                                             |                                                                                                                                                                                                                |
| 为什么transformer块使用LayerNorm而不是BatchNorm？                                 |                                                                                                                                                                                                                |
| Transformer为何使用多头注意力机制？                                                 |                                                                                                                                                                                                                |
| Coding>> **手写multi-head attention**                                     |                                                                                                                                                                                                                |
| Coding>> **手写self attention**                                           |                                                                                                                                                                                                                |
| 解释什么是自注意力机制                                                             |                                                                                                                                                                                                                |
| 很多大模型 decoder-only原理                                                    |                                                                                                                                                                                                                |
| Cross-attention用法                                                       |                                                                                                                                                                                                                |
| 注意力的公式                                                                  |                                                                                                                                                                                                                |
| 幾種position embedding                                                    |                                                                                                                                                                                                                |
| 為什麼大模型LLM 最主流的預訓練方式是自回歸                                                 |                                                                                                                                                                                                                |
| Attention的原理, Attention有什么缺点                                            |                                                                                                                                                                                                                |
| VIT直接用于分割检测等预测密集型的任务上存在什么问题？                                            |                                                                                                                                                                                                                |
| VIT中对输入图像的处理是如何将patch变化为token的？                                         |                                                                                                                                                                                                                |
| Transformer的注意力机制常用softmax函数，可以使用sigmoid代替吗？                            |                                                                                                                                                                                                                |
| ViT、DEIT是如何处理变长序列输入的？                                                   |                                                                                                                                                                                                                |
| 局部注意力如何实现                                                               |                                                                                                                                                                                                                |
| 神经网络引入注意力机制后效果降低的原因                                                     |                                                                                                                                                                                                                |
| Attention计算复杂度以及如何改进                                                    |                                                                                                                                                                                                                |
| 为什么transformer中使用LayerNorm而不是BatchNorm                                  |                                                                                                                                                                                                                |
| Transformer为何使用多头注意力机制                                                  |                                                                                                                                                                                                                |



#### 舉例說明Encoder-Decoder的QKV

```
Question: 有一張照片裡面有一隻貓有清楚的臉跟身體. 如果我用Transformer-based object detection model譬如DETR進行object detection. 請詳細用中文並一步步具體舉例解釋encoder-decoder架構內的multi-head self-attention跟multi-head cross attention跟masked multi-head attention怎麼協作? multi-head self-attention的Query, key, value跟multi-head cross attention的Query, key, value怎麼協作?, 如何形成最後的object detection結果
```

好的，這是一個非常好的問題。讓我以您提供的「一隻有清楚臉跟身體的貓」的照片為例，一步步詳細解釋DETR（DEtection TRansformer）模型中，各種Attention機制是如何協同工作，最終完成物件偵測的。

### 總覽：從照片到偵測結果

整個流程可以想像成一個分工合作的團隊：

1. **特徵提取員 (CNN Backbone)**：先對貓的照片進行初步分析，提取出關鍵的視覺特徵。
    
2. **情境理解專家 (Encoder)**：接收這些視覺特徵，並透過內部討論（**Multi-Head Self-Attention**）來理解圖片的整體情境，例如「貓的耳朵旁邊是貓的眼睛」，「貓的身體連接著尾巴」。
    
3. **偵探團隊 (Decoder)**：派出N個偵探（Object Queries），每個偵探都帶著一個任務：「去圖片裡找一個物體」。他們會：
    
    - 彼此開會溝通，避免重複工作 (Decoder的 **Masked Multi-Head Self-Attention**)。
        
    - 拿著線索去審問「情境理解專家」提供的圖片資訊 ( **Multi-Head Cross-Attention** )，找出自己負責的物體在哪裡。
        
4. **報告生成器 (FFN)**：每個偵探完成調查後，提交一份標準格式的報告，說明他找到了什麼物體（類別）以及在哪裡（邊界框）。
    

---

### 第一步：特徵提取 (CNN Backbone)

首先，這張貓的照片會被輸入一個卷積神經網絡（CNN），例如ResNet。

- **輸入**：`[寬 x 高 x 3]` 的貓咪照片。
    
- **輸出**：一個較小的二維特徵圖（Feature Map），例如 `[W' x H' x C]`。這個特徵圖可以被看作是濃縮後的視覺資訊，其中 `C` 是特徵的維度。
    

為了讓Transformer能夠處理，我們需要將這個二維特Зв圖「拉平」，變成一個序列：

1. **拉平 (Flatten)**：將 `[W' x H' x C]` 的特徵圖轉換成一個 `[W' * H', C]` 的序列。現在我們有 `W' * H'` 個特徵向量，每個向量維度為 `C`。
    
2. **加入位置資訊 (Positional Encoding)**：由於Transformer本身沒有順序概念，我們必須為序列中的每一個特徵向量加入「位置編碼」，告訴模型這個特徵來自於圖片的哪個位置。
    

現在，我們得到了一個帶有位置資訊的特徵序列，準備將它送入Encoder。

---

### 第二步：Encoder - 透過 Multi-Head Self-Attention 理解圖片

Encoder的目標是讓圖片中的每個「局部特徵」都能夠理解它與圖片中所有其他「局部特徵」的關係，建立全域的上下文感知。

#### Multi-Head Self-Attention 的協作機制

這裡的「Self-Attention」意味著注意力機制的 **Query (Q), Key (K), Value (V)** 都來自於 **同一個來源**：上一步產生的「圖片特徵序列」。

想像一下序列中的每個特徵向量（代表圖片的一個小區域，比如貓的左耳）都要問自己三個問題來生成Q, K, V：

- **Query (Q)**：我（左耳）應該去關注圖片中的哪些資訊？
    
- **Key (K)**：我（左耳）攜帶了什麼樣的關鍵資訊，可以被別人查詢？
    
- **Value (V)**：如果有人關注我，我（左耳）應該提供什麼樣的具體內容？
    

**具體協作流程：**

1. **生成Q, K, V**：對於序列中的每一個特徵向量，都透過三個獨立的線性轉換（乘以權重矩陣 WQ​,WK​,WV​）生成各自的Q, K, V向量。
    
2. **計算注意力分數**：以「貓的左耳」這個特徵向量生成的Query（我們稱之為 Q左耳​）為例，它會和 **圖片中所有特徵向量** 的Key（K眼睛​,K鼻子​,K尾巴​,...）進行點積運算。這個分數代表了「左耳」對其他部位的關注程度。如果 Q左耳​ 和 K眼睛​ 的匹配度很高，分數就會很高。
    
3. **權重分配**：將得到的所有分數進行Softmax歸一化，變成一組權重。這些權重加總為1，代表了「左耳」應該將它的注意力如何分配給圖片的所有部分。
    
4. **加權求和**：將這些權重分別乘以對應特徵的Value向量（V眼睛​,V鼻子​,V尾巴​,...），然後全部加總起來。
    
5. **形成新向量**：這個加權總和的結果，就成為了「貓的左耳」這個位置 **新的特徵向量**。這個新向量不僅包含了左耳本身的資訊，還融合了它最關注的（例如，貓眼睛、貓頭頂）其他部位的資訊。
    

**Multi-Head（多頭）的意義：**

這個過程不會只做一次，而是會並行地做 `h` 次（例如 `h=8`），`h` 就是「頭」的數量。每個「頭」都有自己獨立的 WQ​,WK​,WV​ 權重矩陣。

- **協作方式**：這就像讓8個專家同時分析。
    
    - **Head 1** 可能學會了關注「輪廓」關係。
        
    - **Head 2** 可能學會了關注「紋理」關係。
        
    - **Head 3** 可能學會了關注「顏色」關係。
        
- 最後，將8個頭的輸出結果拼接起來，再通過一次線性轉換，得到最終的輸出。
    

經過多層Encoder（每一層都重複上述過程）後，我們得到一組 **富含上下文資訊的記憶（Memory）**。此時，代表「貓左耳」的向量已經「知道」了它旁邊是眼睛，並且同屬於一張貓臉。

---

### 第三步：Decoder - 三種Attention機制如何協作找到貓

Decoder的任務是從Encoder產生的「記憶」中，精準地定位出物體。它有兩個輸入：

1. **Encoder的輸出**（我們稱之為 **Memory**）。
    
2. **N個Object Queries**：這是一組可學習的向量（例如N=100）。可以把它們想像成N個「空的物體槽」或「偵探」。在訓練開始時它們是隨機的，但會逐漸學會代表「一個潛在物體」的查詢。
    

Decoder的每一層都包含以下三種Attention機制，它們的協作至關重要：

#### 1. Masked Multi-Head Self-Attention (在Decoder內部)

- **目標**：讓N個Object Queries之間互相溝通，了解彼此。這主要是為了避免重複偵測。如果兩個Query都開始關注同一隻貓，透過這個機制，它們可以協商，讓其中一個去尋找其他物體，或者抑制其中一個的信心。
    
- **Q, K, V 來源**：全部來自於 **Object Queries** 本身。
    
- **協作方式**：與Encoder的Self-Attention類似，每個Object Query都會生成自己的Q, K, V，並與 **所有其他** Object Queries互動。Query A會關注到Query B也想找貓，從而調整自己的策略。
    
- **Masked的意義**：在原始的用於語言翻譯的Transformer中，Mask是為了防止在生成第 `i` 個詞時看到後面的詞。在DETR中，所有Queries是並行處理的，所以這裡的Self-Attention通常不是因果遮罩（causal mask），而是讓所有Queries互相交流。
    

#### 2. Multi-Head Cross-Attention (跨Encoder和Decoder)

這是 **最關鍵的偵測步驟**。Object Query會在這裡查詢Encoder的Memory，找到它要找的物體。

- **目標**：將物體查詢（Object Query）與圖片內容（Memory）進行匹配。
    
- **Q, K, V 來源**：
    
    - **Query (Q)**：來自於經過了上一步Self-Attention之後的 **Object Queries**。這代表了偵探提出的具體問題，例如：「圖片裡哪部分最像貓？」
        
    - **Key (K)**：來自於 **Encoder的輸出（Memory）**。這代表了圖片各部分內容的「索引」或「關鍵資訊」。
        
    - **Value (V)**：也來自於 **Encoder的輸出（Memory）**。這代表了圖片各部分內容的「具體資訊」。
        
- **協作方式**：
    
    1. 某一個Object Query（我們稱之為Query A）會生成它的 QA​。
        
    2. QA​ 會和 **Encoder Memory中所有特徵向量** 的Key（K貓臉特徵​,K貓身特徵​,K背景特徵​,...）進行比較，計算注意力分數。
        
    3. QA​ 自然會與代表「貓」的那些特徵向量的Key產生高分。
        
    4. 根據這些分數，Query A會對Memory中所有特徵的Value進行加權求和。
        
    5. 結果是，Query A的向量表示中，就融合了圖片中「貓」的精確視覺資訊。這個Query從一個模糊的「物體查詢」變成了「被貓的資訊填充」的查詢。
        

#### 3. 前饋神經網絡 (Feed-Forward Network, FFN)

Cross-Attention的輸出會再經過一個FFN進行資訊的進一步處理和提煉。

**Decoder層的協作總結：**

一個Decoder層的完整流程是： `Object Queries` -> `Self-Attention` (Queries互相溝通) -> `Cross-Attention` (Queries去圖片記憶裡找物體) -> `FFN` (處理找到的資訊)。 這個過程會重複多次（例如6次），每一層都會讓Object Query對物體的定位和理解越來越精確。

---

### 第四步：生成最終結果 (FFN Heads)

經過所有Decoder層後，我們得到了N個更新後的Object Query向量。每一個向量現在都可能包含了某個物體的資訊。

最後，這N個向量會被分別送入兩個獨立的、共享權重的FFN（也稱為預測頭）：

1. **分類頭 (Classification Head)**：一個線性層 + Softmax。它會預測這個Query對應的物體類別。例如，對於我們填充了貓咪資訊的那個Query，它會輸出最高的機率給「貓」這個類別。對於沒找到物體的Query，它會輸出「無物體 (no object)」。
    
2. **邊界框頭 (Bounding Box Head)**：一個多層感知機 (MLP)。它會預測4個數值，代表物體的邊界框：中心點座標 `(x, y)`、寬度 `w` 和高度 `h`。
    

**最終結果：**

對於我們的貓咪照片，假設N=100，最終我們會有100組預測。

- **Query #12** 可能輸出：`{class: "貓", probability: 0.98, box: [0.5, 0.5, 0.4, 0.6]}`
    
- **其他大多數Queries** 可能輸出：`{class: "無物體", probability: 0.99, box: [...]}`
    

通過設定一個信心度閾值（例如0.7），我們就可以篩選出所有被成功偵測到的物體。在這個例子中，模型成功地在照片中找到了貓，並給出了它的類別和精確的邊界框。這就是整個協作流程的最終成果。