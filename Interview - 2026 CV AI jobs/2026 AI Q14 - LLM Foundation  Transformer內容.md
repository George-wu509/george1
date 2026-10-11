
|                                         |     |
| --------------------------------------- | --- |
| [[#### LLM Foundation / Transformer內容]] |     |
|                                         |     |
|                                         |     |

#### LLM Foundation / Transformer內容
```
請完整深入解釋以下LLM Foundation / Transformer內容(先以介紹intern了解知識, 之後則更深入完整介紹engineer技術細節)：

| 知識                  | 需要掌握的深度                                   |
| ------------------- | ----------------------------------------- |
| Transformer         | Attention、Feed-forward、Residual、LayerNorm |
| Self-attention      | Query、Key、Value、Scaled Dot-product        |
| Positional Encoding | Sinusoidal、RoPE 等方法                       |
| Tokenization        | BPE、Token Count、Vocabulary                |
| Model Architecture  | Encoder-only、Decoder-only、Encoder-decoder |
| Generation          | Greedy、Beam Search、Temperature、Top-p      |
| Embeddings          | Semantic Similarity、Representation        |
| Context Window      | Long Context、Truncation、Memory            |
| KV Cache            | 如何加速 Autoregressive Decoding              |
```

import numpy as np
q=np.array([1.,0]); K=np.array([[1.,0],[0.,1],[1.,1.]])
s=K@q/np.sqrt(2)
a=np.exp(s-s.max());a=a/a.sum()
s,a,a@np.array([[10,0],[0,10],[10,10]])
(array([0.70710678, 0.        , 0.70710678]),
 array([0.40111209, 0.19777581, 0.40111209]),
 array([8.02224185, 5.98887907]))
LLM Foundation / Transformer：從 Intern 到 Senior LLM Engineer 的完整技術教學
這九項知識是理解 GPT、Llama、Qwen、Claude 等現代大型語言模型（LLM）的核心基礎。

如果目標是美國的 Senior LLM Engineer、Applied AI Engineer、Foundation Model Research Engineer 或 LLM Inference Engineer，不只是要知道這些名詞，更應該能夠：

解釋每個技術解決什麼問題，以及為什麼需要它。

理解數學原理，包括 Attention、Matrix Multiplication、Softmax、Normalization。

看懂 Transformer 的 PyTorch 實作與 Tensor Shape。

解釋從輸入 Prompt 到生成下一個 Token 的完整運作方式。

分析 GPU Memory、Inference Latency、KV Cache 和長 Context 的效能問題。

在 System Design 面試中提出合理的 Architecture 和 Engineering Trade-offs。

我會把內容分為三個主要部分：

Part I — Intern Level： 用直覺、生活例子理解全部九項知識。

Part II — Engineer Level： 數學公式、架構、訓練原理、程式碼和效能分析。

Part III — 完整整合案例： 把九項技術結合成 LLM 從接收 Prompt、推論到產生回答的完整 Pipeline，並延伸至 Senior 技術面試。

Part I — Intern Level：先理解 LLM 如何運作
1. Transformer：LLM 的核心神經網路架構
Transformer 是一種處理序列資料的 Neural Network Architecture。它由 2017 年論文 
Attention Is All You Need
 提出，是現代 LLM 的主要架構基礎。

arXiv

假設你問 ChatGPT：

小明把手錶放進保險箱，因為它非常珍貴。請問「它」指的是什麼？

模型需要理解「它」比較可能指「手錶」，而不是「保險箱」。

Transformer 最關鍵的能力，就是讓一句話中的不同 Token 相互參考，建立上下文關係。

Transformer 有四個主要元件
Input Tokens

小明 / 把 / 手錶 / 放進 / 保險箱 ...

1. Attention

找出哪些 Token 之間具有重要關係

2. Feed-Forward Network (FFN)

對各 Token 的表示進行非線性特徵轉換

3. Residual Connections + 4. Layer Normalization

維持資訊與梯度流動，改善深層網路的訓練穩定性

輸出更新後的 Token Representations

概念性示意。實際 Transformer Block 中，Residual 與 Normalization 會圍繞 Attention、FFN 配置，而不是只放在最後。
Attention（注意力機制）

就像閱讀文章時，看到「它」會回頭找前面相關的名詞。

Feed-Forward Network（前饋神經網路）

Attention 聚合其他 Token 的資訊，FFN 則在每個位置上利用神經網路轉換特徵，建立更複雜的表示能力。

Residual Connection（殘差連接）

相當於保留原來的資訊，再加上新學到的資訊，避免深層網路很難保留或傳遞有用訊號。

Layer Normalization（層正規化）

對特徵進行標準化，幫助訓練時的數值與梯度保持穩定。

簡單理解：

Transformer 不只是逐字閱讀，而是在一系列神經網路層中，不斷更新每個 Token 對上下文的理解。

2. Self-Attention：Query、Key、Value 到底是什麼？
Self-Attention 是 Transformer 最重要的核心運算之一。

我們可以用搜尋資料庫的方式理解。

假設輸入：

The engineer inspected the watch because it was damaged.

當模型處理 it，需要從上下文中找出與它相關的資訊。

元件

直覺解釋

在例子中的角色

Query（Q）

我現在要找什麼資訊？

it 正在尋找相關上下文

Key（K）

我具有哪些可被匹配的特徵？

各 Token 提供可比對的 Key

Value（V）

我真正提供什麼資訊？

各 Token 提供可聚合的資訊

Attention Score

某兩個位置有多相關？

計算 Query 與 Key 的匹配分數

Attention Weight

應該參考每個位置多少？

把分數轉換成權重

示意：處理 it 時參考哪些詞

watch

58%

engineer

12%

inspected

20%

because

10%

以上只是教學假設的某個 Attention Head 權重，並非實際模型的輸出，也不表示 Attention 權重能直接當成模型解釋。
為什麼叫 Self-Attention？
因為 Q、K、V 都來自同一個輸入序列的表示。

例如：

[The, engineer, inspected, the, watch]

同一個序列中的 Token，可以互相計算 Attention。

但要注意：

Encoder 的 Bidirectional Self-Attention 通常可以讀取整個輸入序列。

生成式 Decoder 的 Causal Self-Attention 不可以讀取尚未生成的未來 Token。

這個區別非常重要，因為它決定了模型能不能執行 Autoregressive Generation。

3. Positional Encoding：模型怎麼知道文字順序？
假設有兩句話：

A. Dog bites man.

B. Man bites dog.

兩句話含有相同的字，但意思完全不同。

Self-Attention 本身的核心運算沒有內建文字排列順序的概念。因此 Transformer 必須引入位置資訊。

常見的兩種方式
Sinusoidal Positional Encoding

使用不同頻率的 sin、cos 數學函數，為各個位置建立不同的數值表示。

例如：

Token 0 有自己的位置向量。

Token 1 有不同的位置向量。

Token 2 又有不同的位置向量。

然後通常將位置向量與 Token Embedding 相加。

RoPE（Rotary Position Embedding）

不直接把位置向量加到 Token Embedding 上，而是根據位置，旋轉 Attention 中 Query 與 Key 的向量。

這種方法讓 Q、K 的內積自然包含相對位置資訊。

RoPE 是很多現代 Decoder-only LLM 所採用的方法之一。

arXiv
+1

RoPE 的直覺：不同位置對應不同旋轉角度


概念示意：此圖只呈現單一二維向量的旋轉，不代表完整 RoPE 的多頻率、高維運算。
Intern 階段最重要的是理解：

Token Embedding 表示 Token 的特徵；Positional Encoding 提供 Token 在序列中的位置資訊。

4. Tokenization：LLM 不是直接讀取文字
我們看到的是：

I love machine learning.

但 LLM 接收的是 Token ID，例如：

[51, 824, 1392, 9921, 13]

以上 ID 只是示意，並非特定 Tokenizer 的真實輸出。

Tokenization 就是把文字轉換成模型可以處理的離散 Token。

Token 不一定等於一個單字
例如：

unbelievable

某個 Tokenizer 可能將它切成：

un + believ + able

另一個 Tokenizer 也可能採用完全不同的切法。

中文更不能簡單假設「一個中文字等於一個 Token」。

BPE（Byte Pair Encoding）
BPE 的基本概念是：

從小單位開始，不斷合併訓練資料中常一起出現的相鄰符號，建立常見子詞。

例如：

初始：
l o w
l o w e r

第一次合併：
lo w
lo w e r

第二次合併：
low
low e r
這只是簡化例子。實際 BPE 包含 Vocabulary 建立、合併排序規則，部分 Tokenizer 以 UTF-8 bytes 為基礎。

Hugging Face

Token Count 為什麼重要？
因為它會影響：

LLM API 使用量與費用

Prompt 是否超過 Context Window

GPU 計算量

KV Cache 大小

Inference Latency

假設一個系統需要處理 100,000 個 Token，和只處理 1,000 個 Token，在成本與延遲上可能有非常大的差異。

5. Model Architecture：Encoder-only、Decoder-only、Encoder-decoder
Transformer 並不是只有一種 Architecture。

Encoder-only

Encoder

Bidirectional

Representations

Decoder-only

Decoder

Causal

Next Token

Encoder-decoder

Encoder

Decoder

架構

代表模型

典型用途

Encoder-only

BERT、RoBERTa

分類、特徵抽取、文字理解

Decoder-only

GPT 系列、Llama

Chat、文字生成、Coding

Encoder-decoder

T5、BART

翻譯、摘要、Text-to-text

需要特別注意：Encoder-only 並不代表只能做分類，Decoder-only 也不代表不能做文字理解。這個分類描述的是主要網路組織方式，不是模型全部能力的限制。

6. Generation：LLM 怎麼挑選下一個 Token？
假設你輸入：

The capital of France is

模型預測下一個 Token 的機率分布。以下是假設結果：

候選 Token

機率

Paris

70%

Lyon

15%

London

10%

Rome

5%

模型可以使用不同的 Decoding Strategies。

Greedy Decoding

永遠選最高機率的 Token。

本例選 Paris。

優點是簡單、可預測；缺點是每一步只顧眼前最高機率，不保證整句話的整體機率最大。

Beam Search

同時保留多條高分候選序列。

它不只選眼前最好的 Token，而是保留多種路徑，降低過早選錯路徑的風險。

Temperature

控制抽樣時的機率分布有多集中。

低 Temperature：分布通常更集中，高機率 Token 更容易被選中。

高 Temperature：分布更平坦，選字變化通常更多。

Top-p（Nucleus Sampling）

依照機率高低挑選一小群 Token，直到累積機率達到指定門檻，再從這個集合抽樣。

例如 Top-p = 0.90，以上四個候選中：

70% + 15% + 10% = 95%

所以前三個 Token 形成候選集合，然後重新正規化機率再抽樣。

關鍵觀念：Temperature 和 Top-p 是如何抽樣的設定，不是讓模型增加知識或提高推理能力的訓練方式。 

Hugging Face

7. Embeddings：如何把文字轉換成有語意的向量？
Embedding 是一組可以由神經網路學習的數值表示。

例如，假設三個句子：

A. The watch is scratched.

B. The timepiece has surface damage.

C. I want to eat pizza.

雖然 A 與 B 使用不同單字，但意思比較接近；C 完全不同。

Embedding Model 可能產生以下概念性結果：

比較

Cosine Similarity（假設值）

A vs. B

0.91

A vs. C

0.08

Embedding 能讓電腦在 Vector Space 中比較文字表示的相似程度。

這對 RAG（Retrieval-Augmented Generation）尤其重要。

假設使用者問：

How do I calibrate the watch inspection camera?

RAG System 可以把問題轉成 Embedding，再到 Vector Database 尋找語意接近的技術文件，將找到的文件交給 LLM 產生回答。

但要區分兩種常被混淆的 Embedding：

Token Embedding： Transformer 內部表示 Token 的向量。

Text/Sentence Embedding： 針對整句、段落或文件訓練或產生的向量，常用於 Semantic Search。

它們可能共用部分模型結構，但使用目的與訓練方式不一定相同。

8. Context Window：LLM 一次可以處理多少資訊？
Context Window 是模型單次處理序列時支援的 Token 範圍。

假設有個模型支援 32,768 Tokens，而你輸入了：

System Prompt：1,000 Tokens

User Prompt：5,000 Tokens

Retrieved Documents：20,000 Tokens

總輸入為 26,000 Tokens。

如果還想生成 4,000 Tokens，總計就是 30,000 Tokens。

在這個簡化例子中，仍位於 32,768 Tokens 的限制內。

但如果輸入本身已經達到 32,000 Tokens，再生成 4,000 Tokens，就可能超過可用上下文容量。

超過 Context Window 怎麼辦？
常見方法包括：

Truncation：刪除部分上下文。

Summarization：摘要較早的內容。

Retrieval：只挑選相關資訊放入 Prompt。

Sliding Window：保留近期資訊並捨棄部分舊內容。

External Memory：將重要資訊存在外部系統，需要時重新檢索。

一個非常重要的觀念是：

Context Window 大，不代表模型可以完美理解其中所有內容，也不代表模型擁有永久記憶。

它只是定義可處理資訊的範圍。實際能否可靠取用不同位置的資訊，需要另外驗證。

9. KV Cache：為什麼 LLM 不需要每次都重新讀完整篇文章？
這個技術對 LLM Inference Engineer 非常重要。

假設輸入：

The watch has

模型準備生成：

a damaged crystal

沒有 KV Cache 時，一個簡化的低效率實作可能這樣處理：

第 1 步：
The watch has
→ 生成 a

第 2 步：
The watch has a
→ 全部重新計算
→ 生成 damaged

第 3 步：
The watch has a damaged
→ 全部重新計算
→ 生成 crystal
使用 KV Cache：

第 1 步：
處理 The watch has
儲存先前 Token 的 K、V

第 2 步：
只處理新增 Token a
重用之前的 K、V
→ 生成 damaged

第 3 步：
只處理新增 Token damaged
重用之前所有 K、V
→ 生成 crystal
KV Cache 會保存各個 Attention Layer 已計算的 Key、Value，使後續 Token 不必反覆計算先前 Token 的這些表示。這是 Autoregressive Inference 的核心加速技術。

GitHub

但是 KV Cache 有一個代價：

它會持續消耗 GPU Memory。

序列越長、同時服務的使用者越多，KV Cache 需要的記憶體通常就越多。

這也是為什麼後來有 GQA、Paged KV Cache、KV Cache Quantization、Sliding-window Cache 等效能設計。

Intern Level：九項知識如何連在一起？
1

User Prompt

使用者輸入文字

2

Tokenization

轉成 Token IDs

3

Token Embeddings

Token IDs 轉成向量

4

Positional Information

加入順序資訊

5

Transformer Blocks

Self-Attention、FFN、Residual、Normalization

6

Output Projection + Softmax

算出下一個 Token 的機率

7

Generation Strategy

挑選下一個 Token

8

KV Cache

重用先前 K、V 加速下一輪

重複生成，直到 EOS 或停止條件

Context Window 限制可處理的總序列長度；模型架構決定 Transformer Blocks 的組織方式。

到這裡，Intern 應該可以解釋：

LLM 為什麼使用 Transformer。

Q、K、V 怎麼建立上下文關係。

Tokenization 與 Embedding 有什麼不同。

模型怎麼挑選下一個 Token。

為什麼較長 Context 會增加成本。

KV Cache 為什麼可以改善生成速度。

接下來進入 Engineer Level，會把以上每一項拆成數學運算與實際程式設計。

Part II — Engineer Level：Transformer 數學、架構與完整技術細節
這部分的目標，是讓你從「知道 Transformer 名詞」進入「可以閱讀模型原始碼、修改 Architecture、分析 GPU 效能」的程度。

先建立工程師必須熟悉的 Tensor Shape
後續會反覆使用這些符號：

符號

意義

範例

𝐵
B

Batch Size

2

𝑇
T

Sequence Length

1,024

𝐷
D

Hidden Dimension / d_model

4,096

𝐻
H

Attention Heads

32

𝑑
ℎ
d 
h
​
 

Head Dimension

128

𝐿
L

Transformer Layers

32

𝑉
V

Vocabulary Size

50,000

𝐷
𝑓
𝑓
D 
ff
​
 

Feed-forward Hidden Dimension

16,384

這些都是教學用的假設參數，並非在描述某個特定的商用模型。

假設輸入一個 Batch，包含兩篇文件，每篇文件有 1,024 個 Token。

在 Token Embedding 後，Tensor Shape 就是：

𝑋
∈
𝑅
𝐵
×
𝑇
×
𝐷
X∈R 
B×T×D
 
也就是：

X.shape = [2, 1024, 4096]

這個 Tensor 會經過多層 Transformer Blocks，不斷產生更新後的 Representation。

1. Transformer 深入：Attention、FFN、Residual、LayerNorm
1.1 Transformer Block 完整架構


2017 年原始 Transformer 的 Encoder–Decoder 架構。左側 Encoder 包含 Self-Attention 與 Feed-forward，右側 Decoder 包含 Masked Self-Attention、Cross-Attention 與 Feed-forward。這不是現代 Decoder-only GPT 的完整架構圖。
原始 Transformer 使用 Post-LayerNorm；許多後續 LLM 則採用 Pre-Norm 或 RMSNorm 等變體。

NeurIPS Papers

以現代常見的 Pre-Norm Decoder Block 為例：

Input X
   |
   +--------------------+
   |                    |
   v                    |
Normalization           |
   |                    |
   v                    |
Multi-Head Attention    |
   |                    |
   +---- Add <----------+
          |
          v
          X1
          |
          +--------------------+
          |                    |
          v                    |
      Normalization            |
          |                    |
          v                    |
      Feed-forward / MLP       |
          |                    |
          +---- Add <----------+
                 |
                 v
               Output
可以用兩組數學式表示：

𝑋
1
=
𝑋
+
MHA
⁡
(
Norm
⁡
(
𝑋
)
)
X 
1
​
 =X+MHA(Norm(X))
𝑌
=
𝑋
1
+
FFN
⁡
(
Norm
⁡
(
𝑋
1
)
)
Y=X 
1
​
 +FFN(Norm(X 
1
​
 ))
其中：

MHA：Multi-Head Attention

FFN：Feed-Forward Network

Norm：LayerNorm 或 RMSNorm 等正規化

𝑋
1
X 
1
​
 ：Attention 更新後的表示

𝑌
Y：該 Transformer Block 的輸出

工程上的重要區分：Pre-Norm vs. Post-Norm
Post-Norm：

𝑌
=
LayerNorm
⁡
(
𝑋
+
Sublayer
⁡
(
𝑋
)
)
Y=LayerNorm(X+Sublayer(X))
Pre-Norm：

𝑌
=
𝑋
+
Sublayer
⁡
(
Norm
⁡
(
𝑋
)
)
Y=X+Sublayer(Norm(X))
Pre-Norm 的殘差通道提供較直接的梯度傳播路徑，通常有助於深層 Transformer 的訓練穩定性。

不過兩者不是完全等價，也不能說 Pre-Norm 對任何模型都一定比較好。

1.2 Feed-Forward Network 如何處理資訊？
Attention 負責從序列其他位置聚合資訊。

FFN 則對每個 Token 的特徵向量做非線性轉換。

最基本形式：

FFN
⁡
(
𝑥
)
=
𝑊
2
𝜙
(
𝑊
1
𝑥
+
𝑏
1
)
+
𝑏
2
FFN(x)=W 
2
​
 ϕ(W 
1
​
 x+b 
1
​
 )+b 
2
​
 
假設：

𝐷
=
4096
,
𝐷
𝑓
𝑓
=
16384
D=4096,D 
ff
​
 =16384
則每個 Token 的表示：

4096 dimensions
       |
       v
Linear 4096 → 16384
       |
       v
GELU / ReLU
       |
       v
Linear 16384 → 4096
       |
       v
4096 dimensions
FFN 對不同 Token 使用相同的一組權重，但每個 Token 都有自己的輸入向量。

特別注意：

標準 Position-wise FFN 本身不負責跨 Token 的資訊交換。 跨 Token 資訊主要由 Attention 聚合進來，再由 FFN 轉換。

現代 LLM：SwiGLU
許多現代模型使用 Gated Feed-Forward，例如 SwiGLU：

SwiGLU
⁡
(
𝑥
)
=
𝑊
down
[
SiLU
⁡
(
𝑊
gate
𝑥
)
⊙
𝑊
up
𝑥
]
SwiGLU(x)=W 
down
​
 [SiLU(W 
gate
​
 x)⊙W 
up
​
 x]
其中 
⊙
⊙ 是 Element-wise Multiplication。

它有兩條中間投影路徑，其中一條使用 SiLU 產生 gating 效果，再與另一條相乘。

這會改變 FFN 的表達方式，也影響參數量、計算量與實際效能。

1.3 Residual Connection 的數學意義
Residual Connection：

𝑦
=
𝑥
+
𝐹
(
𝑥
)
y=x+F(x)
其中 
𝐹
(
𝑥
)
F(x) 是 Attention 或 FFN 等子網路運算。

對輸入求導：

∂
𝑦
∂
𝑥
=
𝐼
+
∂
𝐹
(
𝑥
)
∂
𝑥
∂x
∂y
​
 =I+ 
∂x
∂F(x)
​
 
這個式子解釋了為什麼 Residual Connection 對深層網路很有幫助。

梯度除了經過複雜的 
𝐹
(
𝑥
)
F(x)，還有直接傳回的 Identity Path。

例如一個模型有 64 層 Transformer Blocks，如果每一層都必須穿過多次複雜非線性轉換，梯度傳播會更加困難。

Residual Connection 可以緩解這個問題，但不保證完全消除梯度消失或梯度爆炸。

1.4 LayerNorm 的數學意義
假設某個 Token 的特徵向量為：

𝑥
=
[
𝑥
1
,
𝑥
2
,
…
,
𝑥
𝐷
]
x=[x 
1
​
 ,x 
2
​
 ,…,x 
D
​
 ]
計算平均值：

𝜇
=
1
𝐷
∑
𝑖
=
1
𝐷
𝑥
𝑖
μ= 
D
1
​
  
i=1
∑
D
​
 x 
i
​
 
計算變異數：

𝜎
2
=
1
𝐷
∑
𝑖
=
1
𝐷
(
𝑥
𝑖
−
𝜇
)
2
σ 
2
 = 
D
1
​
  
i=1
∑
D
​
 (x 
i
​
 −μ) 
2
 
正規化：

LayerNorm
⁡
(
𝑥
)
=
𝛾
𝑥
−
𝜇
𝜎
2
+
𝜖
+
𝛽
LayerNorm(x)=γ 
σ 
2
 +ϵ
​
 
x−μ
​
 +β
其中 
𝛾
,
𝛽
γ,β 是可訓練參數。

LayerNorm 通常對每個 Token 的 Hidden Features 進行正規化，不像 BatchNorm 需要依賴整個 Batch 的統計量。

RMSNorm
很多現代 Decoder-only 模型則使用 RMSNorm：

RMSNorm
⁡
(
𝑥
)
=
𝛾
𝑥
1
𝐷
∑
𝑖
=
1
𝐷
𝑥
𝑖
2
+
𝜖
RMSNorm(x)=γ 
D
1
​
 ∑ 
i=1
D
​
 x 
i
2
​
 +ϵ
​
 
x
​
 
它不需要先減去平均值，也通常不使用 LayerNorm 的加性 bias 項。

面試重點： 能說清楚 Attention、FFN、Residual 和 Normalization 各自的功能，以及為什麼深層 Transformer 需要它們。

2. Self-Attention 深入：從 Q、K、V 到 Scaled Dot-product
這是整份內容最應該深入掌握的數學部分。

2.1 Q、K、V 是如何產生的？
假設輸入：

𝑋
∈
𝑅
𝑇
×
𝐷
X∈R 
T×D
 
首先經過三個可學習的 Linear Projections：

𝑄
=
𝑋
𝑊
𝑄
Q=XW 
Q
​
 
𝐾
=
𝑋
𝑊
𝐾
K=XW 
K
​
 
𝑉
=
𝑋
𝑊
𝑉
V=XW 
V
​
 
其中 
𝑊
𝑄
,
𝑊
𝐾
,
𝑊
𝑉
W 
Q
​
 ,W 
K
​
 ,W 
V
​
  是 Training 過程學習的權重矩陣。

要注意：

Q、K、V 不是三組由工程師手動設定的規則，也不是三種固定的文字標籤。

它們都是神經網路從輸入 Representation 轉換得到的向量。

2.2 Scaled Dot-product Attention 公式
完整公式：

Attention
⁡
(
𝑄
,
𝐾
,
𝑉
)
=
softmax
⁡
(
𝑄
𝐾
𝑇
𝑑
𝑘
+
𝑀
)
𝑉
Attention(Q,K,V)=softmax( 
d 
k
​
 
​
 
QK 
T
 
​
 +M)V
​
 
其中：

𝑄
𝐾
𝑇
QK 
T
 ：計算 Query 與 Key 的匹配分數

𝑑
𝑘
d 
k
​
 
​
 ：Scaling Factor

𝑀
M：可選的 Attention Mask / Bias

Softmax：轉換成權重

乘上 
𝑉
V：得到加權後的資訊表示

2.3 用真實數值計算一遍
假設目前只有一個 Query：

𝑄
=
[
1
,
0
]
Q=[1,0]
三個 Keys：

𝐾
=
[
1
0
0
1
1
1
]
K= 
​
  
1
0
1
​
  
0
1
1
​
  
​
 
Values：

𝑉
=
[
10
0
0
10
10
10
]
V= 
​
  
10
0
10
​
  
0
10
10
​
  
​
 
Step 1：計算 Dot Products

𝑄
𝐾
𝑇
=
[
1
,
0
,
1
]
QK 
T
 =[1,0,1]
Step 2：除以 
𝑑
𝑘
d 
k
​
 
​
 

這裡 
𝑑
𝑘
=
2
d 
k
​
 =2。

𝑄
𝐾
𝑇
2
=
[
0.7071
,
0
,
0.7071
]
2
​
 
QK 
T
 
​
 =[0.7071,0,0.7071]
Step 3：執行 Softmax

𝛼
=
softmax
⁡
(
[
0.7071
,
0
,
0.7071
]
)
α=softmax([0.7071,0,0.7071])
得到：

𝛼
≈
[
0.4011
,
0.1978
,
0.4011
]
α≈[0.4011,0.1978,0.4011]
這個 Query 對三個 Keys 的 Attention Weight

Key 1

40.11%

Key 2

19.78%

Key 3

40.11%

Step 4：對 Values 加權求和

Output
=
𝛼
𝑉
Output=αV
=
0.4011
[
10
,
0
]
+
0.1978
[
0
,
10
]
+
0.4011
[
10
,
10
]
=0.4011[10,0]+0.1978[0,10]+0.4011[10,10]
Output
≈
[
8.022
,
5.989
]
Output≈[8.022,5.989]
​
 
這就是 Attention 最核心的運算。

它不是直接複製某個 Value，而是依照 Query–Key 的相關性，把不同 Value 混合成新 Representation。

2.4 為什麼需要除以 
𝑑
𝑘
d 
k
​
 
​
 ？
假設 Q、K 各維度的數值具有適當的獨立性與單位變異數。

它們的 Dot Product：

𝑞
⋅
𝑘
=
∑
𝑖
=
1
𝑑
𝑘
𝑞
𝑖
𝑘
𝑖
q⋅k= 
i=1
∑
d 
k
​
 
​
 q 
i
​
 k 
i
​
 
其變異數會大致隨 
𝑑
𝑘
d 
k
​
  增長。

如果不 Scaling，當 
𝑑
𝑘
d 
k
​
  很大，Attention Logits 的數值可能很極端。

Softmax 容易變得非常尖銳：

[
0.9999
,
0.0001
,
…
]
[0.9999,0.0001,…]
這可能使梯度變得很小，讓訓練更困難。

Scaling by 
𝑑
𝑘
d 
k
​
 
​
  有助於控制 Logit 的尺度。

2.5 Multi-Head Attention：為什麼需要多個 Heads？
一個 Attention Head 不一定能有效捕捉各種不同關係。

因此 Transformer 將 Hidden Dimension 分成多個 Heads。

假設：

𝐷
=
4096
,
𝐻
=
32
D=4096,H=32
則：

𝑑
ℎ
=
4096
32
=
128
d 
h
​
 = 
32
4096
​
 =128
Tensor Shape：

Input:
[B, T, 4096]

Q Projection:
[B, T, 4096]

Reshape:
[B, T, 32, 128]

Transpose:
[B, 32, T, 128]
Attention 計算：

Q: [B, 32, T, 128]
K: [B, 32, T, 128]
V: [B, 32, T, 128]

Q @ K.transpose:
   [B, 32, T, T]

Attention @ V:
   [B, 32, T, 128]
最後：

Concatenate Heads:
[B, T, 4096]

Output Projection:
[B, T, 4096]
數學表示：

head
⁡
𝑖
=
Attention
⁡
(
𝑄
𝑖
,
𝐾
𝑖
,
𝑉
𝑖
)
head 
i
​
 =Attention(Q 
i
​
 ,K 
i
​
 ,V 
i
​
 )
MHA
⁡
(
𝑋
)
=
Concat
⁡
(
head
⁡
1
,
…
,
head
⁡
𝐻
)
𝑊
𝑂
MHA(X)=Concat(head 
1
​
 ,…,head 
H
​
 )W 
O
​
 
不同 Heads 可以學出不同的資訊聚合模式，例如偏重局部語法、遠距依賴或其他特徵，但不能保證每個 Head 對應一個可直接命名的語言功能。

2.6 Causal Mask：為什麼模型不會偷看未來？
Decoder-only LLM 需要預測下一個 Token。

例如：

Input:
I love machine learning
當模型處理 love 的位置，不能看到它右邊尚未允許使用的 Token。

因此 Attention Mask 類似：

Causal Mask — 允許讀取的 Token 位置

Q \ K

I

love

machine

learning

I

love

machine

learning

列是目前 Query，欄是可讀取的 Key。下三角位置允許 Attention，上三角位置被 Mask。
工程上通常把不允許的 Attention Logits 設為 
−
∞
−∞，讓 Softmax 後的權重接近或等於零。

Attention 計算複雜度
對長度 
𝑇
T 的序列，完整 Self-Attention 的 Score Matrix 是：

𝑇
×
𝑇
T×T
因此計算複雜度大致為：

𝑂
(
𝑇
2
𝐷
)
O(T 
2
 D)
若序列從 1,024 Tokens 增加到 4,096 Tokens，也就是增加四倍，該 Attention 核心計算量在其他條件相同時會增加約 16 倍。

但整個 Transformer 的耗時不會必然增加 16 倍，因為還有 FFN、Linear Projections、GPU Kernel、Memory Access 等其他因素。

3. Positional Encoding 深入：Sinusoidal 與 RoPE
3.1 為什麼普通 Self-Attention 不知道順序？
考慮只有一個 Self-Attention Layer，且沒有 Positional Information。

對輸入 Token 重新排列，Attention 的計算結果也會隨相同的排列重新排列。

這種特性稱為 Permutation Equivariance。

所以單純的 Self-Attention 並不能像我們閱讀句子一樣，天然區分某個 Token 位於序列第 1 個還是第 100 個。

Decoder-only 模型的 Causal Mask 雖然本身也引入了序列方向結構，但仍通常需要明確的位置編碼來表示更完整的距離及順序資訊。

3.2 Sinusoidal Positional Encoding
原始 Transformer 採用：

𝑃
𝐸
(
𝑝
𝑜
𝑠
,
2
𝑖
)
=
sin
⁡
(
𝑝
𝑜
𝑠
10000
2
𝑖
/
𝐷
)
PE(pos,2i)=sin( 
10000 
2i/D
 
pos
​
 )
𝑃
𝐸
(
𝑝
𝑜
𝑠
,
2
𝑖
+
1
)
=
cos
⁡
(
𝑝
𝑜
𝑠
10000
2
𝑖
/
𝐷
)
PE(pos,2i+1)=cos( 
10000 
2i/D
 
pos
​
 )
其中：

𝑝
𝑜
𝑠
pos：Token Position

𝑖
i：Feature Dimension Pair 的索引

𝐷
D：Embedding Dimension

不同維度使用不同的頻率。

模型可以從這些不同頻率的週期訊號中，取得位置與距離方面的資訊。

原始實作：

𝑋
input
=
𝐸
token
+
𝑃
𝐸
X 
input
​
 =E 
token
​
 +PE
其中 
𝐸
token
E 
token
​
  是 Token Embedding。

3.3 RoPE 的數學原理
RoPE 將 Q、K 的部分維度按照二維向量組合，然後執行旋轉。

對一個二維向量：

𝑥
=
[
𝑥
1
𝑥
2
]
x=[ 
x 
1
​
 
x 
2
​
 
​
 ]
旋轉矩陣：

𝑅
(
𝜃
)
=
[
cos
⁡
𝜃
−
sin
⁡
𝜃
sin
⁡
𝜃
cos
⁡
𝜃
]
R(θ)=[ 
cosθ
sinθ
​
  
−sinθ
cosθ
​
 ]
則：

𝑥
′
=
𝑅
(
𝜃
)
𝑥
x 
′
 =R(θ)x
對處於 Position 
𝑚
m 的 Query：

𝑞
𝑚
′
=
𝑅
𝑚
𝑞
𝑚
q 
m
′
​
 =R 
m
​
 q 
m
​
 
對 Position 
𝑛
n 的 Key：

𝑘
𝑛
′
=
𝑅
𝑛
𝑘
𝑛
k 
n
′
​
 =R 
n
​
 k 
n
​
 
Attention Score：

(
𝑞
𝑚
′
)
𝑇
𝑘
𝑛
′
=
𝑞
𝑚
𝑇
𝑅
𝑚
𝑇
𝑅
𝑛
𝑘
𝑛
(q 
m
′
​
 ) 
T
 k 
n
′
​
 =q 
m
T
​
 R 
m
T
​
 R 
n
​
 k 
n
​
 
因為旋轉矩陣的性質：

𝑅
𝑚
𝑇
𝑅
𝑛
=
𝑅
𝑛
−
𝑚
R 
m
T
​
 R 
n
​
 =R 
n−m
​
 
因此：

(
𝑞
𝑚
′
)
𝑇
𝑘
𝑛
′
=
𝑞
𝑚
𝑇
𝑅
𝑛
−
𝑚
𝑘
𝑛
(q 
m
′
​
 ) 
T
 k 
n
′
​
 =q 
m
T
​
 R 
n−m
​
 k 
n
​
 
​
 
這就是 RoPE 最重要的數學特性：Attention Dot Product 自然包含兩個 Token 的相對位置差 
𝑛
−
𝑚
n−m。

完整 RoPE 並不是只旋轉一組二維向量，而是在多個二維子空間使用不同頻率進行旋轉。

RoPE 的優點與限制
優點：

將位置資訊融入 Attention。

相對位置資訊自然進入 Q–K 內積。

不需要為所有位置各自學習一個獨立 Position Embedding。

能搭配多種 Attention Architecture。

限制：

RoPE 並不保證模型在訓練長度之外仍維持相同品質。

Long-context Extension 通常還需要 Scaling Strategy、額外訓練或適當評估。

不同 RoPE Scaling 方法可能在短 Context 和長 Context 之間產生 Trade-off。

Senior Engineer 追問
如果使用 KV Cache，RoPE 應該在什麼時候套用？

典型方法是：

目前 Token 產生 Q、K。

根據 Token 的實際 Position ID 套用 RoPE。

將旋轉後的 K 存入 KV Cache。

之後直接重用該 K，不應在每次讀取時又重複旋轉。

最常見 Bug 是 Incremental Decoding 時，把新 Token 的 Position 又從 0 開始，造成 Full Forward 與 Cached Forward 的輸出不一致。

4. Tokenization 深入：BPE、Vocabulary、Token Count
4.1 BPE 的訓練流程
假設訓練語料：

low
lower
lowest
new
newer
一個簡化的 BPE 訓練程序：

Step 1：初始化基本符號

建立字符或 Byte 級別的初始 Vocabulary。

Step 2：統計相鄰 Pair

假設：

l + o 出現很多次
o + w 出現很多次
e + r 出現很多次
Step 3：合併最常出現的 Pair

例如：

l + o → lo
lo + w → low
Step 4：更新 Vocabulary

增加：

lo
low
Step 5：重複直到 Vocabulary 達到目標規模

最後保存 Merge Rules，供 Inference Tokenization 使用。

實際 BPE 可能還包含 Normalization、Pre-tokenization、Byte-Level Encoding 和特殊 Token 處理。

4.2 Vocabulary Size 是什麼？
假設：

𝑉
=
50
,
000
V=50,000
代表模型 Vocabulary 中有約 50,000 個 Token IDs。

Token ID 會透過 Embedding Matrix 轉成向量：

𝐸
∈
𝑅
𝑉
×
𝐷
E∈R 
V×D
 
例如：

𝐸
∈
𝑅
50000
×
4096
E∈R 
50000×4096
 
代表這個 Embedding Matrix 有：

50000
×
4096
=
204
,
800
,
000
50000×4096=204,800,000
個參數。

如果每個權重使用 FP16（2 Bytes）：

204
,
800
,
000
×
2
=
409
,
600
,
000
 Bytes
204,800,000×2=409,600,000 Bytes
約 391 MiB。

這只是 Input Token Embedding 的大小，不包含其他 Transformer Layers。

有些模型會讓 Output Projection 與 Input Embedding 共用權重（Weight Tying）；若不共用，Output Projection 又可能增加一組相當可觀的參數。

4.3 Tokenizer 與 Model 必須匹配
不能隨意把 Model A 的 Tokenizer 換成 Model B 的 Tokenizer。

因為：

Token IDs 可能對應不同文字。

Vocabulary Size 可能不同。

特殊 Token ID 不同。

Chat Template 可能不同。

Embedding Matrix 是跟特定 Token Mapping 一起訓練的。

實際部署時，Tokenizer、Model Weights、Special Tokens、Chat Template 都應該有一致的版本管理。

4.4 實際查看 Tokenization
以下使用 Hugging Face Transformers：

from transformers import AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

text = "The watch has a scratched crystal."

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(
    text,
    add_special_tokens=False
)

print("Tokens:", tokens)
print("IDs:", token_ids)
print("Token count:", len(token_ids))

這段程式可以真實檢查某個模型如何分詞，而不是用字數或英文單字數猜測 Token Count。

5. Model Architecture 深入：三類 Transformer 的差異
5.1 Encoder-only
典型架構：

Input Tokens
      |
      v
Embedding + Position
      |
      v
Bidirectional Self-Attention
      |
      v
FFN / Normalization
      |
      v
Contextual Representations
      |
      v
Task Head / Pooling
核心特徵：

某個 Token 通常可以參考序列左邊和右邊的內容。

例如：

The bank approved the loan.

bank 可以同時參考 approved 與 loan，幫助模型理解這是金融機構，而不是河岸。

BERT 的典型訓練方式
Masked Language Modeling（MLM）：

The bank [MASK] the loan.
模型需要預測：

approved

Encoder-only 很適合學習雙向 Contextual Representations。

5.2 Decoder-only
核心是 Causal Self-Attention。

Token 1 → Predict Token 2
Token 1,2 → Predict Token 3
Token 1,2,3 → Predict Token 4
Autoregressive Probability：

𝑃
(
𝑥
1
,
…
,
𝑥
𝑇
)
=
∏
𝑡
=
1
𝑇
𝑃
(
𝑥
𝑡
∣
𝑥
<
𝑡
)
P(x 
1
​
 ,…,x 
T
​
 )= 
t=1
∏
T
​
 P(x 
t
​
 ∣x 
<t
​
 )
這是 GPT 類模型的重要訓練與生成基礎。

Next-token Prediction Loss
假設輸入 Token IDs：

[BOS, The, watch, is, damaged, EOS]
Training 的 Input 與 Target：

Input:
[BOS, The, watch, is, damaged]

Target:
[The, watch, is, damaged, EOS]
模型預測：

𝑃
𝜃
(
𝑥
𝑡
∣
𝑥
<
𝑡
)
P 
θ
​
 (x 
t
​
 ∣x 
<t
​
 )
Cross-Entropy Loss：

𝐿
=
−
∑
𝑡
log
⁡
𝑃
𝜃
(
𝑥
𝑡
∣
𝑥
<
𝑡
)
L=− 
t
∑
​
 logP 
θ
​
 (x 
t
​
 ∣x 
<t
​
 )
實際上通常會再依有效 Target Tokens 做平均，並透過 Loss Mask 忽略 Padding 或不應計入 Loss 的位置。

Teacher Forcing
Training 時，通常直接提供整段已知 Token 序列，再利用 Causal Mask 並行計算各位置的 Next-token Prediction。

也就是說：

Training 並不需要像 Inference 一樣，每生成一個 Token 就執行一次完整 Forward。

這是 Transformer 能利用 GPU 平行訓練的重要原因之一。

但 Inference 時，新 Token 取決於前一個 Token 的實際生成結果，因此 Autoregressive Decoding 仍是順序性的。

5.3 Encoder-decoder
常見於 Translation、Summarization 等 Seq2Seq 任務。

Source:
Translate English to French:
"The watch is damaged."

          |
          v
       Encoder
          |
          v
   Encoded Source States
          |
          v
        Decoder
   - Causal Self-Attention
   - Cross-Attention
          |
          v
"La montre est endommagée."
Cross-Attention 的核心差異：

𝑄
=
Decoder Hidden States
Q=Decoder Hidden States
𝐾
,
𝑉
=
Encoder Outputs
K,V=Encoder Outputs
Decoder 使用自己的 Query，去讀取 Encoder 產生的 Source Representations。

這與 Self-Attention（Q、K、V 來自同一序列）不同。

Senior Engineer 如何選擇？
工作需求

通常考慮的架構

原因

Sentence Classification

Encoder-only

有效建立輸入序列表示

Embedding Retrieval

Encoder-based Embedding Model

適合產生檢索向量

Chatbot

Decoder-only

天然支援 Autoregressive Generation

Coding Assistant

Decoder-only

適合接續生成程式碼

Machine Translation

Encoder-decoder 或 Decoder-only

視模型、資料與部署條件選擇

Text Summarization

Encoder-decoder 或 Decoder-only

兩者皆可實現

不要把架構與任務做成過度僵硬的一對一對應。

6. Generation 深入：Greedy、Beam Search、Temperature、Top-p
6.1 從 Hidden State 產生 Token Probability
假設最後一個 Transformer Layer 輸出：

ℎ
𝑡
∈
𝑅
𝐷
h 
t
​
 ∈R 
D
 
經過 LM Head：

𝑧
𝑡
=
𝑊
vocab
ℎ
𝑡
+
𝑏
z 
t
​
 =W 
vocab
​
 h 
t
​
 +b
得到 Logits：

𝑧
𝑡
∈
𝑅
𝑉
z 
t
​
 ∈R 
V
 
然後：

𝑃
(
𝑥
𝑡
+
1
=
𝑖
)
=
𝑒
𝑧
𝑖
∑
𝑗
𝑒
𝑧
𝑗
P(x 
t+1
​
 =i)= 
∑ 
j
​
 e 
z 
j
​
 
 
e 
z 
i
​
 
 
​
 
這就是下一個 Token 的 Probability Distribution。

6.2 Greedy Decoding
𝑥
𝑡
+
1
=
arg
⁡
max
⁡
𝑖
𝑃
(
𝑖
∣
𝑥
≤
𝑡
)
x 
t+1
​
 =arg 
i
max
​
 P(i∣x 
≤t
​
 )
優點：

不需要抽樣。

計算與實作相對簡單。

適合需要較穩定輸出的某些任務。

缺點是只做局部最佳選擇。

例如第一步：

Token

Probability

A

0.60

B

0.40

Greedy 選 A。

假設下一步最高機率分別是：

𝑃
(
𝐴
next
∣
𝐴
)
=
0.50
P(A 
next
​
 ∣A)=0.50
𝑃
(
𝐵
next
∣
𝐵
)
=
0.90
P(B 
next
​
 ∣B)=0.90
兩條路徑的 Joint Probability：

𝑃
(
𝐴
,
𝐴
next
)
=
0.60
×
0.50
=
0.30
P(A,A 
next
​
 )=0.60×0.50=0.30
𝑃
(
𝐵
,
𝐵
next
)
=
0.40
×
0.90
=
0.36
P(B,B 
next
​
 )=0.40×0.90=0.36
所以局部最好的 A 路徑，不一定是整體最佳。

6.3 Beam Search
Beam Search 保留 
𝐾
K 條候選序列。

例如：

𝐾
=
3
K=3
每一輪都擴充候選，再依序列 Score 保留較好的三條。

常見分數基礎：

log
⁡
𝑃
(
𝑦
∣
𝑥
)
=
∑
𝑡
log
⁡
𝑃
(
𝑦
𝑡
∣
𝑦
<
𝑡
,
𝑥
)
logP(y∣x)= 
t
∑
​
 logP(y 
t
​
 ∣y 
<t
​
 ,x)
實作中可能加入 Length Penalty 或其他調整，避免過度偏好短序列。

但 Beam Search 不保證找到全域最佳序列，也不保證答案事實上正確。

6.4 Temperature
對 Logits 進行：

𝑃
𝑖
(
𝑇
)
=
𝑒
𝑧
𝑖
/
𝑇
∑
𝑗
𝑒
𝑧
𝑗
/
𝑇
P 
i
​
 (T)= 
∑ 
j
​
 e 
z 
j
​
 /T
 
e 
z 
i
​
 /T
 
​
 
例如：

𝑧
=
[
3
,
2
,
1
]
z=[3,2,1]
Temperature 對 Token Distribution 的影響

T = 1.0

Token A

66.5%

Token B

24.5%

Token C

9.0%

調整 Temperature，可以看到 Logits 不變，但 Softmax 後的 Probability Distribution 改變。這是實際公式計算的互動例子。
當 Temperature 趨近 0，Softmax 會越來越集中於最大 Logit。實際系統常把 Temperature = 0 視為選擇 Greedy Decoding 的特殊設定，而不是直接計算除以 0。

6.5 Top-p
Top-p 不直接改變所有 Token 的相對機率，而是先選候選集合。

假設已排序機率：

[
0.50
,
0.25
,
0.15
,
0.07
,
0.03
]
[0.50,0.25,0.15,0.07,0.03]
當：

𝑝
=
0.90
p=0.90
前三個 Token 累積剛好：

0.50
+
0.25
+
0.15
=
0.90
0.50+0.25+0.15=0.90
保留前三個後重新 Normalization：

[
0.556
,
0.278
,
0.167
]
[0.556,0.278,0.167]
然後從這三個 Token 中抽樣。

注意：

Temperature、Top-p 可以組合使用。

參數作用順序會影響實際分布。

do_sample=False 時，常見實作不會真正執行隨機 Sampling。

Logit Bias、Repetition Penalty、Allowed-token Mask 等處理也可能影響最終選擇。

實際 Hugging Face Generation
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model.eval()

inputs = tokenizer(
    "Explain what an attention head does:",
    return_tensors="pt"
)

# Greedy
output = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False
)

# Temperature + Top-p
sampled = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

# Beam Search
beams = model.generate(
    **inputs,
    max_new_tokens=100,
    num_beams=4,
    do_sample=False
)

這個例子使用普通模型載入方式，適合學習不同 Generation Parameters 的差異；Production 還要處理 Device Placement、Batching、Stopping、Memory Usage 與 Serving Engine 等問題。

Hugging Face

7. Embeddings 深入：Representation、Similarity 與 Training
7.1 Token Embedding Matrix
假設：

𝑉
=
50000
,
𝐷
=
4096
V=50000,D=4096
則：

𝐸
∈
𝑅
50000
×
4096
E∈R 
50000×4096
 
輸入：

token_ids = [15, 328, 2201]

透過 Embedding Lookup：

embeddings = E[token_ids]

結果：

[3, 4096]
這是三個 Token 的初始 Representation。

但進入 Transformer 後，每個 Token 的 Hidden State 會受到上下文影響。

因此應區分：

Static Token Embedding：初始 Token Lookup。

Contextual Representation：Transformer 各層輸出的上下文相關向量。

Sentence Embedding：經過專門設計或 Pooling，用來表示整句語意的向量。

7.2 Cosine Similarity
假設：

𝑎
=
[
1
,
2
,
3
]
a=[1,2,3]
𝑏
=
[
2
,
4
,
6
]
b=[2,4,6]
Cosine Similarity：

cos
⁡
(
𝑎
,
𝑏
)
=
𝑎
⋅
𝑏
∥
𝑎
∥
∥
𝑏
∥
cos(a,b)= 
∥a∥∥b∥
a⋅b
​
 
因為 b 是 a 的兩倍：

cos
⁡
(
𝑎
,
𝑏
)
=
1
cos(a,b)=1
代表方向相同。

要注意這只表示幾何方向一致，不代表兩段自然語言在任何情境下完全等價。

7.3 Semantic Embedding 如何訓練？
常見方法之一是 Contrastive Learning。

假設：

Query：How to calibrate a camera?

Positive Document：Camera calibration procedure.

Negative Document：How to clean the watch bracelet?

Embedding Model 分別產生：

𝑞
,
 
𝑑
+
,
 
𝑑
−
q, d 
+
 , d 
−
 
訓練目標是讓：

sim
⁡
(
𝑞
,
𝑑
+
)
>
sim
⁡
(
𝑞
,
𝑑
−
)
sim(q,d 
+
 )>sim(q,d 
−
 )
常見 InfoNCE / Contrastive Loss 形式：

𝐿
=
−
log
⁡
exp
⁡
(
sim
⁡
(
𝑞
,
𝑑
+
)
/
𝜏
)
∑
𝑗
exp
⁡
(
sim
⁡
(
𝑞
,
𝑑
𝑗
)
/
𝜏
)
L=−log 
∑ 
j
​
 exp(sim(q,d 
j
​
 )/τ)
exp(sim(q,d 
+
 )/τ)
​
 
其中：

𝑑
+
d 
+
 ：正樣本

𝑑
𝑗
d 
j
​
 ：候選文件，包括負樣本

𝜏
τ：Contrastive Temperature

這裡的 Temperature 是訓練 Loss 的超參數，與前面 Generation Sampling 的 Temperature 不應混為一談。

Embedding 與 RAG 的實際關係
Documents
    |
    v
Chunking
    |
    v
Embedding Model
    |
    v
Document Vectors
    |
    v
Vector Index
查詢時：

User Question
    |
    v
Query Embedding
    |
    v
Vector Search
    |
    v
Top-k Documents
    |
    v
LLM Generation
對 Senior Engineer 而言，還應考慮：

Chunk Size 與 Overlap

Metadata Filters

Hybrid Search（Lexical + Vector）

Reranking

Retrieval Recall@k

文件更新與 Embedding Versioning

Domain-specific Evaluation

一個重要的系統設計觀念：

Embedding 的相似度高，不等於文件就是正確答案。

例如相似度非常高的技術文件，可能是舊版規格、不同設備或錯誤流程。

因此 Production RAG 還需要 Metadata、版本限制、權限過濾與 Grounding Evaluation。

8. Context Window 深入：Long Context、Truncation、Memory
8.1 Context Window 到底限制什麼？
對 Decoder-only LLM，Context Window 通常限制模型在一次推論過程中可以保留或注意到的 Token 數量。

假設：

𝐶
max
⁡
=
32768
C 
max
​
 =32768
若使用的是完整保留歷史的 Causal Attention，通常要滿足：

𝑇
prompt
+
𝑇
generated
≤
𝐶
max
⁡
T 
prompt
​
 +T 
generated
​
 ≤C 
max
​
 
但實際 API 也可能另外限制 Maximum Output Tokens，因此不能只用一條公式判斷所有商用服務的上限。

Context Window 與 Attention Computation 的關係
前面介紹：

Attention
⁡
(
𝑄
,
𝐾
,
𝑉
)
=
softmax
⁡
(
𝑄
𝐾
𝑇
𝑑
𝑘
)
𝑉
Attention(Q,K,V)=softmax( 
d 
k
​
 
​
 
QK 
T
 
​
 )V
對 Full Attention：

𝑄
𝐾
𝑇
∈
𝑅
𝑇
×
𝑇
QK 
T
 ∈R 
T×T
 
因此 Prompt Length 增加時，Attention 的計算和記憶體需求都可能大幅提高。

完整 Attention Score Matrix 的元素數量

Full attention score matrix elements across different sequence lengths; excludes batch and number of heads.

0M
300M
600M
900M
1.2KM
1K
4K
8K
16K
32K
1K: 元素數量 1.1M
4K: 元素數量 16.8M
8K: 元素數量 67.1M
16K: 元素數量 268.4M
32K: 元素數量 1.1KM
僅顯示單一 Head 的 T×T Score Matrix 規模。若直接 materialize Attention Matrix，還需要乘上 Batch Size、Head 數量和每個元素的 Bytes；FlashAttention 等實作可避免完整儲存此矩陣。
8.2 Long Context 常見的三種問題
問題 A：計算量
例如某模型必須處理 128K Tokens。

相較於 8K Tokens，序列長度增加 16 倍。

標準 Full Attention 核心計算量可能增加：

16
2
=
256
16 
2
 =256
倍。

這不表示整個模型的延遲必然變為 256 倍，因為整體還受模型大小、Linear/FFN、硬體和 Attention Kernel 影響。

問題 B：模型可能無法可靠利用全部 Context
即使模型技術上支援 128K Tokens，也不代表它能精確找出任何位置的資訊。

例如把重要資料分別放在：

文件開頭

文件中間

文件結尾

模型在不同位置的檢索準確率可能不同。

這類現象曾被稱為 Lost in the Middle。

Hugging Face

因此 Senior Engineer 不能只看官方宣稱的 Maximum Context Length。

還要實測：

Needle-in-a-Haystack Retrieval

多文件 Question Answering

跨文件資訊整合

不同 Position 的 Retrieval Accuracy

Long-context Latency 與 Memory

問題 C：Context Window 不等於永久記憶
例如 Chatbot 一個月以前的對話，不會因為使用了 Transformer 就自動永久存在模型內。

必須區分以下三種 Memory：

Memory

資訊保存在哪裡？

用途

Model Parameters

神經網路 Weights

訓練後的統計知識及能力

Context / KV Cache

當前推論序列及其快取

使用當前上下文生成

External Memory

DB、Vector DB、Files

跨 Session 保存資訊

KV Cache 是計算快取，不是跨 Session 的持久記憶資料庫。

8.3 如何設計 Long-context System？
假設公司要讓 LLM 讀取 500 頁的技術文件，回答某台機器的 Autofocus 設定問題。

有四種典型選擇。

方法

具體作法

Trade-off

Full Long Context

將大量文件全部放入 Prompt

簡單，但成本高，且可能干擾定位

Truncation

只保留部分內容

便宜，但可能刪除重要證據

Summarization

摘要後放入 Context

節省 Tokens，但可能遺失細節

RAG

先檢索相關片段

通常較有效率，但受 Retrieval Quality 影響

對專業工程文件，較好的起點通常是帶有 Metadata Filter 的 RAG，而不是不加選擇地把整個資料庫塞進 Prompt。

例如：

User Question
      |
      v
Query Parsing
      |
      v
Equipment / Version Filter
      |
      v
Hybrid Retrieval
      |
      v
Reranking
      |
      v
Relevant Chunks
      |
      v
Prompt Construction
      |
      v
LLM Answer + Citations
如果模型需要進行跨數十篇文件的複雜比較，則可能需要 RAG 結合 Long Context，而不是只選其中一個。

9. KV Cache 深入：Autoregressive Inference 的核心優化
這一節特別重要，因為 KV Cache 不只關係到 LLM Architecture，也直接影響 Production Serving 的速度、成本和同時服務能力。

9.1 理解兩種推論階段：Prefill vs. Decode
假設 Prompt：

Explain why the camera autofocus failed.

Tokenization 後假設有 10 個 Tokens。

模型會進行兩個階段。

Phase A — Prefill

Prompt Processing
一次處理 Prompt 中已知的多個 Tokens，為各層建立初始 KV Cache。

1

2

3

4

5

6

7

8

9

10

10 個 Prompt Tokens：通常可利用 GPU 平行運算。

Phase B — Decode

Autoregressive
逐一生成新 Token，重用已儲存的 Key、Value。

Cached K,V × 10

Token 11

Cached K,V × 11

Token 12

Prefill
Prefill 會處理已知的 Prompt Tokens。

對長度 
𝑇
T 的 Prompt，在標準 Full Self-Attention 下，主要 Attention 計算複雜度約為：

𝑂
(
𝑇
2
𝐷
)
O(T 
2
 D)
由於整段 Prompt 已知，因此可以平行計算多個 Token Position。

Decode
在已經建立 KV Cache 後，每次只需要處理新增的 Token。

當前 Query：

𝑞
𝑡
∈
𝑅
𝑑
ℎ
q 
t
​
 ∈R 
d 
h
​
 
 
與快取中所有有效的 Keys：

𝐾
≤
𝑡
∈
𝑅
𝑡
×
𝑑
ℎ
K 
≤t
​
 ∈R 
t×d 
h
​
 
 
進行：

Attention
⁡
(
𝑞
𝑡
,
𝐾
≤
𝑡
,
𝑉
≤
𝑡
)
Attention(q 
t
​
 ,K 
≤t
​
 ,V 
≤t
​
 )
單一 Head、單一 Token 的 Attention 核心計算隨歷史長度約為：

𝑂
(
𝑡
𝑑
ℎ
)
O(td 
h
​
 )
而不是重新計算整段序列所有位置之間的 
𝑇
×
𝑇
T×T Attention。

但要注意：

有 KV Cache 不代表每個新 Token 的運算完全固定。 歷史序列越長，當前 Query 仍需讀取越多先前 K、V，除非使用 Sliding Window 等限制可見歷史的機制。

9.2 KV Cache 的 Tensor Shape
對每一個 Transformer Layer：

𝐾
cache
∈
𝑅
𝐵
×
𝐻
𝑘
𝑣
×
𝑇
×
𝑑
ℎ
K 
cache
​
 ∈R 
B×H 
kv
​
 ×T×d 
h
​
 
 
𝑉
cache
∈
𝑅
𝐵
×
𝐻
𝑘
𝑣
×
𝑇
×
𝑑
ℎ
V 
cache
​
 ∈R 
B×H 
kv
​
 ×T×d 
h
​
 
 
其中 
𝐻
𝑘
𝑣
H 
kv
​
  是 Key/Value Heads 的數量。

標準 MHA 中通常：

𝐻
𝑘
𝑣
=
𝐻
𝑞
H 
kv
​
 =H 
q
​
 
GQA 則可能：

𝐻
𝑘
𝑣
<
𝐻
𝑞
H 
kv
​
 <H 
q
​
 
9.3 GPU Memory 怎麼計算？
對常見、每層相同 KV 配置的模型：

𝑀
KV
=
2
𝐵
𝐿
𝑇
𝐻
𝑘
𝑣
𝑑
ℎ
𝑠
M 
KV
​
 =2BLTH 
kv
​
 d 
h
​
 s
​
 
其中：

2：Key 和 Value 兩份

B：Batch Size

L：Number of Layers

T：Cached Sequence Length

𝐻
𝑘
𝑣
H 
kv
​
 ：KV Heads

𝑑
ℎ
d 
h
​
 ：Head Dimension

s：每個元素的 Bytes

例如 FP16：

𝑠
=
2
s=2
實際計算器：KV Cache 需要多少 GPU Memory？
Batch Size
4

Transformer Layers
32

Sequence Length
KV Heads
KV Precision
FP16/BF16
8-bit
Estimated Raw KV Cache Memory

4.00 GiB
假設 Head Dimension = 128

估算純 KV Tensor 儲存量，不包含模型權重、其他 Activations、Allocator Fragmentation、量化 Metadata、預留空間等。8-bit 模式是假設每個元素的原始儲存量為 1 byte。
用上面的預設設定：

𝐵
=
4
,
𝐿
=
32
,
𝑇
=
8192
B=4,L=32,T=8192
𝐻
𝑘
𝑣
=
8
,
𝑑
ℎ
=
128
,
𝑠
=
2
H 
kv
​
 =8,d 
h
​
 =128,s=2
得到：

𝑀
KV
=
4
 GiB
M 
KV
​
 =4 GiB
如果改成 MHA：

𝐻
𝑘
𝑣
=
32
H 
kv
​
 =32
則：

𝑀
KV
=
16
 GiB
M 
KV
​
 =16 GiB
這就解釋了為什麼 GQA 對大規模推論特別重要。

9.4 MHA、MQA、GQA 的區別
相同 8 個 Query Heads 時的 KV Sharing

MHA

8 KV Heads

KV

KV

KV

KV

KV

KV

KV

KV

每個 Q Head 各有自己的 KV Head

GQA

2 KV Heads

KV

KV

每 4 個 Q Heads 共用一個 KV Head

MQA

1 KV Heads

KV

全部 Q Heads 共用同一個 KV Head

概念示意。實際每個 KV Head 會被對應的一組 Query Heads 使用。
MHA
Multi-Head Attention：

𝐻
𝑞
=
𝐻
𝑘
𝑣
H 
q
​
 =H 
kv
​
 
優點是每個 Query Head 都有對應的 KV 表示。

缺點是 KV Cache 的容量與讀取頻寬成本較高。

MQA
Multi-Query Attention：

𝐻
𝑘
𝑣
=
1
H 
kv
​
 =1
多個 Query Heads 共用同一組 Key/Value Head。

這能顯著降低 KV Memory，但可能帶來 Model Quality 的 Trade-off。

GQA
Grouped-Query Attention：

1
<
𝐻
𝑘
𝑣
<
𝐻
𝑞
1<H 
kv
​
 <H 
q
​
 
例如：

𝐻
𝑞
=
32
,
𝐻
𝑘
𝑣
=
8
H 
q
​
 =32,H 
kv
​
 =8
每組四個 Query Heads 共用一組 KV Head。

GQA 是 MHA 與 MQA 之間的一種折衷，能降低 KV Cache 和 Memory Bandwidth，並嘗試維持較好的模型品質。

ACL Anthology

9.5 其他重要的 KV Cache 技術
技術

核心方法

解決問題

Dynamic KV Cache

根據序列增長分配快取

彈性支援不同長度

Static KV Cache

預先配置固定容量

有利於固定 Shape 和 Compiler Optimization

Paged KV Cache

將 KV 分成 Blocks/Pages

改善 Memory Fragmentation

KV Quantization

降低快取資料精度

減少 VRAM 用量

Sliding-window Cache

只保留指定歷史範圍

限制記憶體成長

Prefix Cache

重用共同 Prompt Prefix 的 KV

降低重複 Prefill 成本

Cache Offloading

把部分 KV 移至 CPU

緩解 GPU VRAM 限制

Hugging Face Transformers 已提供 Dynamic、Static、Quantized、Offloaded 等相關機制。

Hugging Face

PagedAttention 為什麼重要？
想像有 1,000 個使用者同時呼叫 LLM。

每個使用者的 Context Length 都不同。

傳統連續記憶體配置容易造成 Fragmentation 或浪費。

PagedAttention 將 KV Cache 分成較小的 Memory Blocks，再利用映射管理，概念上類似 Operating System 的 Virtual Memory。

vLLM 的 PagedAttention 就是為了改善 LLM Serving 的 KV Memory Management 與 Throughput。

DOI

9.6 KV Cache 可以加速多少？
沒有單一固定答案。

要比較：

Prompt Length

Output Length

Model Size

GPU Type

Batch Size

Memory Bandwidth

KV Layout

Serving Framework

一個常見誤解是：

使用 KV Cache 後，每生成一個 Token 就不需要再跑 Transformer。

這不正確。

實際上，每個新 Token 仍然需要通過所有 Transformer Layers，計算新的 Q、K、V、Attention 和 FFN。

KV Cache 只是避免重算過去 Tokens 的 K、V 及其上游表示。

如果模型很大，單 Token Decode 可能仍需要從 GPU Memory 讀取大量 Model Weights，造成 Memory-bandwidth Bottleneck。

所以 KV Cache 只是 Inference Optimization 的重要一環，而不是全部。

Part III — 實作與整合：從 PyTorch 到 Production LLM
10. 用 PyTorch 實作 Causal Self-Attention + KV Cache
以下是一個可執行的簡化範例。

它示範：

Linear QKV Projection

Multi-Head Reshape

Causal Attention

KV Cache

Incremental Decoding

驗證 Full Forward 與 Cached Forward 是否一致

此範例已使用 PyTorch 在 CPU 環境驗證。為了聚焦 Attention，未包含 RoPE、Dropout、完整 Transformer Block 或整個語言模型。


import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, d_model=128, n_heads=4):
        super().__init__()

        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Create Query, Key and Value projections
        self.qkv = nn.Linear(
            d_model,
            3 * d_model,
            bias=False
        )

        self.out = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

    def forward(
        self,
        x,
        past_kv=None,
        use_cache=False
    ):


預期輸出：

Cache equivalence: True
K shape: torch.Size([1, 4, 5, 32])
V shape: torch.Size([1, 4, 5, 32])
這代表在此測試條件下，使用 KV Cache 與不使用 Cache，最後一個 Token 的 Attention Output 數值一致。

這是工程師開發 KV Cache 時應建立的基本 Unit Test 之一。

但要注意實際 Production 還要測試：

多 Layer Cache

RoPE Position Offset

Padding Mask

Incremental Chunk Length 大於 1

Sliding Window

GQA

Batch 中不同 Sequence Length

Cache Reset 與 Request Isolation

上述程式在 T=1 的 Incremental Decoding 使用 is_causal=False 是正確的，因為當前 Query 位於所有既有 Keys 之後。

但如果一次新增多個 Tokens，就不能直接沿用這個簡化邏輯，而需要建立正確對齊的 Causal Mask。

PyTorch 的 scaled_dot_product_attention 可以根據設備、條件與版本選擇優化的 Attention Kernel。

PyTorch main documentation

11. 如何把 Attention 組成 Transformer Block？
假設使用 Pre-Norm + Feed-forward：

class TransformerBlock(nn.Module):

    def __init__(self, d_model=128, n_heads=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)

        self.attention = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads
        )

        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):

        # Pre-Norm Attention + Residual
        attn_output, _ = self.attention(
            self.norm1(x)
        )

        x = x + attn_output

        # Pre-Norm FFN + Residual
        x = x + self.ffn(self.norm2(x))

        return x

此範例需要與前面的 CausalSelfAttention 類別放在同一程式中。

再將多個 Blocks 疊起來：

layers = nn.ModuleList([
    TransformerBlock(
        d_model=128,
        n_heads=4
    )
    for _ in range(6)
])

x = torch.randn(2, 32, 128)

for layer in layers:
    x = layer(x)

print(x.shape)

# torch.Size([2, 32, 128])

這就是 Transformer Backbone 的基本組成方式。

若要做真正的 Decoder-only Language Model，還需要：

Token IDs
    |
    v
Token Embedding
    |
    v
Positional Information
    |
    v
Transformer Block × N
    |
    v
Final Normalization
    |
    v
LM Head
    |
    v
Vocabulary Logits
    |
    v
Cross-Entropy Training
這個教學模型與現代 LLM 有哪些差距？
現代 Foundation Model 通常還要考慮：

RoPE 或其他 Position Encoding

GQA / MQA / MHA

RMSNorm

SwiGLU

Tensor Parallelism

Mixed Precision

Distributed Training

Optimized Attention Kernels

高效率 KV Cache

更複雜的 Weight Initialization、Training Schedules 和數值穩定性設計

但以上的簡化版本已經涵蓋 Transformer 最重要的計算骨架。

12. 完整案例：LLM 如何回答一個技術問題？
用一個非常具體的工程問題整合九項知識。

假設公司有一套攝影機影像檢測設備，使用者詢問：

Why does the camera autofocus fail when the lens moves near the boundary?

我們假設這是一個搭配 RAG 的 Decoder-only LLM Application。

Step 1：Tokenization
文字先轉換成 Token IDs：

Why
does
the
camera
autofocus
...
實際分詞會依 Tokenizer 而異。

接下來：

Token IDs:
[...]
Token Count 決定後續 Prompt 的長度與成本。

Step 2：Embedding Retrieval
因為這是專業技術問題，系統先查詢文件。

Question
    |
    v
Embedding Model
    |
    v
Query Vector
    |
    v
Vector Database
    |
    v
Autofocus Troubleshooting Documents
系統可能找到：

Document A:
Autofocus boundary handling

Document B:
Liquid lens sweep configuration

Document C:
Focus metric validation
這些文件都是假設的檢索結果，不代表已讀取任何真實設備文件。

Step 3：Prompt Construction
系統建構：

SYSTEM:
Answer using the supplied technical documents.
If information is insufficient, state the uncertainty.

CONTEXT:
[Document A]
[Document B]
[Document C]

USER:
Why does the camera autofocus fail
when the lens moves near the boundary?
此時需要計算總 Token Count，檢查 Context Window。

Step 4：Token Embedding + Positional Encoding
假設完整 Prompt 是 3,000 Tokens：

𝑇
=
3000
T=3000
如果 Hidden Dimension 是 4,096：

𝑋
∈
𝑅
3000
×
4096
X∈R 
3000×4096
 
每個 Token 被轉換成 4,096 維向量。

Position Encoding 讓模型能利用位置與順序資訊。

Step 5：Transformer Processing
假設 32 層 Transformer Blocks。

每層會進行：

Input Representations
       |
       v
Attention
       |
       v
Residual + Normalization
       |
       v
FFN
       |
       v
Residual + Normalization
       |
       v
Next Layer
每個 Token 的 Representation 逐層更新。

例如處理：

autofocus

時，模型可能聚合與：

lens、boundary、fail

有關的 Context Representation。

但真正學到什麼關係，需要經過模型分析才能確認，不能只由文字表面推定。

Step 6：Output Logits
最後一個位置的 Hidden State 透過 LM Head 產生：

𝑧
∈
𝑅
𝑉
z∈R 
V
 
也就是所有 Vocabulary Tokens 的 Logits。

例如模型開始回答：

The autofocus system ...

實際生成內容取決於檢索到的技術文件與模型行為。

Step 7：Generation
假設系統使用：

temperature = 0.2
top_p = 0.9
max_new_tokens = 500

這些參數控制抽樣行為及最大輸出長度。

但如果希望固定輸出、便於測試，可以改用 Greedy Decoding 或其他受控的生成策略。

Step 8：KV Cache
Prefill 已經計算 3,000 個 Prompt Tokens 的 KV。

生成第一個 Token 後，Cache 長度增加。

接下來每輪：

New Token
    |
    v
Q, K, V Projection
    |
    v
Append New K, V to Cache
    |
    v
Attention with Previous KV
    |
    v
FFN
    |
    v
Next Token Logits
直到模型產生 EOS、達到長度限制，或觸發其他停止條件。

Step 9：Production Validation
最後不能只是把生成文字顯示給使用者。

對企業技術系統，還應加入：

Technical Document Citation

Retrieval Grounding Check

不確定性處理

Latency Monitoring

Token Usage Monitoring

Hallucination Evaluation

Regression Testing

文件與模型版本記錄

這樣才把 Foundation Model 的計算能力轉換成可以交付的 Engineering System。

Part IV — Senior LLM Engineer：效能分析與技術面試
13. Training 與 Inference 的差異
比較

Training

Inference

主要目標

更新 Model Weights

根據既有 Weights 產生結果

Backpropagation

需要

通常不需要

Optimizer State

通常需要

通常不需要

Token Processing

常使用 Teacher Forcing 並行運算

Prefill 並行，Decode 逐 Token

KV Cache

標準完整序列訓練通常不使用

Autoregressive Decoding 常使用

Memory Usage

Weights、Gradients、Activations、Optimizer

Weights、KV Cache、Temporary Activations

重要 Metrics

Training Loss、Validation Loss、Throughput

TTFT、TPOT、Latency、Throughput

TTFT（Time to First Token）
從請求開始到第一個輸出 Token 可用的時間。

受 Prompt Length、排隊、Prefill、GPU 負載等因素影響。

TPOT（Time per Output Token）
輸出 Token 之間的平均時間。

主要反映 Decode 階段的效能。

Throughput
例如：

1000
 output tokens/sec
1000 output tokens/sec
代表系統在某個量測條件下每秒可以產生 1,000 個 Output Tokens。

要明確區分：

Per-request Throughput

Aggregate Throughput

Input Token Throughput

Output Token Throughput

因為這些數字不一定可以直接比較。

14. 為什麼 FlashAttention 能加速 Transformer？
普通 Attention 的直覺實作可能會建立完整：

𝑆
=
𝑄
𝐾
𝑇
S=QK 
T
 
其中：

𝑆
∈
𝑅
𝑇
×
𝑇
S∈R 
T×T
 
當 T 很大時，這個矩陣的 Memory Footprint 非常可觀。

FlashAttention 的核心不是改變 Attention 的數學定義，而是改善計算與記憶體存取方式。

它使用 Tiling、IO-aware Optimization 與高效率 Softmax 計算，減少 GPU High-bandwidth Memory（HBM）之間不必要的讀寫。

因此可以在維持標準 Attention 數學結果（容許浮點運算誤差）的前提下，降低 Memory Traffic、改善效能。

它不是把所有 Full Attention 的計算複雜度直接變成線性。

arXiv

Senior Engineer 要懂的重點
GPU Bottleneck 不一定是純 FLOPs。

可能是：

HBM Bandwidth

Kernel Launch Overhead

不必要的 Intermediate Tensors

Low GPU Occupancy

Batch Size 不合適

Fragmented KV Cache

CPU–GPU Synchronization

例如在 Decode 階段，低 Batch Size 的大型 Decoder Model 往往容易受到 Memory Bandwidth 限制。

這就是為什麼只提高 GPU 的理論 FLOPs，不一定能等比例降低 Token Latency。

15. Senior 技術面試常見問題與回答方向
面試問題

必須答出的核心技術

Why use Multi-Head Attention?

不同 Attention Projections、Representation Capacity、Head Dimension

Why divide by square root of d_k?

Dot-product Variance、Softmax Saturation、Gradient Stability

Why is Causal Mask needed?

Autoregressive Factorization、Prevent Future-token Leakage

Pre-Norm vs. Post-Norm?

Gradient Flow、Residual Path、Training Stability

LayerNorm vs. RMSNorm?

Centering、Variance/RMS Scaling、Efficiency

RoPE vs. Sinusoidal?

Position Injection、Relative-position Properties、Extrapolation

Why Decoder-only for Chat LLM?

Next-token Training Objective、Autoregressive Generation

Why does KV Cache speed up decoding?

Reuse Past Keys/Values、Reduce Repeated Computation

Why does KV Cache consume so much memory?

Layers × Tokens × KV Heads × Head Dimension × Bytes

MHA vs. GQA vs. MQA?

KV Sharing、Bandwidth、Memory and Quality Trade-offs

Why does long context slow inference?

Quadratic Prefill Attention、Longer KV Reads、Memory Pressure

How do you optimize LLM Serving?

Profiling、Batching、GQA、Quantization、Optimized Kernels、Paged KV

Why use Embeddings for RAG?

Semantic Representation、Similarity Search、Retrieval

Why is Temperature not a reasoning method?

It changes Token Selection, not underlying Learned Weights

Why can Cached Decoding produce wrong results?

Position ID、Causal Mask、Cache Alignment、Padding、RoPE Bugs

一道比較接近 Senior / Staff 的 System Design 題目
We have a 32-layer decoder-only LLM serving 1,000 concurrent users. Long prompts cause high latency and GPU OOM. How would you diagnose and optimize the system?

理想答案不應直接跳到「增加 GPU」。

應該建立以下分析流程。

第一階段：Measure

量測 TTFT、TPOT、Aggregate Throughput、Input/Output Token Length Distribution、GPU Memory、Memory Bandwidth、Batch Size、KV Cache Utilization、OOM Frequency。

第二階段：Locate Bottleneck

確認究竟是：

Prefill Compute Bottleneck

Decode Memory-bandwidth Bottleneck

KV Cache Capacity

Scheduling / Fragmentation

Model Weight Memory

CPU / Tokenizer / Network Overhead

第三階段：Targeted Optimization

例如：

Prefill 特別慢，可以考慮 FlashAttention、Chunked Prefill、減少無關 Context、Prefix Cache。

Decode 特別慢，可以評估 Continuous Batching、GQA/MQA 模型、低精度計算、合適的 GPU Memory Layout。

如果是 KV Cache OOM，則可以評估 Paged KV、KV Quantization、合理的 Context Limits、Offloading 或增加 GPU 容量。

第四階段：Regression Validation

任何 Optimization 都必須比較：

TTFT P50 / P95 / P99

TPOT P50 / P95 / P99

Tokens per Second

Memory Peak

Task Accuracy

Long-context Retrieval Accuracy

Cost per 1M Tokens

並確保新方案沒有造成不可接受的 Quality Regression。

真正的 Senior Engineer 不只會說哪個技術比較快，而是知道怎麼測量、找瓶頸、決定 Trade-off，並安全地推出 Production 改動。

最後總結：九項技術應掌握到什麼程度？
知識

Intern 應掌握

Senior Engineer 應掌握

Transformer

四個核心元件與作用

Block Design、Pre/Post-Norm、FFN Variants、Gradient Flow

Self-Attention

Q、K、V 的直覺

數學、Tensor Shape、Masks、Scaling、Multi-Head

Positional Encoding

為什麼需要順序資訊

Sinusoidal、RoPE 數學、Position Offsets、長序列問題

Tokenization

Token 與單字不同

BPE Training、Vocabulary、Special Tokens、Memory/Cost

Model Architecture

三種 Architecture 的差異

Attention Masks、Training Objectives、Cross-Attention、選型

Generation

四種 Decoding Methods

Probability、Logits、Sampling Implementation、Sequence Scoring

Embeddings

語意向量與 Similarity

Contrastive Learning、Vector Retrieval、Evaluation

Context Window

Token 上限與 Truncation

Long-context Evaluation、Compute、Memory、RAG Trade-offs

KV Cache

重用舊 K/V 加速生成

Prefill/Decode、Memory Formula、GQA、Paged Cache、Serving Optimization

建議的學習優先順序
對準備 Senior Applied LLM Engineer 與 Foundation Model Engineer 技術面試，我建議分三階段學習。

第一階段：真正理解 Transformer 數學與程式碼

優先精通 Self-Attention、Multi-Head Attention、Tensor Shapes、Causal Mask、Residual、Normalization 和 Next-token Training Objective。這些是閱讀其他 LLM 技術的基礎。

第二階段：實作完整 Decoder-only Mini LLM

自己實作 Token Embedding、Position Encoding、Transformer Blocks、LM Head、Cross-Entropy Training，再用小型文字資料集進行訓練。接著實作 Greedy/Top-p Decoding 和 KV Cache，驗證 Cached 與 Non-cached Forward 的數值一致性。

第三階段：轉向 Production 與大型模型技術

學習 FlashAttention、RoPE、GQA、KV Cache Memory、Continuous Batching、vLLM、量化、Profiling、Long-context Evaluation。這部分尤其適合準備 Senior LLM Inference Engineer 與 Foundation Model Infrastructure Engineer 職位。

對於 Senior Applied AI Engineer，第一、第二階段的理解足以支撐大部分底層選型與問題分析；還需要額外加強 RAG、Agent、Evaluation、Deployment、Security 等應用系統能力。

對於 Foundation Model Research / Training Engineer，則應在這九項基礎之上，更深入 Distributed Training、Optimization、Data Mixture、Pretraining、Post-training 和 Scaling Laws。

如果要選最值得完整手寫、測試並解釋的技術組合，我會選：

Multi-Head Causal Attention + Transformer Block + RoPE + KV Cache + Autoregressive Generation。

因為這個組合把 Transformer 的數學、模型架構、Training/Inference 差異、GPU 效能和 Production 問題直接連接起來，也是跨越 Intern 理解與 Senior Engineering 深度的關鍵。

可延伸閱讀的兩個高價值實作資源是 
Harvard — The Annotated Transformer
（逐步解釋 Transformer 實作）與 
PyTorch — Scaled Dot Product Attention
（Attention API、Tensor Shape 與最佳化實作）。