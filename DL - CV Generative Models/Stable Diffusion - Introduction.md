
##### Theory
[從頭開始學習Stable Diffusion：一個初學者指南](https://chrislee0728.medium.com/%E5%BE%9E%E9%A0%AD%E9%96%8B%E5%A7%8B%E5%AD%B8%E7%BF%92stable-diffusion-%E4%B8%80%E5%80%8B%E5%88%9D%E5%AD%B8%E8%80%85%E6%8C%87%E5%8D%97-ec34d7726a6c)

[十分钟读懂Stable Diffusion](https://zhuanlan.zhihu.com/p/600251419)

[深入浅出完整解析Stable Diffusion（SD）核心基础知识](https://zhuanlan.zhihu.com/p/632809634)

[How to Use Stable Diffusion Effectively](https://machinelearningmastery.com/how-to-use-stable-diffusion-effectively/)

[stable-diffusion-tutorial](https://github.com/ai-vip/stable-diffusion-tutorial)  全网最全Stable Diffusion教程

[Stable Diffusion WebUI使用手冊](https://ivonblog.com/posts/stable-diffusion-webui-introduction/)

[AI绘画新突破-ControlNet精准控图](https://zhuanlan.zhihu.com/p/607139523)



##### Notebook
[01_stable_diffusion_introduction.ipynb](https://colab.research.google.com/drive/1wOIvZW4ic6LTg-VkPrSTFuhotJ-bVJnj#scrollTo=noPILuVQET7y)

[Stable Diffusion Deep Dive.ipynb](https://github.com/fastai/diffusion-nbs/blob/master/Stable%20Diffusion%20Deep%20Dive.ipynb)

[Stable Diffusion Online](https://link.zhihu.com/?target=https%3A//stablediffusionweb.com/%23demo)

[Stable Diffusion img2img colab](https://colab.research.google.com/drive/1hs5dYbPHVDv3AhkpZTGhck7H2E_4NBwZ?usp=sharing)

[stable_diffusion.ipynb](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb)



|                                                            |                                                                                                             |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| SD<br>(Text2Img)                                           | 功能：輸入文字描述，生成符合描述的圖像。<br>影像：生成藝術插畫、海報設計、角色設計、概念圖。<br>影片：為短片或動畫生成場景概念圖，作為創意參考。<br>3D：創建3D角色或場景設計的參考圖像，提供建模靈感。 |
| SD<br>(Img2Img)                                            | 功能：基於輸入圖像生成風格化或修改後的圖像。<br>影像：風格遷移（如照片變油畫）、增強細節（如增加紋理）。<br>影片：基於影片幀進行風格化處理，實現影片轉換效果<br>3D：為3D模型提供材質或紋理參考。    |
| SD<br>(InPainting)                                         | 功能：對圖像中的特定區域進行重繪。<br>影像：修復圖像損壞部分、去除或更換物體。<br>影片：修補影片中的瑕疵（如擦除水印或不需要的物體）。<br>3D：修正3D場景渲染後的缺陷或補充細節。            |
| SD<br>(Depth2Image)                                        | 功能：基於深度圖生成新圖像，保持場景結構不變。<br>影像：生成具有一致透視和深度的替代圖像。<br>影片：生成場景深度一致的動畫。<br>3D：根據深度圖生成3D模型參考，幫助創建更加真實的3D環境。       |
| SD+ControlNet<br>(canny,hed,mlsd,scribble)<br>黑底白色edge img | 功能：基於邊緣檢測控制生成圖像。<br>影像：生成符合物體邊緣輪廓的圖像，提升細節控制。<br>影片：生成多幀具有連續邊緣一致性的動畫。<br>3D：利用邊緣檢測圖生成3D模型參考，確保邊緣清晰。          |
| SD+ControlNet<br>(openpose)<br>keypoint pose img           | 功能：基於人體姿勢骨架生成圖像。<br>影像：生成符合特定人體姿勢的藝術作品。<br>影片：生成連續人體動作的動畫。<br>3D：用於創建動作捕捉數據，幫助3D角色動畫設計。                     |
| SD+ControlNet<br>(depth)<br>灰階深度影像                         | 功能：利用深度圖控制生成圖像。<br>影像：強化場景的空間感，保持建築結構或地形的一致性。<br>影片：生成具有一致場景深度的多幀動畫。<br>3D：協助生成符合深度信息的3D模型參考。               |
| SD+ControlNet<br>(normal)<br>法線貼圖                          | 功能：利用法線圖控制生成圖像。<br>影像：生成符合光照與表面細節的圖像。<br>影片：根據法線信息生成動畫，保持光照一致性。<br>3D：幫助生成真實感法線貼圖或高精度模型參考。                  |
| SD+ControlNet<br>(seg)<br>語意分割                             | 功能：利用語義分割圖控制生成圖像。<br>影像：生成符合分割區域定義的圖像，適合自動化場景設計。<br>影片：根據分割區域生成場景一致的影片。<br>3D：根據分割信息生成3D場景建模參考。             |





![[stablediffusion.webp]]

Stable Diffusion的原理、模式、架構、输入和输出:

## Stable Diffusion基本原理

Stable Diffusion模型是一個強大的生成模型，可以根據文字描述（prompt）生成圖片。它利用了一些重要的單元，包括文字編碼器（Text Encoder）、擴散模型（Diffusion Model）、變分自動編碼器（VAE）、以及控制網路（ControlNet）等。以下是Stable Diffusion的各主要組件的詳細解釋和運作流程。

### 1. Text Encoder（文字編碼器）

Text Encoder是Stable Diffusion的第一個步驟。通常使用的是CLIP模型中的文字編碼部分。文字編碼器的主要任務是將文字（prompt）轉化為語義豐富的向量表示，該表示包含了描述中的語意、情緒和關鍵詞。

#### 運作流程：

- 輸入：文字描述，例如「一隻坐在草地上的貓」。
- 編碼：文字描述經過Tokenization（分詞）後，轉化為詞嵌入（Word Embeddings）。
- 輸出：輸出為一個嵌入向量（Text Embedding），這個向量用於向Diffusion Model提供指導，使生成的圖片能夠符合文字描述的內容。

### 2. Diffusion Model（擴散模型）

Diffusion Model是Stable Diffusion的核心部分，主要用於生成圖片。該模型的工作原理是「去噪聲」，將一張噪聲圖片逐步轉化為一張清晰的圖片。

#### 運作流程：

- 擴散過程（Forward Process）：訓練時，模型學習如何從清晰圖片中逐步添加隨機噪聲。該過程讓模型理解每一層的圖片結構。
- 去噪過程（Reverse Process）：在生成圖片時，模型從隨機噪聲開始，逐步去除噪聲，最終生成符合輸入文字描述的圖片。
- 梯度下降：在訓練過程中，模型會使用梯度下降優化其參數，以最小化生成圖片與目標圖片之間的差異。

#### 具體步驟：

- 輸入：隨機噪聲圖片和文字編碼器的輸出（嵌入向量）。
- 每個時間步生成：將噪聲圖片經過多次迭代去噪，並在每次去噪時參考文字編碼器的嵌入向量。
- 輸出：生成出符合文字描述的清晰圖片。

### 3. VAE（變分自動編碼器）

VAE用於圖像生成的最後步驟，它將生成圖片從「潛在空間」（latent space）轉換回到「像素空間」（pixel space）。在Stable Diffusion中，VAE有時用於壓縮圖片到潛在空間，這樣可以降低模型的計算負擔。

#### 運作流程：

- 編碼器（Encoder）：將圖片壓縮到潛在空間，提取出低維度的特徵表示。
- 解碼器（Decoder）：將潛在空間中的圖片轉回原始的像素空間。
- 作用：將生成圖片從模型內部的低維空間轉回到可視的高分辨率圖片。

#### 具體步驟：

- 輸入：Diffusion Model生成的低維圖片表示。
- 輸出：高分辨率的圖片，即最終生成結果。

### 4. ControlNet（控制網路）

ControlNet是一種擴展模型，通常在Stable Diffusion基礎上進行調整，為生成過程提供更多控制。這些控制可以包括圖像風格、構圖、視角等，從而提高生成圖片的定制性。

#### 運作流程：

- ControlNet學習如何在生成過程中插入某些特定的控制信息，例如邊緣地圖、姿態、深度等。
- 通過將這些輔助信息引入模型，ControlNet可以在生成圖片時進一步控制圖片的構圖和細節。

#### 具體步驟：

- 輸入：ControlNet會接收一組控制信息，例如特定的輪廓圖、姿態標記等。
- 輸出：在生成過程中參考這些信息，生成符合特定要求的圖片。

### Stable Diffusion生成圖片的完整流程

1. **文字編碼**：輸入的prompt通過Text Encoder轉換為嵌入向量。
2. **噪聲注入**：模型從隨機噪聲圖片開始生成。
3. **逐步去噪**：Diffusion Model在多個步驟中逐步去除噪聲，生成符合prompt的圖片。
4. **潛在空間到像素空間轉換**：最終的潛在圖片表示通過VAE的解碼器轉換為像素空間中的高分辨率圖片。
5. **控制網路的應用**（如果需要）：在生成過程中添加一些控制信息（如姿態或邊緣），進一步優化圖片的細節。

### 訓練流程

1. **數據準備**：收集大量的圖像和文字描述對，這些數據將用於訓練模型。
2. **文字嵌入學習**：使用CLIP或其他文字編碼器，將文字描述轉化為嵌入向量。
3. **噪聲添加與去噪學習**：在訓練中，模型學習在不同噪聲水平下如何去除噪聲以還原清晰圖片。這一步中，模型會逐步從完全噪聲的圖片學習如何生成清晰的圖片。
4. **VAE編碼與解碼**：學習如何將圖片壓縮到潛在空間，並在需要時解碼回到像素空間。
5. **控制網路的訓練**（如果有）：引入更多控制信號（例如姿態信息）來進行輔助學習，使模型在生成圖片時可更靈活地控制構圖和姿態等細節。

經過這些步驟的訓練後，Stable Diffusion就能夠根據prompt生成高質量的圖片。


#### Q請中文詳細解釋並具體舉例(列出shape or size): Word2Vec, Tokenizer, Text Encoder, BPE, CLS, 靜態詞嵌入, 上下文感知嵌入, WordPiece. 所以Word2Vec跟Text Encoder的輸出都是text embedding? 靜態詞嵌入跟動態編碼器的關係或差異?

### 中文詳細解釋並具體舉例：Word2Vec、Tokenizer、Text Encoder、BPE、CLS、靜態詞嵌入、上下文感知嵌入、WordPiece

---

#### **1. Word2Vec**

- **定義**：一種生成詞嵌入的技術，將單詞轉換為固定長度的向量，嵌入表示捕捉詞的語義關聯。
- **核心技術**：
    - **Skip-gram**：給定中心詞，預測上下文。
    - **CBOW（Continuous Bag of Words）**：給定上下文，預測中心詞。
- **輸出**：
    - 每個詞對應一個固定長度的嵌入向量，無論上下文如何，該嵌入始終不變（靜態嵌入）。
- **舉例**：
    - 假設詞彙表中包含 `["cat", "dog", "apple"]`，並設定嵌入向量的維度為 3。
    - **輸出形狀**：
        - `cat` → `[0.1, 0.3, -0.2]` （shape: `(3,)`）
        - `dog` → `[-0.4, 0.5, 0.8]` （shape: `(3,)`）
        - 詞彙表嵌入矩陣大小為 `(3, 3)`，其中行數為詞數，列數為嵌入維度。

---

#### **2. Tokenizer**

- **定義**：將文本拆分為基本單位（Token），如單詞、子詞或字符，生成模型可處理的輸入。
- **類型**：
    - **Word-based**：按單詞分詞（如 "I love dogs" → `["I", "love", "dogs"]`）。
    - **Subword-based（如 BPE 和 WordPiece）**：將單詞拆分為子詞（如 "loving" → `["lov", "ing"]`）。
    - **Character-based**：逐字符分詞（如 "dog" → `["d", "o", "g"]`）。
- **舉例**：
    - 文本 `"I love programming."` 使用 Word-based Tokenizer：
        - Token 結果：`["I", "love", "programming", "."]`
        - 索引：`[1, 2, 3, 4]`（假設字典中詞的索引從 1 開始）
        - **輸出形狀**：`(4,)`（4 個 Token 索引）。
- **作用**：
    - Tokenizer 是嵌入生成的第一步，將原始文本轉化為離散標籤索引。

---

#### **3. Text Encoder**

- **定義**：接受 Tokenizer 的輸出，將其轉換為上下文感知的嵌入向量。
- **架構**：
    - 通常是基於 Transformer（如 BERT、GPT）的模型。
- **輸出**：
    - 每個 Token 會映射為一個高維向量表示，上下文信息會影響這些向量。
- **舉例**：
    - 文本 `"I love dogs"` 經過 Tokenizer 和 Text Encoder：
        - Tokenizer 輸出：`[1, 2, 3]`
        - Encoder 輸出：
            - `[[0.5, -0.2, 0.1], [0.3, 0.7, -0.1], [-0.6, 0.8, 0.4]]`
        - **輸出形狀**：`(3, 3)`（3 個 Token，每個嵌入維度為 3）。

---

#### **4. BPE（Byte Pair Encoding）**

- **定義**：一種子詞級分詞方法，將少量常用字符組合成子詞，減少 OOV（未登錄詞）問題。
- **工作方式**：
    - 基於頻率合併字符對，例如：
        - 初始詞彙：`["l", "o", "v", "e", " ", "p", "r", "o", "g", "a", "m", "m", "i", "n", "g"]`
        - 頻率最高的字符對 `("m", "m")` → 合併成 `"mm"`。
    - 不斷合併，直到生成子詞單元。
- **舉例**：
    - 文本 `"programming"`：
        - BPE Tokenizer 輸出：`["prog", "ram", "ming"]`。
        - 對應索引：`[10, 23, 35]`。

---

#### **5. CLS（Classification Token）**

- **定義**：Transformer 模型（如 BERT）中特殊的預留標記，用於表示整段輸入的語義表示。
- **作用**：
    - CLS Token 的嵌入通常作為分類、回歸等任務的輸入。
- **舉例**：
    - 文本 `"I love dogs"` 經過 BERT 的 Tokenizer：
        - Tokenizer 輸出：`[CLS, "I", "love", "dogs", SEP]`。
        - 編碼器輸出：
            - CLS 的嵌入向量：`[0.1, -0.3, 0.5]`（shape: `(3,)`，表示整段語義）。

---

#### **6. 靜態詞嵌入**

- **定義**：固定的向量表示，單詞的嵌入不依賴上下文。
- **特徵**：
    - 單詞在所有句子中對應的嵌入始終相同。
    - 例如 Word2Vec、GloVe。
- **舉例**：
    - 單詞 `"bank"` 在 "river bank" 和 "financial bank" 中的嵌入完全相同。
    - 嵌入向量：`[0.2, -0.5, 0.7]`。

---

#### **7. 上下文感知嵌入**

- **定義**：動態生成的嵌入向量，單詞的表示受其上下文影響。
- **特徵**：
    - 使用 Transformer 編碼器。
    - 可以處理多義詞，根據上下文生成不同的嵌入。
- **舉例**：
    - 單詞 `"bank"`：
        - 在 "river bank" 中 → 嵌入向量：`[0.1, 0.3, 0.8]`。
        - 在 "financial bank" 中 → 嵌入向量：`[-0.2, 0.6, -0.1]`。

---

#### **8. WordPiece**

- **定義**：一種子詞分詞方法，通過概率模型決定如何將單詞拆分為子詞。
- **與 BPE 的區別**：
    - BPE 是基於字符對合併，按頻率選擇。
    - WordPiece 是基於語料庫中最大化似然估計的合併。
- **舉例**：
    - 文本 `"unhappiness"`：
        - WordPiece 輸出：`["un", "##happiness"]`（`##` 表示子詞）。
        - BPE 輸出：`["un", "hap", "piness"]`。

---

### **Word2Vec 與 Text Encoder 的輸出都是 Text Embedding 嗎？**

- **是的，但本質不同**：
    - **Word2Vec**：靜態詞嵌入，不考慮上下文，對每個單詞生成固定的向量。
    - **Text Encoder**：上下文感知嵌入，生成的嵌入向量會根據整句的語義動態調整。

---

### **靜態詞嵌入與動態編碼器的關係與差異**

|**比較項目**|**靜態詞嵌入（如 Word2Vec）**|**動態編碼器（如 Transformer）**|
|---|---|---|
|**上下文依賴性**|無上下文依賴|嵌入根據上下文動態生成|
|**處理多義詞能力**|無法區分，如 "bank" 始終同一嵌入|能區分上下文語義，例如 "river bank" 和 "financial bank" 不同|
|**嵌入表示**|固定向量|動態向量|
|**適用場景**|靜態任務，如詞相似度計算|高語義需求任務，如分類、生成等|
|**示例嵌入**|`[0.1, -0.2, 0.3]`（固定）|`[-0.3, 0.4, 0.5]`（上下文調整後不同）|

靜態詞嵌入已逐漸被上下文感知嵌入取代，後者能更準確地捕捉語義，適應複雜語言任務需求。