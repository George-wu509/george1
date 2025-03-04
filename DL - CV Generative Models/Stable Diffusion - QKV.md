
Stable Diffusion 中，**Positive Prompt** 和 **Negative Prompt** 是分別經過 CLIP 產生兩個 **Text Embedding**，然後這兩個嵌入會影響 U-Net 的去噪過程。在 Cross-Attention 機制中，這些嵌入會影響 U-Net 的 **Query (Q), Key (K), Value (V) 計算**，進而影響最終生成的圖像。

## **整體流程概覽**

1. **Prompt 輸入**
    
    - Positive Prompt: `"cyberpunk city at night with neon lights, cinematic lighting"`
    - Negative Prompt: `"low resolution, blurry, distorted"`
2. **CLIP Text Encoder 轉換**
    
    - CLIP 將 **Positive Prompt** 轉換為 `text_embedding_positive`
    - CLIP 將 **Negative Prompt** 轉換為 `text_embedding_negative`
3. **U-Net Cross-Attention**
    
    - Positive Embedding 用於指導 U-Net 生成符合 prompt 的圖像。
    - Negative Embedding 用於告訴 U-Net 應該避免某些特徵。
4. **最終輸出**
    
    - U-Net 完成去噪 → VAE 解碼 → 生成符合 Prompt 的高質量圖像。

---

# **第一步：Positive & Negative Prompt 經過 CLIP 產生 Text Embedding**

Stable Diffusion 使用 **CLIP Text Encoder**（如 OpenAI 的 ViT-L/14）來將文本轉換為潛在語義向量。

- CLIP 的輸入：Tokenized Prompt (`77` 個 token，因為 Stable Diffusion 限制最大長度為 `77`）
- CLIP 的輸出：一個 `(77, 768)` 或 `(77, 1024)` 的嵌入矩陣（視 CLIP 版本而定）

以示例來說：

mathematica

複製編輯

`Positive Prompt: "cyberpunk city at night with neon lights, cinematic lighting" Negative Prompt: "low resolution, blurry, distorted"`

CLIP Text Encoder 計算：

text_embedding_positive=E(Positive Prompt)\text{text\_embedding\_positive} = E(\text{Positive Prompt})text_embedding_positive=E(Positive Prompt) text_embedding_negative=E(Negative Prompt)\text{text\_embedding\_negative} = E(\text{Negative Prompt})text_embedding_negative=E(Negative Prompt)

這裡：

- `text_embedding_positive` 是 **Positive Prompt** 的 `(77, 768)` 矩陣
- `text_embedding_negative` 是 **Negative Prompt** 的 `(77, 768)` 矩陣

**這兩個嵌入都會被傳遞到 U-Net**，用來影響去噪過程。

---

# **第二步：U-Net 的 Cross-Attention 計算**

U-Net 在每個時步 ttt 執行去噪時，會使用 **Cross-Attention 機制** 來引導生成過程，使圖像與 Text Embedding 對齊。

## **Cross-Attention 機制**

在 U-Net 裡，Cross-Attention 的計算方式如下：

Attention(Q,K,V)=softmax(QKTd)V\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) VAttention(Q,K,V)=softmax(d​QKT​)V

其中：

- **Query (Q)** 來自 U-Net 的潛在表示（Latent Feature）。
- **Key (K)** 和 **Value (V)** 來自 **Text Embedding**。

在 Stable Diffusion 中：

- `text_embedding_positive` 會作為 **Key (K)** 和 **Value (V)**
- `text_embedding_negative` 會影響去噪過程，可能影響 **Attention Weights** 或 U-Net 模型的 Hidden State。

## **具體計算步驟**

### **(1) U-Net 的 Query 計算**

U-Net 在去噪過程中的一層 Transformer 會計算 Query：

Q=WQhQ = W_Q hQ=WQ​h

其中：

- hhh 是來自 U-Net 的 Feature Map（維度：`(Batch, Height × Width, Hidden_dim)`）
- WQW_QWQ​ 是一個投影矩陣，將 Feature 轉換為 Query 向量。

### **(2) Key 和 Value 計算**

K=WKE(Prompt),V=WVE(Prompt)K = W_K E(\text{Prompt}), \quad V = W_V E(\text{Prompt})K=WK​E(Prompt),V=WV​E(Prompt)

這裡：

- E(Prompt)E(\text{Prompt})E(Prompt) 是 **Positive 或 Negative Prompt 的 CLIP 嵌入**
- WKW_KWK​ 和 WVW_VWV​ 是學習參數，用來將 Text Embedding 轉換為 Attention 機制中的 Key 和 Value。

### **(3) 計算 Attention**

U-Net 計算：

A=softmax(QKTd)VA = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) VA=softmax(d​QKT​)V

這表示：

- **Query (Q)** 來自 U-Net Feature Map
- **Key (K)** 來自 Prompt Embedding
- **Value (V)** 來自 Prompt Embedding

這個過程會讓 U-Net 遵循文本描述來調整生成內容。

---

# **第三步：Negative Prompt 如何影響 Attention**

Stable Diffusion 使用 **兩組 CLIP Embedding** 來影響 U-Net：

1. **Positive Prompt（text_embedding_positive）**
    
    - 指導 U-Net 生成符合 prompt 描述的圖像。
2. **Negative Prompt（text_embedding_negative）**
    
    - 控制 U-Net 避免不想要的特徵，如模糊、低解析度等。
    - 在計算時，可能是： K=WKE(Positive)−λWKE(Negative)K = W_K E(\text{Positive}) - \lambda W_K E(\text{Negative})K=WK​E(Positive)−λWK​E(Negative) V=WVE(Positive)−λWVE(Negative)V = W_V E(\text{Positive}) - \lambda W_V E(\text{Negative})V=WV​E(Positive)−λWV​E(Negative)
    - 這表示 Negative Prompt 的 Embedding 會被「扣除」，以減少其對 Attention Weights 的影響。

**舉例** 如果 **Positive Prompt** 是：

arduino

複製編輯

`"cyberpunk city at night with neon lights, cinematic lighting"`

這會導致 **K, V** 方向與「霓虹燈光線」和「電影感」相關。

如果 **Negative Prompt** 是：

arduino

複製編輯

`"low resolution, blurry, distorted"`

這會抑制 Attention Weights 中與「模糊」或「低解析度」相關的部分，確保最終圖像更清晰。

---

# **第四步：U-Net 最終去噪 & VAE 解碼**

1. **U-Net 在每個時間步 ttt 依據 Attention 計算結果來去噪**
    
    - 這個過程會慢慢將噪聲轉變為符合 Prompt 的圖像。
2. **最後一步：VAE 解碼**
    
    - 生成的潛在向量 zTz_TzT​ 經過 **VAE Decoder** 轉換為最終圖像。

---

# **總結**

### **1. Positive & Negative Prompt 產生兩組獨立的 CLIP Text Embedding**

- `text_embedding_positive = E(Prompt)`
- `text_embedding_negative = E(Negative Prompt)`

### **2. U-Net 在 Cross-Attention 中如何使用這些 Embedding**

- **Positive Embedding** 作為 `K, V`，指導 U-Net 生成內容。
- **Negative Embedding** 影響 `K, V`，用來抑制不想要的特徵。

### **3. Attention 計算**

- **Query (Q)** 來自 U-Net Feature Map
- **Key (K) 和 Value (V)** 來自 Positive 和 Negative Prompt 的線性投影。

### **4. Negative Prompt 的影響**

- 透過 **減少 Attention Weights** 中與 Negative Prompt 相關的部分，確保圖像沒有模糊或失真。

這就是 Stable Diffusion 如何將文字 prompt 轉換為圖像的詳細過程！