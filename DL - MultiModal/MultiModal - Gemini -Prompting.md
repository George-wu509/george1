
Ref: [解读 Gemini 技术报告（Gemini: A Family of Highly Capable Multimodal Models）](https://zhuanlan.zhihu.com/p/670944833)

Germini:  
实实在在的 32K Context Length 预训练，对于 Long Context 场景更友好

Transformer Decoder 是主体训练结构，全部的多模态数据 Tokenizer 之后，打成 Multimodal Token Sequence 进行 Auto Regressive 训练，而不是 GPT4 使用不同模态 Encoder Or Decoder 进行 Cross Attention 之后组装到一起，又一次见证了 AR 训练方式的强大

在 RL 的 Reward 部分，选择了类似传统搜索 or 推荐场景的精排模型，即多目标 OEC 分数综合 Reward，对齐信号的可解释性更强，而且 Google 验证了这条通路之后，说不定以后的 Reward Model 都会这样铺开

选择多模态数据进行 AR，同时也意味着训练数据的远远扩充，Scaling Law 依旧稳定发挥作用
针对预训练和对齐方面的数据配比，都是在小模型上做了消融实验之后，Extend 到大模型
多次强调了数据质量的重要性大于数量

最后是一些疑点：评估方式不是令人完全信服，总体看下来：多模态能力可能比 GPT4V 好，但推理能力有待验证是不是真的好于 GPT4

### Model Architecture

- Gemini 模型的输入，可以夹杂着文本、音频、视觉（图片、图表、PDF、视频），并且最终输出文本和图片

- 文本部分的 Tokenizer 跟常规的 LLM 一样
- 视觉部分的 Tokenizer 受到了 Flamingo、CoCa 和 PaLI 的启发，但跟上述模型不同之处在于这部分从开始就是为了多模态任务而生的，可以把图像信息处理成离散的 Token
- 音频部分的 Tokenizer 使用了 USM（Universal Speech Model）将音频信息按照 16KHz 的采样率，处理成音频特征

- 这里和直接处理成文本的区别是：这种方法可以捕捉更细微的差别，ASR 之后不行

- 这里需要补充一点点知识

- Flamingo 原理

- [Flamingo: a Visual Language Model for Few-Shot Learning](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2204.14198.pdf)
- 简单的说，输入是图像，输出是视觉特征特征（一组数字）

- USM 原理

- [Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2303.01037.pdf)
- 简单的说，输入是音频信号，输出是音频特征（一组数字）

### Training Dataset

- Gemini 模型目标是多模态模型，因此在训练数据这一块，使用了 Web Document、Books、Code、Image、Audio、Video Data
- Tokenizer 也比较常规，没有花太多篇幅，用 ==SentencePiece Tokenizer== 在大语料训了一个
- Scaling Law 按照 ChinChilla 做的（不是很让人信服）
- 数据处理这块也比较常规

- 启发式规则过滤
- 利用模型进行质量过滤
- 移除有害内容
- 移除评估集内容

- 最终的数据配比，是利用小模型做消融实验得到的，然后 Extend 到大模型上
- 分阶段进行了训练，在快结束训练时，增加了领域相关数据
- 作者们发现数据质量对于模型效果的影响是最大的，这一块值得深入研究

### 特點 1：實實在在的 32K Context Length 預訓練，對於 Long Context 場景更友好

**解釋**：  
32K Context Length 意味著 Gemini 可以處理長度高達 32,000 個 token 的上下文，這樣的上下文長度允許模型在處理長文本、視頻、或其他長時段的多模態數據時保持連貫性和上下文記憶。這對於需要長距離依賴的場景（如視頻解析、書籍生成等）尤其有用。

**分析**：  
這一點是正確的，32K Context Length 是 Gemini 的一大亮點，突破了傳統 Transformer 模型僅支援短上下文的限制，這也讓它特別適合於長文本或長視頻的應用。

---

### 特點 2：Transformer 解碼器作為主要訓練結構，全部多模態數據 Tokenizer 後打包成多模態 Token 序列，進行自回歸（Auto Regressive, AR）訓練

**解釋**：  
Gemini 的核心結構基於 Transformer 解碼器，而不是標準的編碼器-解碼器結構。多模態數據在經過 Tokenizer 處理後，被轉換為同一格式的 token 序列，這樣一來，無論是文字、圖像還是視頻，都可以統一輸入到解碼器中進行自回歸訓練。這樣的設計不同於 GPT-4，它採用不同模態的編碼器或解碼器來通過 Cross Attention（交叉注意力）進行組合。

**分析**：  
這一特點說明了 Gemini 如何在結構上簡化了多模態的處理，使模型可以在一個統一的架構下處理不同的模態數據。自回歸訓練也使模型在生成時能依賴之前生成的內容，有利於文本或視覺生成的連貫性。這種自回歸訓練方式確實展示了其強大之處。

---

### 特點 3：在強化學習（RL）中的 Reward 部分，採用了類似傳統搜索或推薦場景的精排模型，使用多目標 OEC 分數作為綜合 Reward，使對齊信號的可解釋性更強

**解釋**：  
Gemini 的強化學習中使用了 OEC（Objective Evaluation Criteria）分數作為 Reward（回報），這樣的多目標綜合分數可以更好地評估生成內容的質量和可解釋性。這種方法不同於單一目標的 Reward，能夠對應到不同的評價標準，讓生成結果的質量更有保障。

**分析**：  
這個設計在生成任務中相當有創意，因為它能平衡不同的 Reward 需求，對於一些需要精確控制輸出結果的場景（如生成圖像質量的同時保證描述的一致性）有更強的適應性。因此，這樣的 Reward 設計確實有助於提升多模態生成任務中的表現。

---

### 特點 4：選擇多模態數據進行自回歸訓練，同時意味著能擴大訓練數據範疇，Scaling Law 依然穩定發揮作用

**解釋**：  
選擇多模態數據進行自回歸訓練，使得模型能夠有效地擴充數據集的多樣性和數量。Scaling Law（規模定律）指的是當模型的數據量、參數量增加時，性能會穩定提升。Gemini 的架構意味著它可以利用更多數據，並在更大規模下仍然發揮出穩定的表現。

**分析**：  
這個特點符合 Scaling Law 的應用場景，多模態自回歸訓練在 Gemini 中的應用，確實會使模型更易於訓練出更高效的結果。尤其在多模態場景下，有更多樣的數據可以進行擴充訓練，能讓模型在多模態生成和理解中有更好表現。

---

### 特點 5：針對預訓練和對齊方面的數據配比，通過小模型的消融實驗，進而擴展到大模型

**解釋**：  
Gemini 先使用小模型進行消融實驗，來測試不同數據配比的效果，並將成功的設定擴展到更大的模型上。這樣的過程可以在小模型中找到最佳的數據分配方法，以減少大模型訓練中的不確定性，並且確保大模型的效果更加穩定。

**分析**：  
這是一種可靠的訓練策略，先在小模型上驗證數據配比的合理性，再延伸到大模型中，有助於優化訓練效率和模型的最終性能。因此，這個過程和策略是正確且有效的。

---

### 特點 6：數據質量的重要性大於數量

**解釋**：  
在訓練過程中，Gemini 強調了數據質量的重要性，而非單純地增加數據量。這意味著模型在訓練數據上進行了嚴格篩選，確保輸入數據的標註精確度和品質，以達到更好的訓練效果。

**分析**：  
這個觀點非常符合現代大模型訓練的需求，尤其是在多模態數據的訓練中，低質量數據容易引入噪聲並影響模型的學習效果。數據質量的重要性已被許多先進模型驗證，因此這一點在 Gemini 的設計中是合理且正確的。

---

### 總結

Gemini 的設計結合了多模態支援、自回歸訓練、長上下文支持，以及多目標的強化學習 Reward。這些特性使它在多模態生成和理解中具有顯著的優勢，並且在效能、穩定性和應用範圍上較標準 Transformer 架構更加廣泛。Gemini 不僅能處理文本到文本的生成，還可以處理文本、圖像、音訊、視頻等多種模態的輸入與輸出，因此在多模態 AI 模型中具有很高的潛力。


---

### Day1. Prompt Engineering

text prompt





### Day1. Video

[1]
ML develop team - 
1. [grounding](https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview) with google search
   llms and hallucination and grounding and rag
2. [Open AI compatibility ](https://ai.google.dev/gemini-api/docs/openai?hl=zh-tw)

3. Multimodal application

[2]
Reinforcement learning with human feedback(RLHF)
How the Gemini app is using RLHF to help improve its response(fine-tuned LLMs) 
twosteps: sft+RLHF sft
Reward model to penalize responses that are bad, and reward the good

[3]
Model making new discovery



