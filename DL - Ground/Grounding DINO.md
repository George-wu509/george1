

十分钟解读Grounding DINO-根据文字提示检测任意目标 - 李小羊学AI的文章 - 知乎
https://zhuanlan.zhihu.com/p/627646794

由文本提示检测图像任意目标(Grounding DINO)的使用以及全网最详细源码讲解 - 共由的文章 - 知乎
https://zhuanlan.zhihu.com/p/680808735

Reference:
LLMDet and MM-Grounding-DINO on HuggingFace (2025.08.13)
https://www.linkedin.com/posts/niels-rogge-a3b7a3127_some-new-impressive-open-vocabulary-detectors-activity-7361380179752939520-vMyg/?utm_source=share&utm_medium=member_android&rcm=ACoAABMNj2MBAJSP3cWd4xpiz4wB7qdx43hvW18
These models are so-called "open vocabulary" or "zero-shot" detection models. This means that they can detect objects in an image just via prompting, no training involved!





首先，您的總結非常到位。以功能來說，GLIP和GroundingDINO的目標一致：**實現「接地氣」的開放集物件偵測（Grounded Open-Set Object Detection）**。

您可以給定同一張圖片和同一段描述性文字，例如：

> **「A woman holds a blow dryer, wearing protective goggles」** （一位女士拿著吹風機，戴著護目鏡）

理論上，GLIP和GroundingDINO都能根據這段話，在圖上定位並框出「woman」、「blow dryer」和「protective goggles」這幾個物件。

而您的第二個判斷也符合學術界和業界的共識：**GroundingDINO的綜合效能，尤其是在開放集（檢測從未見過的物體類別）和對複雜語言的理解能力上，普遍優於GLIP。**

---

### 效能差異的根源：為何GroundingDINO更強？

這個差異的根源主要來自三個層面：**模型架構的先進性**、**跨模態融合的深度**，以及**訓練方法的不同**。簡而言之，GroundingDINO站在了更現代、更適合多模態任務的巨人（DINO/DETR）的肩膀上。

讓我們來詳細解析這兩者的模型架構。

### GLIP (Grounded Language-Image Pre-training) 模型架構解析

GLIP的開創性在於**首次將「物件偵測」和「片語接地」兩個任務在預訓練階段進行了統一**。它巧妙地將傳統的物件偵測任務（輸入圖像，輸出類別和框）轉化為片語接地任務（輸入圖像和文本，輸出對應的框）。

其架構主要包含以下幾個部分：

1. **圖像編碼器 (Image Encoder)**：
    
    - **作用**：提取圖像的視覺特徵。
        
    - **常用模型**：通常使用強大的視覺Transformer，如 **Swin Transformer**。它會將輸入圖像轉換成一系列的視覺特徵圖 (Feature Maps)。
        
2. **文本編碼器 (Text Encoder)**：
    
    - **作用**：提取文本的語義特徵。
        
    - **常用模型**：使用強大的語言模型，通常是 **BERT** 或其變體。它會將輸入的句子（如 "a woman holds a blow dryer..."）轉換成每個詞或子詞 (token) 的特徵向量。
        
3. **跨模態深度融合模塊 (Cross-Modality Deep Fusion)**：
    
    - **這是GLIP的核心**。在分別提取完圖像和文本特徵後，GLIP會通過一個**融合編碼器**讓兩者進行深度交互。
        
    - **機制**：這個融合模塊通常採用多層的**跨模態注意力機制 (Cross-Attention)**。簡單來說，文本特徵會「關注」圖像特徵中的相關區域，反之亦然。這使得模型能夠學習到「語言感知的視覺特徵」。例如，當文本中出現「blow dryer」時，圖像中吹風機區域的特徵會被強化。
        
4. **預測頭 (Prediction Head)**：
    
    - **機制**：GLIP的訓練目標是**對比學習**。它會計算圖像中的每個候選區域 (region proposals) 與文本中的每個片語 (phrase) 之間的**對齊分數 (alignment score)**，通常是計算兩者特徵向量的點積或餘弦相似度。
        
    - **輸出**：分數最高的「區域-片語」配對，就是最終的偵測結果。
        

**GLIP的侷限性**：GLIP雖然創新，但其底層的偵測思想仍然基於**傳統的物件偵測器框架**，需要生成大量的候選區域，然後與文本進行匹配。這在架構上不如後來的端到端模型簡潔高效。

### GroundingDINO 模型架構解析

GroundingDINO的設計哲學是「**將最強的閉集檢測器（DINO）改造成一個開放集檢測器**」。它完美地繼承了DINO/DETR這類基於Transformer的端到端檢測器的所有優點。

其架構可以概括為一個**「雙編碼器-單解碼器」**的設計：

1. **圖像編碼器 (Image Encoder)**：
    
    - **作用**：提取多尺度的圖像視覺特徵。
        
    - **常用模型**：同樣是 **Swin Transformer** 等。
        
2. **文本編碼器 (Text Encoder)**：
    
    - **作用**：提取文本片語的語義特徵。
        
    - **常用模型**：同樣是 **BERT** 等。
        
3. **特徵增強器 (Feature Enhancer) - 關鍵差異點1**：
    
    - **GLIP**的融合發生在一個集中的模塊，而**GroundingDINO**的融合設計得更為精巧和深入。
        
    - **機制**：它包含多個堆疊的**跨模態注意力層**，不僅讓文本特徵去增強圖像特徵，也讓圖像特徵去增強文本特徵。這種**雙向的、貫穿始終的深度融合**，使得兩種模態的資訊在進入最終的解碼器之前，就已經達到了非常高層次的對齊。
        
4. **語言引導的查詢選擇 (Language-Guided Query Selection) - 關鍵差異點2**：
    
    - 這是繼承自DINO/DETR的**核心機制**。傳統檢測器需要成千上萬的錨框 (anchors) 或候選框，而DINO使用少量的、可學習的**物件查詢 (Object Queries)**。
        
    - **GroundingDINO的創新**：它的Object Queries**不是隨機初始化**的，而是**根據輸入文本的特徵來選擇和初始化**的。這意味著模型從一開始就帶著「我要找什麼」的先驗知識去圖像中尋找目標，極大地提高了效率和準確性。
        
5. **跨模態解碼器 (Cross-Modality Decoder) - 關鍵差異點3**：
    
    - **機制**：這是一個標準的Transformer解碼器。它接收由語言引導的Object Queries，並讓這些Queries與融合後的圖像特徵圖進行反覆的交互（通過自注意力和交叉注意力）。
        
    - **端到端輸出**：解碼器的輸出直接就是最終的**「邊界框 + 對應片語」**，無需非極大值抑制（NMS）等後處理步驟。這種端到端的設計極大地簡化了流程，並允許整個模型進行聯合優化。
        

### 核心差異總結表

|特性|GLIP|GroundingDINO|為什麼GroundingDINO更優|
|---|---|---|---|
|**底層檢測框架**|基於傳統檢測器思想，依賴**區域-片語對比學習**。|基於現代**端到端Transformer檢測器 (DINO/DETR)**。|架構更先進、更簡潔。無需NMS等手工設計的後處理，整個模型可以進行更徹底的端到端優化。|
|**跨模態融合**|在一個集中的**融合模塊**中進行。|在**特徵增強器**和**解碼器**中進行**貫穿始終**的雙向深度融合。|資訊交互更充分、更深入。文本不僅指導檢測，還在檢測的每一步都參與其中。|
|**目標定位方式**|依賴大量的候選區域與文本匹配。|使用少量由**語言引導的Object Queries**主動去尋找目標。|定位更高效、更具目的性。從「大海撈針」變為「按圖索驥」。|
|**訓練數據集**|依賴大量的「圖像-文本」配對數據和傳統檢測數據進行預訓練。|同樣使用大規模圖文對數據，並受益於更強的閉集檢測器預訓練。|其先進架構能更有效地從大規模數據中學習，泛化能力更強。|

---

### 近期還有沒有類似但功能最強的模型？

截至目前，**GroundingDINO系列本身仍然是這個領域的標竿和最強的代表之一**。其後續的發展主要體現在對自身的改進和優化上：

1. **GroundingDINO 1.5 (Pro / Edge)**：
    
    - 這是原團隊對GroundingDINO的**官方升級版**。
        
    - **GroundingDINO 1.5 Pro**：追求極致的性能，通過使用更強大的骨幹網路、更優化的融合策略和更豐富的訓練數據，在各大基準測試上刷新了紀錄，是目前**性能最強**的開源模型之一。
        
    - **GroundingDINO 1.5 Edge**：則專注於在保持高精度的同時，平衡**速度和效率**，使其能夠在邊緣設備或對延遲敏感的應用中部署。
        
2. **MM-Grounding-DINO**：
    
    - 這是在著名的OpenMMLab框架下對GroundingDINO的一個**開源復現和改進版本**。
        
    - 它提供了一套完整的、可復現的訓練和微調流程，讓研究者和開發者能夠更容易地在自己的數據集上使用和改進GroundingDINO，其性能也與官方版本相當甚至在某些任務上更優。
        

**結論**：目前，如果您要尋找一個功能最強、最受認可的「接地氣」物件偵測模型，**GroundingDINO 1.5 Pro** 是當之無愧的首選。雖然有其他新的研究在不斷湧現，但還沒有一個像GroundingDINO這樣，在性能、開源生態和社區影響力上都達到如此高度的顛覆性模型。