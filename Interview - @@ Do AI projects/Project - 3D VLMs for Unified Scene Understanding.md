Post
https://www.linkedin.com/posts/ahsenkhaliq_3d-r1-is-out-on-hugging-face-enhancing-reasoning-activity-7358300182335168514-S9JI?utm_source=share&utm_medium=member_desktop&rcm=ACoAABMNj2MBAJSP3cWd4xpiz4wB7qdx43hvW18

Hugging Face
https://huggingface.co/papers/2507.23478

Website
https://aigeeksgroup.github.io/3D-R1/

Github
https://github.com/AIGeeksGroup/3D-R1

3D-R1 is an open-source **generalist** model that enhances the reasoning of 3D VLMs for unified scene understanding. 3D-R1 是一個開源通用模型，旨在增強 3D VLM 的推理能力，從而實現統一的場景理解。

Large vision-language models (VLMs) have made significant strides in 2D visual understanding tasks, sparking interest in extending these capabilities to 3D scene understanding. However, current 3D VLMs often struggle with robust reasoning and generalization due to limitations in high-quality spatial data and the static nature of viewpoint assumptions. To address these challenges, we propose **3D-R1**, a foundation model that enhances the reasoning capabilities of 3D VLMs. Specifically, we first construct a high-quality synthetic dataset with CoT, named Scene-30K, leveraging existing 3D-VL datasets and a data engine based on Gemini 2.5 Pro. It serves as cold-start initialization data for 3D-R1. Moreover, we leverage RLHF policy such as GRPO in the reinforcement learning training process to enhance reasoning capabilities and introduce three reward functions: a perception reward, a semantic similarity reward and a format reward to maintain detection accuracy and answer semantic precision. Furthermore, we introduce a dynamic view selection strategy that adaptively chooses the most informative perspectives for 3D scene understanding. Extensive experiments demonstrate that 3D-R1 delivers an average improvement of 10% across various 3D scene benchmarks, highlighting its effectiveness in enhancing reasoning and generalization in 3D scene understanding.

大型視覺語言模型 (VLM) 在二維視覺理解任務中取得了顯著進展，引發了人們將其能力擴展到 3D 場景理解的興趣。然而，由於高品質空間資料的限制以及視點假設的靜態性，目前的 3D VLM 往往難以實現穩健的推理和泛化。為了應對這些挑戰，我們提出了 3D-R1，這是一個增強 3D VLM 推理能力的基礎模型。具體而言，我們首先利用現有的 3D-VL 資料集和基於 Gemini 2.5 Pro 的資料引擎，建立了一個基於 CoT 的高品質合成資料集，名為 Scene-30K。該資料集將作為 3D-R1 的冷啟動初始化資料。此外，我們在強化學習訓練過程中利用 RLHF 策略（例如 GRPO）來增強推理能力，並引入三個獎勵函數：感知獎勵、語義相似性獎勵和格式獎勵，以保持檢測準確率和答案語義準確率。此外，我們還引入了動態視圖選擇策略，可以自適應地選擇最具資訊量的視角來理解 3D 場景。大量實驗表明，3D-R1 在各種 3D 場景基準測試中平均提升了 10%，凸顯了其在增強 3D 場景理解的推理和泛化能力方面的有效性。

### Key Contributions

#### Foundation Model

A pioneering 3D-VLM leverages reinforcement learning and dynamic view selection to enhance reasoning capabilities in 3D scene understanding. 開創性的 3D-VLM 利用強化學習和動態視圖選擇來增強 3D 場景理解的推理能力。

#### Scene-30K Data Engine

A high-quality 30K scene CoT dataset is constructed with a data engine based on Gemini-Pro and existing 3D-VL datasets. 利用基於Gemini-Pro和現有3D-VL資料集的資料引擎建構了高品質的30K場景CoT資料集。

#### SOTA Performance

Extensive experiments demonstrate that 3D-R1 achieves an average improvement of 10% across 7 downstream tasks and various 3D scene benchmarks. 大量實驗表明，3D-R1 在 7 個下游任務和各種 3D 場景基準測試中實現了平均 10% 的提升。

![[Pasted image 20250807003104.png]]



|                                                |                                                                                                                                                                                                                          |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1. 3D場景密集描述 (3D Scene Dense Captioning, 3D-DC) | 此任務要求模型不僅僅是概括性地描述整個場景，而是要偵測出場景中所有重要的物體和區域，並為每一個物體或區域生成一句詳細、獨立的文字描述。這個描述需要包含物體的屬性（如顏色、材質）、位置關係以及它們在場景中的作用。這是一個「密集」的過程，因為目標是為場景中的多個點生成大量的、豐富的描述。<br><br>Q: Please locate the sofa<br>A: "這是一張位於房間中央的灰色L型布藝沙發，上面放著兩個黃色的抱枕。" |
| 2. 3D物體描述 (3D Object Captioning)               | 相較於密集描述，3D物體描述的焦點更為集中。任務是針對場景中**某一個特定**的物體，生成一段詳細的描述。這個描述不僅要說明物體本身的外觀（形狀、顏色、紋理），還要結合其在3D空間中的上下文，例如它的位置、朝向以及與周圍其他物體的關係。<br><br>Q: Please describe this object<br>A: "這是一個白色的陶瓷馬克杯，帶有一個把手。它被放置在深色木質茶几的右側，靠近遙控器的地方。         |
| 3. 3D視覺定位 (3D Visual Grounding, 3D-VG)         | 這個任務正好與前兩者相反。不是「看圖說話」，而是「聽話找圖」。模型會接收一段自然語言描述（指令或問題），然後需要在複雜的3D場景中，準確地定位並標示出這段話所指代的那個物體或區域。<br><br>Q: "找到那個放在書架最上層的藍色封面的書。"<br>在它的視覺界面中，代表整個書架的3D點雲會被分析，最終只有那本符合「書架最上層」和「藍色封面」這兩個條件的書會被高亮顯示或框選出來。                         |
| 4. 3D問答 (3D Question Answering, 3D-QA)         | 3D-QA是指模型需要觀察一個3D場景，然後回答關於這個場景的具體問題。這些問題可以涵蓋物體的數量、屬性、空間關係、功能等多個層面。這需要模型不僅能識別物體，還要能進行計數、比較和空間推理。<br><br>Q: "餐桌周圍有幾把椅子？"<br>A: "餐桌周圍有四把椅子。"                                                                                |
| 5. 3D對話 (3D Dialogue)                          | 3D對話是3D-QA的延伸，它不是一次性的「一問一答」，而是一個持續的、有多輪交流的對話過程。在對話中，後續的問題和回答可能會依賴於之前的交流內容。這要求模型具備記憶和上下文理解能力。<br><br>Q: "我的書桌上有檯燈嗎？"<br>A: "是的，您的書桌左側有一盞黑色的檯燈。"<br>Q: "那它是開著的嗎？"<br>A: "是的，它是亮著的。"                                        |
| 6. 3D推理 (3D Reasoning)                         | 3D推理是一個更高級的認知任務。它不僅僅是描述或定位，而是需要模型根據場景中的視覺訊息，結合常識或物理知識，進行邏輯推斷，以回答一些「為什麼」或「怎麼樣」的深層次問題。<br><br>Image - 床上的被子是凌亂的，旁邊的地板上有一本書攤開著，床頭燈還亮著。<br>Q: "這個房間的主人可能剛剛在做什麼？"<br>A: "這個房間的主人很可能剛剛在床上看書，並且可能剛剛離開不久，因為燈還沒有關。"               |
| 7. 3D規劃 (3D Planning)                          | 3D規劃是將視覺理解轉化為實際行動的終極任務。模型需要分析當前的3D場景，理解一個高層次的指令（例如「整理房間」），然後將這個複雜指令分解成一系列具體、可執行的子步驟，並規劃出完成這些步驟的物理動作序列。<br><br>Q: "幫我準備好喝咖啡的桌子。"<br>模型的3D規劃過程: 場景分析 ->任務分解(移動到廚櫃旁,打開櫥櫃門,抓取一個乾淨的馬克杯...) ->執行                               |


#### 1. 3D場景密集描述 (3D Scene Dense Captioning, 3D-DC)

![[Pasted image 20250827023206.png]]
**任務解釋：** 此任務要求模型不僅僅是概括性地描述整個場景，而是要偵測出場景中所有重要的物體和區域，並為每一個物體或區域生成一句詳細、獨立的文字描述。這個描述需要包含物體的屬性（如顏色、材質）、位置關係以及它們在場景中的作用。這是一個「密集」的過程，因為目標是為場景中的多個點生成大量的、豐富的描述。

**室內場景的目標：** 在室內，3D-DC的目標是產生一份關於房間佈局、家具、裝飾品等的詳細清單和描述。這能讓機器對空間有非常細緻的理解，知道「什麼東西在哪裡」以及「它長什麼樣子」。

**具體舉例說明：** 假設一個AI掃描了您的客廳，它的輸出會是這樣的：

- **偵測到沙發區域：** "這是一張位於房間中央的灰色L型布藝沙發，上面放著兩個黃色的抱枕。"
    
- **偵測到茶几：** "沙發前面有一張長方形的深色木質茶几，上面放著一個電視遙控器和一個白色馬克杯。"
    
- **偵測到電視櫃：** "靠牆擺放著一個白色的電視櫃，上面有一台大尺寸的黑色電視。"
    
- **偵測到角落的盆栽：** "在窗戶旁邊的角落裡，有一盆綠色的龜背竹，放在一個陶瓷花盆裡。"




#### 2. 3D物體描述 (3D Object Captioning)
![[Pasted image 20250827023239.png]]
**任務解釋：** 相較於密集描述，3D物體描述的焦點更為集中。任務是針對場景中**某一個特定**的物體，生成一段詳細的描述。這個描述不僅要說明物體本身的外觀（形狀、顏色、紋理），還要結合其在3D空間中的上下文，例如它的位置、朝向以及與周圍其他物體的關係。

**室內場景的目標：** 幫助機器精準地識別和理解單一物品。當您對機器人說「幫我拿桌上那本書」時，它需要先在腦中對桌上的每個物體進行準確的「描述」，才能分辨出哪一個是「書」。

**具體舉例說明：** 在同一個客廳場景中，如果我們選定**茶几上的馬克杯**作為目標：

- **模型生成的描述會是：** "這是一個白色的陶瓷馬克杯，帶有一個把手。它被放置在深色木質茶几的右側，靠近遙控器的地方。"
    

這個描述不僅說了杯子是「白色陶瓷」的，還點出了它在茶几上的具體位置（右側）和與遙控器的鄰近關係，這就是3D上下文的體現。




#### 3. 3D視覺定位 (3D Visual Grounding, 3D-VG)
![[Pasted image 20250827023259.png]]
**任務解釋：** 這個任務正好與前兩者相反。不是「看圖說話」，而是「聽話找圖」。模型會接收一段自然語言描述（指令或問題），然後需要在複雜的3D場景中，準確地定位並標示出這段話所指代的那個物體或區域。

**室內場景的目標：** 這是實現人機互動的關鍵。它讓機器能夠理解人類的空間指令。例如，當你告訴掃地機器人「去清理沙發底下」時，它就需要運用3D-VG能力來定位「沙發底下」這個具體的三維空間。

**具體舉例說明：** 您對一個搭載AI的智慧眼鏡或機器人說：

- **您的指令：** "找到那個放在書架最上層的藍色封面的書。"
    
- **模型的反應：** 在它的視覺界面中，代表整個書架的3D點雲會被分析，最終只有那本符合「書架最上層」和「藍色封面」這兩個條件的書會被高亮顯示或框選出來。




#### 4. 3D問答 (3D Question Answering, 3D-QA)
![[Pasted image 20250827023314.png]]
**任務解釋：** 3D-QA是指模型需要觀察一個3D場景，然後回答關於這個場景的具體問題。這些問題可以涵蓋物體的數量、屬性、空間關係、功能等多個層面。這需要模型不僅能識別物體，還要能進行計數、比較和空間推理。

**室內場景的目標：** 讓機器成為一個場景的「專家」，能夠回答用戶對當前環境的任何疑問。這對於視障人士輔助、或遠程場景勘查等應用非常重要。

**具體舉例說明：** 您向一個家庭助理AI提問關於廚房的場景：

- **問題1 (計數):** "餐桌周圍有幾把椅子？"
    
- **模型回答:** "餐桌周圍有四把椅子。"
    
- **問題2 (屬性):** "冰箱是什麼顏色的？"
    
- **模型回答:** "冰箱是銀色的。"
    
- **問題3 (空間關係):** "微波爐是放在流理台上還是冰箱上面？"
    
- **模型回答:** "微波爐是放在流理台上的。"





#### 5. 3D對話 (3D Dialogue)
![[Pasted image 20250827023326.png]]
**任務解釋：** 3D對話是3D-QA的延伸，它不是一次性的「一問一答」，而是一個持續的、有多輪交流的對話過程。在對話中，後續的問題和回答可能會依賴於之前的交流內容。這要求模型具備記憶和上下文理解能力。

**室內場景的目標：** 創造更自然、更流暢的人機互動體驗。用戶不需要在每個問題中重複已經提過的訊息，機器人可以像一個真正的助手一樣跟隨你的思路。

**具體舉例說明：** 您和一個智慧管家機器人之間關於書房的對話：

- **您 (第一輪):** "我的書桌上有檯燈嗎？"
    
- **機器人:** "是的，您的書桌左側有一盞黑色的檯燈。"
    
- **您 (第二輪):** "那它是開著的嗎？" (這裡的 "它" 指代了上一輪提到的檯燈)
    
- **機器人:** "是的，它是亮著的。"
    
- **您 (第三輪):** "幫我把它關掉。"
    
- **機器人:** "好的，我已經將書桌上的檯燈關掉了。"





#### 6. 3D推理 (3D Reasoning)
![[Pasted image 20250827023344.png]]
**任務解釋：** 3D推理是一個更高級的認知任務。它不僅僅是描述或定位，而是需要模型根據場景中的視覺訊息，結合常識或物理知識，進行邏輯推斷，以回答一些「為什麼」或「怎麼樣」的深層次問題。

**室內場景的目標：** 讓機器具備初步的「理解力」和「判斷力」。它能推斷物體的功能、狀態，甚至人的意圖。例如，看到地上有水漬和一個倒了的杯子，機器能推斷出「可能有人不小心把水打翻了」。

**具體舉例說明：** 模型分析一個臥室的場景：

- **場景觀察：** 床上的被子是凌亂的，旁邊的地板上有一本書攤開著，床頭燈還亮著。
    
- **推理問題：** "這個房間的主人可能剛剛在做什麼？"
    
- **模型的推理回答：** "這個房間的主人很可能剛剛在床上看書，並且可能剛剛離開不久，因為燈還沒有關。"
    

這個回答無法直接從任何單一物體上看出來，而是綜合了多個線索（凌亂的被子、攤開的書、亮著的燈）進行邏輯鏈接後得出的結論。





#### 7. 3D規劃 (3D Planning)
![[Pasted image 20250827023357.png]]
**任務解釋：** 3D規劃是將視覺理解轉化為實際行動的終極任務。模型需要分析當前的3D場景，理解一個高層次的指令（例如「整理房間」），然後將這個複雜指令分解成一系列具體、可執行的子步驟，並規劃出完成這些步驟的物理動作序列。

**室內場景的目標：** 這是實現真正意義上的自主家庭機器人的核心能力。讓機器人能夠自主地在真實、複雜的家庭環境中完成任務。

**具體舉例說明：** 您對一個家務機器人下達指令：

- **高層次指令：** "幫我準備好喝咖啡的桌子。"
    
- **模型的3D規劃過程：**
    
    1. **場景分析：** 首先，掃描並理解廚房和客廳的3D佈局，定位咖啡機、櫥櫃裡的馬克杯、以及客廳的茶几。
        
    2. **任務分解：**
        
        - 步驟1：移動到廚櫃旁。
            
        - 步驟2：打開櫥櫃門。
            
        - 步驟3：抓取一個乾淨的馬克杯。
            
        - 步驟4：關上櫥櫃門。
            
        - 步驟5：將馬克杯移動到咖啡機下方。
            
        - 步驟6：操作咖啡機，製作一杯咖啡。
            
        - 步驟7：等待咖啡製作完成。
            
        - 步驟8：小心地拿起裝有咖啡的馬克杯。
            
        - 步驟9：規劃一條從廚房到客廳茶几的無障礙路徑。
            
        - 步驟10：將咖啡杯平穩地放在茶几上。
            
    3. **執行：** 按照規劃好的步驟序列，控制機械臂和輪子完成整個任務。





### Architecture

Our 3D-R1 model is designed based on Qwen2.5-VL-7B-Instruct and trained with the high-quality synthetic Scene-30K dataset. It takes text, multi-view images, 3D point clouds, and depth maps as input and formulates comprehensive 3D tasks as autoregressive sequence prediction. 我們的 3D-R1 模型是基於 Qwen2.5-VL-7B-Instruct 設計，並使用高品質合成 Scene-30K 資料集進行訓練。該模型以文字、多視角圖像、3D 點雲和深度圖作為輸入，並以自回歸序列預測的形式實現綜合 3D 任務。

![[Pasted image 20250827102455.png]]

### CoT Data Engine

The point cloud of a scene is first sent to scene dscription generator to get a description of the scene. Then based on the description, we apply Gemini-Pro to synthetic CoT data. 首先將場景的點雲傳送到場景描述產生器，以獲得場景的描述。然後，我們基於該描述，將 Gemini-Pro 應用於合成的 CoT 數據。

![[Pasted image 20250827102521.png]]


### Reinforcement Learning

The policy model generates N outputs from a point cloud and question. Then perception IoU, semantic CLIP-similarity, and format-adherence rewards are computed, grouped, and combined with a KL term to a frozen reference model to update the policy. 策略模型根據點雲和問題產生 N 個輸出。然後，計算感知 IoU、語意 CLIP 相似度和格式遵循獎勵，並進行分組，並與 KL 項組合到凍結參考模型中，以更新策略。

![[Pasted image 20250827102538.png]]


### Multi-Task Generalist

3D-R1 is a generalist model capable of handling various downstream tasks and applications in a zero-shot manner with incredible generalizability, significantly reducing the need for expensive adaptation. 3D-R1 是一種通用模型，能夠以零樣本方式處理各種下游任務和應用，具有令人難以置信的通用性，大大減少了昂貴的適應性需求。
![[Pasted image 20250827102550.png]]



```
3D-R1 is out on Hugging Face  
  
Enhancing Reasoning in 3D VLMs for Unified Scene Understanding  
  
3D-R1 is an open-source generalist model that enhances the reasoning of 3D VLMs for unified scene understanding  
  
discuss with author: [https://lnkd.in/eShYx2AU](https://lnkd.in/eShYx2AU)
```
