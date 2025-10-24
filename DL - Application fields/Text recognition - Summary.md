

|                                 |     |
| ------------------------------- | --- |
| [[#### NLP multimodal outline]] |     |
|                                 |     |
|                                 |     |



#### NLP multimodal outline

|                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **DATA**                                                                                                                                                                                                                                                                                                                                                     |
| **MODEL**<br>1. sequence-to-sequence models(attention mechanisms)<br>2. multimodal models                                                                                                                                                                                                                                                                    |
| **MODEL TRAINING**<br>1. optimization, regularization, model scaling, and distributed training<br>2. large-scale multimodal foundation models and techniques for fine-tuning/adaptation<br>3. evaluation methodologies for sequence and multimodal models                                                                                                    |
| **APPLICATION - multimodal tasks**<br>                                                                                                                                                                                                                                                                                                                       |
| ==A: Image -> Text== <br><br>1. **text recognition 文字辨識** - 影像中的文字轉換為機器可讀、可編輯的文字格式<br><br>2. **OCR 光學字元辨識** - Text Detection在影像中定位文字的位置；以及Text Recognition將定位出的文字影像轉換為文字字串, Text Recognition可以視為 OCR 流程中的一個核心環節<br><br>3. **image captioning 影像字幕** - 讓電腦自動生成一段能夠準確流暢地描述影像內容的文字<br><br>4. **document understanding 文件理解** - 更深層次的目標是分析文件的整體結構和語意<br><br><br> |
| ==B: Text -> Text==<br><br>1. **machine translation 機器翻譯** - 將一種自然語言（源語言）的文本翻譯成另一種自然語言（目標語言）<br><br>2. **text generation 文字生成** - 「提示」(prompt)輸出一段新生成的、與輸入提示相關的文字<br><br>3. **Text Summarization / QA / Sentiment Analysis**: 專注於對現有文字的理解、推理和濃縮。<br><br>                                                                                                     |
| ==C: text  -> Image==<br><br>1. **Text-to-Image Generation**: 將語言模態的複雜指令，轉換為視覺模態的像素輸出<br><br>                                                                                                                                                                                                                                                                |
| ==D: Speech <--> text==<br><br>1. **speech-to-text 語音轉文字** -自動語音辨識將人類的語音訊號轉換為對應的文字內容<br><br>2. **Text-to-Speech (TTS)**: 將文字轉換為聲學訊號<br><br>                                                                                                                                                                                                                  |
| ==E: Multimodal==<br><br>1. **Visual Question Answering (VQA)**: 需要同時深度理解視覺（影像）和語言（問題）兩種模態，並生成語言（答案）<br><br>                                                                                                                                                                                                                                                 |





### 1. 文字辨識 (Text Recognition)

- **目的 (Purpose):** 文字辨識的主要目的是將影像中的文字轉換為機器可讀、可編輯的文字格式。它專注於辨識和轉譯單個字詞或文字行，無論這些文字是出現在自然場景（如街景、產品包裝）中，還是掃描的文件裡。這項技術是許多更複雜應用的基礎，例如車牌辨識、名片資訊擷取等。
    
- **輸入輸出 (Input/Output):**
    
    - **輸入:** 包含文字的影像檔案（例如 JPEG, PNG）。這些影像可以是來自相機拍攝的自然場景照片，也可以是掃描的文件。
        
    - **輸出:** 純文字字串 (`string`) 或帶有每個文字位置座標的結構化資料（例如 JSON）。
        
- **主流方法 (Mainstream Methods):**
    
    - **卷積循環神經網路 (CRNN):** 這是一種經典且非常流行的方法。它首先使用卷積神經網路（CNN）來提取影像的視覺特徵，然後利用循環神經網路（RNN，特別是 LSTM 或 GRU）來處理序列化的特徵，捕捉文字的上下文關係。
        
    - **注意力機制 (Attention Mechanism):** 在 CRNN 的基礎上加入注意力機制，允許模型在解碼（輸出文字）的每一步都能專注於影像中最相關的區域，這對於處理不規則或彎曲的文字特別有效。
        
    - **Transformer 為基礎的模型:** 近年來，基於 Transformer 架構的模型（如 ViT - Vision Transformer）也被應用於文字辨識。這類模型透過自注意力機制（Self-Attention）直接從影像區塊（patches）中學習特徵和上下文關係，在許多基準測試中取得了頂尖的成果。
        
- **常用資料集 (Datasets):**
    
    - **IIIT-5K:** 包含 5000 張從網路收集的場景文字影像。
        
    - **Street View Text (SVT):** 從 Google 街景中擷取的影像，包含許多戶外場景的文字。
        
    - **ICDAR (International Conference on Document Analysis and Recognition) 系列資料集:** 例如 ICDAR 2013, 2015, 2019 等，是學術界評估文字偵測與辨識模型的標準基準。
        
    - **SynthText:** 一個大規模的合成資料集，透過將文字渲染到自然場景影像上，產生數百萬張訓練樣本。
        

---

### 2. 文件理解 (Document Understanding)

- **目的 (Purpose):** 文件理解不僅僅是辨識文件中的文字，其更深層次的目標是分析文件的整體結構和語意，並從中提取有價值的資訊。它需要理解文字區塊之間的關係、文件的排版佈局（如表格、列表、標題），以及內容的含義。常見應用包括自動處理發票、合約審查、履歷篩選和保險理賠文件分析。
    
- **輸入輸出 (Input/Output):**
    
    - **輸入:** 文件影像或數位文件（如 PDF, DOCX, PNG）。
        
    - **輸出:** 結構化的資料，通常是 JSON 格式。這些資料包含了提取出的關鍵資訊（Key-Value Pairs），例如發票中的「總金額：$150」、表格內容、或是合約中的「簽約方」和「生效日期」。
        
- **主流方法 (Mainstream Methods):**
    
    - **佈局分析模型 (Layout Analysis Models):** 這些模型（如 LayoutLM, DiT）結合了文字、影像和佈局資訊。它們通常使用 Transformer 架構，在預訓練階段同時學習文字的語意和文字在頁面上的二維位置關係，從而能更好地理解文件結構。
        
    - **圖神經網路 (Graph Neural Networks, GNN):** 將文件中的文字區塊視為圖的節點，文字間的空間關係（如相鄰、對齊）視為邊，利用 GNN 來學習和推斷這些區塊的功能和關係（例如，哪個標題對應哪個段落）。
        
    - **OCR + 自然語言處理 (NLP):** 這是一種較傳統的管線方法。首先使用 OCR 技術提取所有文字和其位置，然後將這些文字輸入到 NLP 模型（如 BERT）中進行命名實體辨識（NER）、關係抽取等任務，以找出關鍵資訊。
        
- **常用資料集 (Datasets):**
    
    - **SROIE (Scanned Receipts OCR and Information Extraction):** 包含大量掃描收據影像及其標註的關鍵資訊。
        
    - **FUNSD (Form Understanding in Noisy Scanned Documents):** 一個用於理解掃描表格中語意實體的資料集。
        
    - **CORD (Consolidated Receipt Dataset for Post-OCR Parsing):** 專注於從收據中提取結構化資訊的資料集。
        
    - **DocVQA:** 用於文件視覺問答的資料集，模型需要根據文件的影像內容回答相關問題。
        

---

### 3. 光學字元辨識 (OCR)

- **目的 (Purpose):** OCR (Optical Character Recognition) 是一個廣泛的術語，指的是將影像中的印刷體或手寫文字轉換為機器可讀文字的整個過程。它通常包含兩個主要階段：**文字偵測 (Text Detection)**，即在影像中定位文字的位置；以及 **文字辨識 (Text Recognition)**，即將定位出的文字影像轉換為文字字串。因此，前面提到的「Text Recognition」可以視為 OCR 流程中的一個核心環節。
    
- **輸入輸出 (Input/Output):**
    
    - **輸入:** 包含文字的影像或 PDF 文件。
        
    - **輸出:** 包含文字內容和其在原始影像中位置座標的結構化資料（例如，每個字詞的邊界框 `bounding box` 和對應的文字）。
        
- **主流方法 (Mainstream Methods):**
    
    - **傳統管線方法:** 先用一個模型進行文字偵測（如 EAST, DBNet），找到文字區域的邊界框，然後將每個邊界框內的影像切割出來，再送入另一個文字辨識模型（如 CRNN）進行辨識。
        
    - **端到端 (End-to-End) 方法:** 使用單一的深度學習模型同時完成文字的偵測和辨識。這類模型直接從輸入影像生成最終的文字結果，架構更簡潔，訓練也更直接。
        
- **常用資料集 (Datasets):** OCR 的資料集通常與文字偵測和辨識的資料集重疊，例如 **ICDAR 系列資料集**和 **COCO-Text**。此外，許多商業 OCR 引擎（如 Google Vision API, Tesseract）也都是基於大量的內部資料集訓練的。
    

---

### 4. 機器翻譯 (Machine Translation)

- **目的 (Purpose):** 機器翻譯的目標是利用電腦程式自動將一種自然語言（源語言）的文本翻譯成另一種自然語言（目標語言），同時盡可能保持原文的語意、風格和流暢度。
    
- **輸入輸出 (Input/Output):**
    
    - **輸入:** 一段源語言的文字字串。
        
    - **輸出:** 一段翻譯後的目標語言的文字字串。
        
- **主流方法 (Mainstream Methods):**
    
    - **神經機器翻譯 (Neural Machine Translation, NMT):** 這是目前絕對的主流方法，完全取代了早期的統計機器翻譯（SMT）。
        
    - **基於 RNN 的 Seq2Seq 模型:** 早期的 NMT 模型通常採用編碼器-解碼器（Encoder-Decoder）架構，使用 RNN（如 LSTM）來讀取源語言句子並生成目標語言句子，通常會搭配注意力機制。
        
    - **Transformer:** 自 2017 年推出以來，Transformer 模型憑藉其自注意力機制和並行計算能力，成為機器翻譯領域的標竿架構。Google 翻譯、DeepL 等頂級翻譯服務都基於此類模型。大型語言模型（如 GPT 系列、PaLM）在經過微調後也能執行高品質的機器翻譯任務。
        
- **常用資料集 (Datasets):** 機器翻譯需要大量的平行語料庫（即成對的、互相翻譯的句子）。
    
    - **WMT (Workshop on Machine Translation) 提供的資料集:** 每年 WMT 會議都會發布用於比賽和評測的資料集，涵蓋多種語言對。
        
    - **Europarl Corpus:** 包含歐洲議會的會議記錄，涵蓋多種歐盟官方語言。
        
    - **OpenSubtitles:** 從電影和電視節目的字幕中收集的大規模多語言平行語料庫。
        

---

### 5. 文字生成 (Text Generation)

- **目的 (Purpose):** 文字生成的目標是讓電腦模型能夠像人類一樣創造出通順、連貫且有意義的文字。這是一個非常廣泛的領域，涵蓋了從寫作輔助、內容創作、對話系統到程式碼生成等多種應用。
    
- **輸入輸出 (Input/Output):**
    
    - **輸入:** 通常是一個「提示」(prompt)，可以是一個問題、一句話的開頭、一些關鍵字，或是一段需要被摘要或改寫的文本。
        
    - **輸出:** 一段新生成的、與輸入提示相關的文字。
        
- **主流方法 (Mainstream Methods):**
    
    - **大型語言模型 (Large Language Models, LLMs):** 這是當前最主流且效果最好的方法。這類模型（如 GPT-3/4, LLaMA, PaLM）在海量的文本資料上進行預訓練，學習語言的規律、事實知識和推理能力。
        
    - **基於 Transformer 的解碼器架構:** 大多數成功的文字生成模型都採用了 Transformer 的解碼器部分（Decoder-only architecture），這種架構非常適合根據前面的文字預測下一個詞，從而逐詞生成完整的句子和段落。
        
    - **生成對抗網路 (GANs):** 雖然在影像生成領域非常成功，但在文字生成領域，由於文字的離散性，GAN 的應用相對較少且訓練困難，不如 LLMs 普及。
        
- **常用資料集 (Datasets):** LLMs 的訓練通常使用來自網路的超大規模文本資料。
    
    - **Common Crawl:** 一個包含數十億個網頁的公開資料集。
        
    - **Wikipedia:** 維基百科的所有文章，是高品質、知識密集的文本來源。
        
    - **BooksCorpus / Gutenberg:** 包含大量書籍內容的資料集。
        

---

### 6. 語音轉文字 (Speech-to-Text)

- **目的 (Purpose):** 也稱為自動語音辨識（Automatic Speech Recognition, ASR），其目的是將人類的語音訊號轉換為對應的文字內容。這是語音助理（如 Siri, Google Assistant）、會議記錄、語音輸入法等應用的核心技術。
    
- **輸入輸出 (Input/Output):**
    
    - **輸入:** 音訊檔案（如 WAV, MP3）或即時的音訊流。
        
    - **輸出:** 辨識出的文字字串。
        
- **主流方法 (Mainstream Methods):**
    
    - **端到端深度學習模型:** 目前的主流方法使用單一的神經網路模型直接將音訊特徵對應到文字輸出，取代了傳統的聲學模型+發音詞典+語言模型的複雜管線。
        
    - **CTC (Connectionist Temporal Classification):** 一種常用於 ASR 的損失函數，它解決了音訊訊號和文字標籤之間長度不對齊的問題，無需預先進行音訊和文字的對位。
        
    - **基於注意力機制的 Seq2Seq 模型:** 類似於機器翻譯，使用編碼器將音訊訊號編碼成特徵，再由帶有注意力機制的解碼器生成文字。
        
    - **Transformer / Conformer:** Conformer 模型結合了 Transformer 和 CNN 的優點，能夠同時捕捉音訊的全局和局部依賴關係，是目前許多頂尖 ASR 系統的核心架構。Whisper by OpenAI 就是一個基於 Transformer 的強大模型。
        
- **常用資料集 (Datasets):**
    
    - **LibriSpeech:** 一個包含約 1000 小時英語有聲書的大型、公開資料集。
        
    - **Common Voice:** 由 Mozilla 發起的多語言語音資料集，由志願者錄製和驗證。
        
    - **AISHELL:** 中文語音領域常用的開源資料集。
        
    - **Switchboard:** 包含大量電話對話錄音的資料集，常用於學術研究。
        

---

### 7. 影像字幕 (Image Captioning)

- **目的 (Purpose):** 影像字幕的目標是讓電腦自動生成一段能夠準確、流暢地描述影像內容的文字。這項技術結合了電腦視覺（理解影像）和自然語言處理（生成文字），可用於輔助視障人士、影像檢索和社群媒體內容自動化。
    
- **輸入輸出 (Input/Output):**
    
    - **輸入:** 一張影像（例如 JPEG, PNG）。
        
    - **輸出:** 一句或一段描述影像內容的文字字串。
        
- **主流方法 (Mainstream Methods):**
    
    - **編碼器-解碼器架構 (Encoder-Decoder):** 這是最經典和主流的方法。
        
        - **編碼器 (Encoder):** 通常使用一個預訓練好的卷積神經網路（CNN），如 ResNet, VGG 或 Vision Transformer (ViT)，來提取影像的視覺特徵。
            
        - **解碼器 (Decoder):** 通常使用一個循環神經網路（RNN，如 LSTM）或 Transformer 解碼器，來接收影像特徵並逐詞生成描述性的句子。
            
    - **注意力機制:** 在解碼器生成每個詞時，注意力機制會讓模型專注於影像中最相關的區域。例如，在生成 "dog" 這個詞時，模型會更關注影像中狗所在的區域。
        
    - **大型多模態模型 (Large Multimodal Models, LMMs):** 近期的發展趨勢是使用大型模型（如 Flamingo, BLIP, GPT-4V）來處理此類任務。這些模型在大量的影像-文本對上進行預訓練，能夠生成更豐富、更準確且更具上下文理解能力的描述。
        
- **常用資料集 (Datasets):**
    
    - **MS COCO (Microsoft Common Objects in Context):** 最常用於影像字幕的標竿資料集，每張影像都有 5 句不同的人工標註描述。
        
    - **Flickr8k / Flickr30k:** 規模較小的資料集，同樣包含從 Flickr 收集的影像和對應的文字描述。
        
    - **Conceptual Captions:** 一個規模非常大的資料集，包含數百萬張從網頁上自動收集的影像和其對應的 alt-text 描述，常用於預訓練大型模型。