
### 如何準備「Hands-on Skills」面試

準備的重點在於將你的經驗「**故事化**」和「**具體化**」。你必須能夠證明你_寫過_程式碼，而不只是_讀過_論文。

**1. 準備你的「項目故事」(Project Stories)**

這是最重要的一步。選擇 2-3 個你最熟悉、最有挑戰性的 AI/CV 專案。你必須能像說故事一樣，詳細解說這個專案的「PyTorch 動手實作」細節。

- **動機 (Why)：** 為什麼選擇 PyTorch？為什麼選擇這個特定的模型架構 (例如 U-Net, ViT, YOLO)？
    
- **實作 (How)：**
    
    - **數據 (Data)：** 你如何實作 `Dataset` 和 `DataLoader`？數據量多大？你做了哪些 data augmentation (資料增強)？
        
    - **模型 (Model)：** 你是從頭寫 (`nn.Module`)，還是 fine-tune (微調)？如果是 fine-tune，你凍結了哪些層？為什麼？
        
    - **訓練 (Training)：** 你的訓練循環 (training loop) 是怎麼寫的？你用了什麼損失函數 (Loss Function)？為什麼？你用了什麼優化器 (Optimizer)？
        
- **挑戰 (Challenge)：** 這是 "hands-on" 的精髓。
    
    - _「我遇到了 overfitting，所以我加入了 Dropout 和更強的 Augmentation。」_
        
    - _「訓練一開始 Loss 變成 NaN，我發現是 learning rate 太高導致梯度爆炸，所以我加入了 gradient clipping。」_
        
    - _「`DataLoader` 速度很慢，我把 `num_workers` 調高並啟用了 `pin_memory`。」_
        
- **結果 (Result)：** 你的模型表現如何 (e.g., Accuracy, mAP, F1-score)？你如何評估它？
    

**2. 複習 PyTorch 的「關鍵區別」**

面試官會用一些具體問題來測試你是否真的懂 PyTorch。

- `model.train()` vs `model.eval()`：一定要知道它們的區別（例如對 `BatchNorm` 和 `Dropout` 層的影響）。
    
- `.detach()` vs `with torch.no_grad()`：兩者在什麼情境下使用？
    
- `nn.Sequential` vs `nn.ModuleList` vs `nn.ModuleDict`：它們的用途有何不同？
    
- `optimizer.zero_grad()`：為什麼這一步是必要的？如果忘了做會怎樣？
    
- `loss.backward()` vs `optimizer.step()`：它們各自做了什麼？
    

**3. 準備 Live Coding (現場程式設計)**

"Hands-on" 的終極考驗就是 live coding。你不需要能背出複雜的 Transformer 架構，但你應該要能在白板或線上編輯器中流暢地寫出：

- 一個自定義的 PyTorch `Dataset` 類別。
    
- 一個基本的 CNN 模型架構 (`nn.Module`)。
    
- 一個完整的 PyTorch 訓練循環 (training loop)（包含 data loading, forward pass, loss calculation, backward pass, optimizer step）。
    
- 一個 fine-tuning (微調) 預訓練模型的流程 (例如 ResNet-50)。
    

**4. 深入理解「你用過的」CV 模型**

如果你說你用過 ResNet，你必須能解釋「Residual Connection (殘差連接)」是什麼，以及它解決了什麼問題（梯度消失、深度網路訓練）。如果你用過 U-Net，你必須能解釋「Skip Connection (跳躍連接)」的作用（融合高層語義和低層紋理特徵）。

---

### 40 個 PyTorch 與 AI/CV 模型面試問題

以下問題分為六類，從 PyTorch 基礎到 CV 應用的實戰問題。

#### A. PyTorch 核心概念 (10 題)

1. PyTorch 的 `Tensor` 和 NumPy 的 `array` 有什麼主要區別？
    
2. 請解釋 `autograd` (自動微分) 的工作原理。`requires_grad=True` 標記的作用是什麼？
    
3. `model.train()` 和 `model.eval()` 的區別是什麼？為什麼在 validation (驗證) 時必須呼叫 `model.eval()`？
    
4. 為什麼我們需要在 `optimizer.step()` 之前呼叫 `optimizer.zero_grad()`？
    
5. `loss.backward()` 這一行程式碼具體做了什麼？
    
6. `with torch.no_grad()` 和 `.detach()` 有什麼區別？你分別會在什麼情境下使用它們？
    
7. 什麼是動態計算圖 (Dynamic Computation Graph)？它和 TensorFlow 1.x 的靜態圖有什麼優勢？
    
8. `nn.Module` 是什麼？在 `__init__` 和 `forward` 方法中各自應該放什麼？
    
9. `nn.Sequential` 和 `nn.ModuleList` 有什麼不同？
    
10. 你如何儲存和載入一個 PyTorch 模型？儲存 "checkpoint" (包含 optimizer 狀態) 和儲存 "inference-only model" (僅模型權重) 的程式碼有何不同？
    

#### B. 數據處理 (Dataset / DataLoader) (5 題)

11. 你如何為你的專案建立一個自定義的 `Dataset` 類別？需要實作哪兩個 `__magic__` 方法？
    
12. `Dataset` 和 `DataLoader` 的職責有何不同？
    
13. `DataLoader` 中的 `num_workers` 參數是什麼作用？設定太高或太低會有什麼問題？
    
14. 什麼是 `pin_memory=True`？它在什麼情況下有幫助？
    
15. 在 CV 任務中，你通常會使用 `torchvision.transforms` 來做哪些 data augmentation？你如何決定使用哪些 augmentation？
    

#### C. 模型建構與訓練 (8 題)

16. 請在白板上（或口頭上）描述一個完整的 PyTorch 訓練循環 (training loop)。
    
17. 什麼是 Batch Normalization (BN)？它在訓練和推論 (inference) 時的行為有何不同？
    
18. 什麼是 Dropout？它在訓練和推論 (inference) 時的行為有何不同？
    
19. 請解釋 Adam 和 SGD with Momentum 這兩種優化器的主要區別。
    
20. 什麼是 Learning Rate Scheduling (學習率調度)？你用過哪些 (例如 `StepLR`, `ReduceLROnPlateau`)？
    
21. 你如何在 PyTorch 中實作 Transfer Learning (遷移學習)？如何「凍結」(freeze) 預訓練模型的某些層？
    
22. 什麼是 1x1 卷積 (convolution)？它在 ResNet 或 Inception 網路中有什麼作用？
    
23. 什麼是 Residual Connection (殘差連接)？它解決了什麼問題？
    

#### D. 電腦視覺 (CV) 模型 (7 題)

24. (針對你的專案) 你為什麼選擇 [你用過的模型，例如 YOLOv5] 而不是 [其他模型，例如 Faster R-CNN]？
    
25. (影像分類) ResNet 和 VGG 相比，主要的架構創新是什麼？
    
26. (物件偵測) 請高層次地解釋 One-Stage (如 YOLO, SSD) 和 Two-Stage (如 Faster R-CNN) 偵測器的區別。
    
27. (影像分割) U-Net 架構中的 "Skip Connections" (跳躍連接) 有什麼關鍵作用？
    
28. (Transformer) Vision Transformer (ViT) 是如何將一張 2D 圖片轉換為 1D 序列 (sequence of patches) 來處理的？
    
29. (GANs) 生成器 (Generator) 和判別器 (Discriminator) 的損失函數 (loss function) 是如何設計的？
    
30. 什麼是 mAP (mean Average Precision)？你如何解釋這個指標？
    

#### E. 調試 (Debugging) 與優化 (8 題)

31. **情境題：** 你的模型訓練一開始，loss 就變成了 `NaN`。你會從哪些地方開始檢查？
    
32. **情境題：** 你的訓練 (training) loss 不斷下降，但驗證 (validation) loss 卻停滯或上升。這代表什麼？你會如何解決 (overfitting)？
    
33. **情境題：** 你的訓練和驗證 loss 都很高，且不再下降。這代表什麼？你會如何解決 (underfitting)？
    
34. 什麼是梯度消失 (Vanishing Gradients) 和梯度爆炸 (Exploding Gradients)？
    
35. 什麼是 Gradient Clipping (梯度裁剪)？你通常在什麼情況下使用它？
    
36. 你如何監控你的模型訓練過程？(例如使用 TensorBoard 或 Weights & Biases)
    
37. **情境題：** 你的 GPU 利用率很低 (e.g., 只有 30%)，但 CPU 卻很忙。你認為瓶頸可能在哪裡？
    
38. 你知道什麼是 Mixed Precision Training (混合精度訓練) 嗎？(例如 `torch.cuda.amp`) 它有什麼好處？
    

#### F. 部署與進階 (2 題)

39. 當你需要將 PyTorch 模型部署到生產環境 (production) 時，你會考慮哪些步驟？(例如 ONNX, TorchScript, Quantization)
    
40. 你有在多個 GPU 上訓練模型的經驗嗎？你能否簡單解釋 `DataParallel` (DP) 和 `DistributedDataParallel` (DDP) 的區別？