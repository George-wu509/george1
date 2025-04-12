
 Physics-Informed Neural Networks - PINNs

|                          |     |
| ------------------------ | --- |
| [[###FEM vs PINNs]]      |     |
| [[###PINN pytorch code]] |     |
|                          |     |


### FEM vs PINNs

具體說明如何使用傳統有限元素法 (Finite Element Method, FEM) 和物理訊息神經網路 (Physics-Informed Neural Networks, PINNs) 來求解一個二維矩形域內的 PDE 問題，例如，在給定初始條件 (Initial Condition, IC) 和邊界條件 (Boundary Condition, BC) 下，求解矩形內的流速場 `u(x, y, t)`。

假設我們要解的 PDE 是簡化的二維非定常不可壓縮流體流動問題（例如，用速度-渦度法或某種形式的 Navier-Stokes 方程），定義在一個矩形域 `Ω = [0, L] x [0, H]` 上，時間從 `t=0` 開始。

**問題定義:**

1. **PDE:** `∂u/∂t + (u · ∇)u = -∇p + ν ∇²u + f` (動量方程) 和 `∇ · u = 0` (連續性方程)。這裡 `u = (u_x, u_y)` 是速度向量，`p` 是壓力，`ν` 是運動黏度，`f` 是外力項。為簡化說明，有時會從更簡單的 PDE 開始，如擴散方程 `∂u/∂t = ν ∇²u`。
2. **計算域 (Domain):** `Ω = [0, L] x [0, H]` (一個長 L 寬 H 的矩形)。
3. **邊界條件 (BC):** 在矩形的四條邊 `∂Ω` 上給定。例如：
    - 底部 (`y=0`): 無滑移邊界 `u(x, 0, t) = (0, 0)`
    - 頂部 (`y=H`): 自由滑移或移動蓋 `u(x, H, t) = (U_top, 0)`
    - 左側 (`x=0`): 入口流速 `u(0, y, t) = u_in(y)`
    - 右側 (`x=L`): 出口條件（例如，壓力 `p=0` 或 `∂u/∂x = 0`）
4. **初始條件 (IC):** 在 `t=0` 時，域內所有點的初始流速 `u(x, y, 0) = u_0(x, y)`。

**1. 傳統有限元素法 (Traditional FEM) 求解流程**

FEM 的核心思想是將連續的求解域離散化，並在每個小單元上用簡單的函數（基函數）來近似解。

- **步驟 1: 網格生成 (Meshing)**
    
    - 將矩形域 `Ω` 劃分成許多小的、非重疊的子區域（單元），例如三角形或四邊形。
    - 這些單元的頂點稱為節點 (Nodes)。
    - 網格的質量（單元形狀、大小分佈）對計算精度和穩定性至關重要。在邊界層或預期梯度較大的區域，通常需要更密的網格。
    - _範例_: 對於矩形域，可以使用結構化網格（整齊排列的四邊形）或非結構化網格（三角形）。
- **步驟 2: 選擇基函數 (Basis Functions)**
    
    - 在每個單元內，用節點上的未知數值和一組預定義的插值函數（基函數或形函數，通常是低階多項式，如線性或二次）來近似真實解。
    - 全局解是這些局部近似的組合。例如，速度場 `u(x, y, t)` 被近似為 `u_h(x, y, t) = Σ N_i(x, y) * U_i(t)`，其中 `N_i` 是與節點 `i` 相關的基函數，`U_i(t)` 是節點 `i` 上的待求速度向量（自由度）。
    - _範例_: 對於流體問題，常用 P2-P1 元素（速度用二次基函數，壓力用線性基函數）來滿足 LBB 穩定性條件。
- **步驟 3: 導出弱形式 (Weak Formulation)**
    
    - 將原始的 PDE（強形式）轉換成積分形式（弱形式）。通常使用加權餘量法（如 Galerkin 法），將 PDE 乘以一個測試函數 `v`，然後在整個計算域上積分。
    - 通過分部積分（Green 公式），可以降低解的導數階數要求，並自然地引入 Neumann 邊界條件。
    - _範例_: 對於動量方程，乘以測試函數 `v`，在 `Ω` 上積分，並應用分部積分處理黏性項 `ν ∇²u` 和壓力項 `-∇p`。
- **步驟 4: 組裝系統方程 (Assembly)**
    
    - 將弱形式應用到每個單元上，計算單元勁度矩陣 (stiffness matrix) 和單元載荷向量 (load vector)。
    - 將所有單元的貢獻組裝成一個大型的全局代數方程組 `M * dU/dt + K(U) * U = F`（如果 PDE 是非線性的，如 Navier-Stokes，則 `K` 依賴於 `U`）或 `K * U = F`（對於穩態線性問題）。`M` 是質量矩陣，`K` 是勁度矩陣，`U` 是所有節點未知數的向量，`F` 是包含源項和邊界條件貢獻的載荷向量。
- **步驟 5: 時間離散化與求解 (Time Discretization & Solving)**
    
    - 對於非定常問題，需要對時間導數 `dU/dt` 進行離散化，例如使用有限差分法（如 Crank-Nicolson 法、向後歐拉法）。
    - 這會將微分代數方程轉化為在每個時間步長需要求解的一系列（可能非線性的）代數方程組。
    - 使用數值線性代數方法（如直接法 LU 分解，或迭代法 GMRES、共軛梯度法）求解該方程組，得到每個時間步的節點值 `U`。
    - _範例_: 在 `t=t_n` 時刻，已知 `U^n`，通過時間離散格式（如向後歐拉：`(U^{n+1} - U^n) / Δt` 替換 `dU/dt`），求解關於 `U^{n+1}` 的代數系統，得到下一時刻 `t_{n+1}` 的解。
- **步驟 6: 後處理 (Post-processing)**
    
    - 得到所有節點的解 `U` 後，可以通過基函數插值得到域內任意點的解。
    - 計算衍生量（如渦度、流線），並進行可視化。

**2. 物理訊息神經網路 (PINNs) 求解流程**

PINNs 利用神經網路的通用近似能力來直接擬合 PDE 的解，同時將物理定律（PDE、BC、IC）作為約束（罰函數）納入損失函數中進行訓練。

- **步驟 1: 定義神經網路 (Define Neural Network)**
    
    - 構建一個神經網路（通常是多層感知器 MLP），其輸入是時空座標 `(x, y, t)`，輸出是 PDE 的解，即速度場 `û(x, y, t; θ) = (û_x, û_y)`。`θ` 代表網路的所有可學習參數（權重和偏差）。
    - _範例_: 一個包含多個隱藏層（例如 4 層，每層 50 個神經元）和激活函數（如 tanh 或 swish）的 MLP。輸入層有 3 個神經元 (x, y, t)，輸出層有 2 個神經元 (u_x, u_y)。如果需要同時求解壓力，輸出層可以有 3 個神經元。
- **步驟 2: 定義損失函數 (Define Loss Function)**
    
    - 損失函數是 PINNs 的核心，它由多個部分組成，確保網路輸出滿足物理定律和邊界/初始條件。
        - **PDE 損失 (Loss_PDE):** 衡量網路輸出 `û` 在計算域內部 `Ω` 滿足 PDE 的程度。
            - 利用自動微分 (Automatic Differentiation, AD) 計算 `û` 對輸入 `x, y, t` 的導數（例如 `∂û/∂x`, `∂²û/∂y²`, `∂û/∂t` 等）。
            - 將這些導數代入 PDE 方程，得到殘差 `R(x, y, t; θ) = PDE_Operator(û) - source_term`。
            - 在域內隨機採樣大量配置點 (collocation points) `{(x_i, y_i, t_i)}_pde`，計算殘差的均方誤差 (Mean Squared Error, MSE)：`Loss_PDE = (1/N_pde) * Σ ||R(x_i, y_i, t_i; θ)||^2`。
            - _範例_: 對於動量方程，需要計算 `∂û/∂t`, `û`, `∇û`, `∇²û`，代入方程得到動量殘差。對於連續性方程，計算 `∇ · û`。
        - **邊界條件損失 (Loss_BC):** 衡量網路輸出 `û` 在邊界 `∂Ω` 上滿足 BC 的程度。
            - 在邊界上採樣配置點 `{(x_j, y_j, t_j)}_bc`。
            - 計算網路輸出與給定邊界值之間的 MSE：`Loss_BC = (1/N_bc) * Σ ||BC_Operator(û(x_j, y_j, t_j; θ)) - BC_Target||^2`。
            - _範例_: 在底部 `y=0` 採樣點，計算 `||û(x_j, 0, t_j; θ) - (0, 0)||^2`。在入口 `x=0` 採樣點，計算 `||û(0, y_j, t_j; θ) - u_in(y_j)||^2`。
        - **初始條件損失 (Loss_IC):** 衡量網路輸出 `û` 在初始時刻 `t=0` 滿足 IC 的程度。
            - 在域內 `Ω` 且 `t=0` 時採樣配置點 `{(x_k, y_k, 0)}_ic`。
            - 計算網路輸出與給定初始值之間的 MSE：`Loss_IC = (1/N_ic) * Σ ||û(x_k, y_k, 0; θ) - u_0(x_k, y_k)||^2`。
    - **總損失 (Total Loss):** `Loss = w_pde * Loss_PDE + w_bc * Loss_BC + w_ic * Loss_IC`。`w` 是各項損失的權重，需要仔細調整以平衡不同約束的重要性。
- **步驟 3: 訓練網路 (Train the Network)**
    
    - 選擇一個優化器（如 Adam, L-BFGS）。
    - 隨機生成（或預先定義）大量的 PDE 配置點、BC 配置點和 IC 配置點。
    - 通過最小化總損失函數 `Loss` 來迭代更新神經網路的參數 `θ`。這個過程通常需要數千甚至數萬次迭代。
    - 訓練過程中需要監控各項損失值的下降情況。
- **步驟 4: 評估與預測 (Evaluation & Prediction)**
    
    - 訓練完成後，神經網路 `û(x, y, t; θ*)` (其中 `θ*` 是最優參數) 就代表了 PDE 的近似解。
    - 可以在域內和邊界上的任意點 `(x, y, t)` 輸入網路，即可獲得該點的預測速度值，無需插值。解是連續可微的（只要激活函數光滑）。

**3. 雙方差異與比較**

|特性|傳統 FEM|PINNs|
|:--|:--|:--|
|**基本原理**|空間離散化，變分原理，求解大型代數方程組|通用函數逼近（神經網路），物理約束的損失函數最小化|
|**離散化**|需要顯式地對計算域進行網格劃分 (Meshing)|無需網格 (Mesh-free)，依賴於配置點的採樣|
|**解的表示**|分片多項式（在單元內插值），節點處離散|全局連續函數（由神經網路定義），處處可微（理論上）|
|**導數計算**|基於基函數計算，精度可能低於解本身，跨單元不連續|利用自動微分 (AD) 精確計算網路輸出的導數|
|**邊界/初始條件**|通常強加（直接賦值）或通過弱形式自然引入|作為損失函數的一部分 "軟" 強加，可能不會精確滿足|
|**實施複雜度**|需要網格生成器、有限元庫。概念成熟，庫豐富|需要深度學習框架 (TF/PyTorch)、AD。相對較新，正在發展|
|**處理複雜幾何**|網格生成可能非常困難且耗時|原則上更容易處理，只需在幾何域內/邊界採樣點|
|**計算成本**|主要在求解大型稀疏線性/非線性系統。內存需求大|主要在神經網路訓練（反向傳播）。計算量大，GPU 加速明顯|
|**精度與收斂性**|有成熟的收斂理論（h-p refinement），可控性好|理論不完善，精度依賴網路結構、訓練、超參數，不易控制|
|**求解非線性 PDE**|需要迭代求解器（如 Newton 法）|直接納入損失函數優化，但訓練可能更困難|
|**逆問題/數據融合**|通常需要額外框架，迭代求解|天然適合，可直接將觀測數據加入損失函數|
|**維度災難**|對高維問題（>3D）網格數量指數增長，非常困難|潛力上更能應對高維問題（但訓練難度也增加）|
|**程式碼實現**|需要處理網格數據結構、矩陣組裝、求解器接口等|主要涉及神經網路定義、損失函數構建和訓練循環|

匯出到試算表

**總結:**

- **FEM** 是一種非常成熟、理論完善、精度可控的方法，特別適用於有明確邊界和幾何的低維（2D/3D）問題。其主要挑戰在於網格生成和求解大型代數系統。對於工業界的標準問題，有大量可靠的商業和開源軟體可用。
- **PINNs** 是一種較新的、基於數據驅動和物理約束的方法，具有無網格、自動微分計算導數、易於處理複雜幾何（原則上）和天然融合數據/求解逆問題的優勢。其主要挑戰在於訓練的穩定性、超參數調整、精度保證和缺乏完善的收斂理論。它在高維問題和需要結合物理與數據的問題上顯示出潛力。

在求解您提到的二維矩形內流速場問題時：

- 如果精度要求高、問題相對標準、且有成熟的 FEM 軟體可用，**FEM** 通常是更可靠、更直接的選擇。
- 如果幾何形狀未來可能變得非常複雜、或者您希望結合一些稀疏的實驗測量數據來輔助求解、或者對探索新方法感興趣，**PINNs** 提供了一個有前景的替代方案，但可能需要更多的調試和驗證工作。


===============================================

### PINN pytorch code

好的，這是一個使用 PyTorch 實現 PINN 的完整程式碼範例骨架。這個範例將包含您指定的 MLP 架構（輸入 3 維，4 個隱藏層各 50 神經元，輸出 2 或 3 維），並展示求解一個假設的二維非定常 PDE 的流程。

**請注意：**

1. **PDE 的選擇：** 為了演示，這裡並未完整實現複雜的 Navier-Stokes 方程，而是提供了一個通用框架。你需要將 `pde_residual` 函數中的 PDE 方程替換為你實際想要解的具體方程（例如 Navier-Stokes 的動量方程和連續性方程，或是簡化的對流擴散方程等）。
2. **邊界/初始條件：** 同樣地，`Loss_BC` 和 `Loss_IC` 的計算需要根據你問題的具體邊界和初始條件來編寫目標值。
3. **超參數：** 學習率、迭代次數、各項 Loss 的權重 (`w_pde`, `w_bc`, `w_ic`)、配置點數量等都需要根據具體問題仔細調整。
4. **壓力項：** 程式碼中包含了輸出 3 個變數（包括壓力 `p`）的選項。如果選擇包含壓力，你需要：
    - 修改 `PINN` 網路的 `output_features` 為 3。
    - 在 `pde_residual` 中加入與壓力相關的項（例如 `-∇p`）以及連續性方程 `∇ · u = 0` 的殘差。
    - 可能需要為壓力設定邊界條件，並在 `Loss_BC` 中加入相應的項。


```python
import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt

# --- 1. 設定與裝置 ---
# 檢查是否有可用的 CUDA 裝置 (GPU)，否則使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 亂數種子確保結果可重現性 (可選)
# torch.manual_seed(1234)
# np.random.seed(1234)

# --- 2. 神經網路模型定義 (MLP) ---
class PINN(nn.Module):
    def __init__(self, input_features=3, hidden_features=50, num_hidden_layers=4, output_features=2, activation=nn.Tanh):
        """
        初始化 PINN 網路
        Args:
            input_features (int): 輸入特徵數 (x, y, t) -> 3
            hidden_features (int): 每個隱藏層的神經元數量
            num_hidden_layers (int): 隱藏層的數量
            output_features (int): 輸出特徵數 (u_x, u_y) -> 2 或 (u_x, u_y, p) -> 3
            activation (torch.nn.Module): 激活函數 (例如 nn.Tanh, nn.SiLU for Swish)
        """
        super().__init__()

        layers = []
        # 輸入層
        layers.append(nn.Linear(input_features, hidden_features))
        layers.append(activation())
        # 隱藏層
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(activation())
        # 輸出層
        layers.append(nn.Linear(hidden_features, output_features))

        # 使用 nn.Sequential 組合所有層
        self.network = nn.Sequential(*layers)

        # 初始化權重 (可選，有時有助於訓練)
        # self.init_weights()

    def forward(self, x, y, t):
        """
        前向傳播
        Args:
            x (torch.Tensor): x 座標, shape [N, 1]
            y (torch.Tensor): y 座標, shape [N, 1]
            t (torch.Tensor): 時間 t, shape [N, 1]
        Returns:
            torch.Tensor: 網路輸出 (u_x, u_y, [p]), shape [N, output_features]
        """
        # 將輸入合併成一個張量
        inputs = torch.cat([x, y, t], dim=1)
        outputs = self.network(inputs)
        return outputs

    # 可選的權重初始化方法
    # def init_weights(self):
    #     for m in self.network.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)

# --- 3. PDE 殘差計算 ---
# !!! 重要：你需要在此處定義你的 PDE 方程 !!!
def pde_residual(network, x, y, t, nu=0.01, output_dim=2):
    """
    計算 PDE 殘差
    Args:
        network (PINN): PINN 網路模型
        x (torch.Tensor): x 座標 (需要梯度)
        y (torch.Tensor): y 座標 (需要梯度)
        t (torch.Tensor): 時間 t (需要梯度)
        nu (float): 物理參數 (例如: 黏度)
        output_dim (int): 網路輸出維度 (2 for u_x, u_y; 3 for u_x, u_y, p)
    Returns:
        Tuple[torch.Tensor, ...]: PDE 方程的殘差張量
                                   (例如: momentum_x_residual, momentum_y_residual, [continuity_residual])
    """
    # 確保輸入需要梯度
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    # 獲取網路輸出
    outputs = network(x, y, t)
    u_x = outputs[:, 0:1]
    u_y = outputs[:, 1:2]
    if output_dim == 3:
        p = outputs[:, 2:3] # 如果求解壓力

    # --- 使用 torch.autograd.grad 計算導數 ---
    # 一階導數
    grads = torch.autograd.grad(outputs.sum(), [x, y, t], create_graph=True)[0] # Summing over batch and outputs for grad
    
    # Need to compute gradients for each output component separately for higher orders
    u_x_grads = torch.autograd.grad(u_x.sum(), [x, y, t], create_graph=True)
    u_x_x = u_x_grads[0]
    u_x_y = u_x_grads[1]
    u_x_t = u_x_grads[2]

    u_y_grads = torch.autograd.grad(u_y.sum(), [x, y, t], create_graph=True)
    u_y_x = u_y_grads[0]
    u_y_y = u_y_grads[1]
    u_y_t = u_y_grads[2]

    if output_dim == 3:
        p_grads = torch.autograd.grad(p.sum(), [x, y], create_graph=True) # Pressure gradient only spatial
        p_x = p_grads[0]
        p_y = p_grads[1]

    # 二階導數 (Laplacian)
    u_x_xx = torch.autograd.grad(u_x_x.sum(), x, create_graph=True)[0]
    u_x_yy = torch.autograd.grad(u_x_y.sum(), y, create_graph=True)[0]
    laplacian_u_x = u_x_xx + u_x_yy

    u_y_xx = torch.autograd.grad(u_y_x.sum(), x, create_graph=True)[0]
    u_y_yy = torch.autograd.grad(u_y_y.sum(), y, create_graph=True)[0]
    laplacian_u_y = u_y_xx + u_y_yy

    # --- 代入你的 PDE 方程 ---
    # 範例: 2D 黏性 Burgers' 方程 (忽略壓力，無連續性方程)
    if output_dim == 2:
      # 動量方程 x 方向殘差: u_t + u*u_x + v*u_y - nu*(u_xx + u_yy) = 0
      momentum_x_res = u_x_t + u_x * u_x_x + u_y * u_x_y - nu * laplacian_u_x
      # 動量方程 y 方向殘差: v_t + u*v_x + v*v_y - nu*(v_xx + v_yy) = 0
      momentum_y_res = u_y_t + u_x * u_y_x + u_y * u_y_y - nu * laplacian_u_y
      return momentum_x_res, momentum_y_res

    # 範例: 包含壓力的 Navier-Stokes (簡化形式 - 缺少連續性方程實現)
    elif output_dim == 3:
      # 動量方程 x 方向殘差: u_t + u*u_x + v*u_y + p_x - nu*(u_xx + u_yy) = 0
      momentum_x_res = u_x_t + u_x * u_x_x + u_y * u_x_y + p_x - nu * laplacian_u_x
      # 動量方程 y 方向殘差: v_t + u*v_x + v*v_y + p_y - nu*(v_xx + v_yy) = 0
      momentum_y_res = u_y_t + u_x * u_y_x + u_y * u_y_y + p_y - nu * laplacian_u_y
      # 連續性方程殘差: u_x + v_y = 0  <-- 你需要實現這個！
      continuity_res = u_x_x + u_y_y # 這裡只是佔位符，需要正確計算 div(u)
      print("Warning: Continuity residual not fully implemented in example.")
      return momentum_x_res, momentum_y_res, continuity_res # 返回三個殘差


# --- 4. 資料採樣 ---
# 定義計算域和時間範圍
x_domain = [0.0, 1.0]
y_domain = [0.0, 1.0]
t_domain = [0.0, 1.0]

def sample_domain(N, device):
    """在計算域內部隨機採樣"""
    x = torch.rand(N, 1, device=device) * (x_domain[1] - x_domain[0]) + x_domain[0]
    y = torch.rand(N, 1, device=device) * (y_domain[1] - y_domain[0]) + y_domain[0]
    t = torch.rand(N, 1, device=device) * (t_domain[1] - t_domain[0]) + t_domain[0]
    return x, y, t

def sample_boundary(N, device):
    """在邊界上採樣"""
    N_each = N // 4 # 平均分配到四條邊
    # x=0 boundary
    x0 = torch.zeros(N_each, 1, device=device) + x_domain[0]
    y0 = torch.rand(N_each, 1, device=device) * (y_domain[1] - y_domain[0]) + y_domain[0]
    t0 = torch.rand(N_each, 1, device=device) * (t_domain[1] - t_domain[0]) + t_domain[0]
    # x=L boundary
    xL = torch.zeros(N_each, 1, device=device) + x_domain[1]
    yL = torch.rand(N_each, 1, device=device) * (y_domain[1] - y_domain[0]) + y_domain[0]
    tL = torch.rand(N_each, 1, device=device) * (t_domain[1] - t_domain[0]) + t_domain[0]
    # y=0 boundary
    x_b0 = torch.rand(N_each, 1, device=device) * (x_domain[1] - x_domain[0]) + x_domain[0]
    y_b0 = torch.zeros(N_each, 1, device=device) + y_domain[0]
    t_b0 = torch.rand(N_each, 1, device=device) * (t_domain[1] - t_domain[0]) + t_domain[0]
    # y=H boundary
    x_bH = torch.rand(N_each, 1, device=device) * (x_domain[1] - x_domain[0]) + x_domain[0]
    y_bH = torch.zeros(N_each, 1, device=device) + y_domain[1]
    t_bH = torch.rand(N_each, 1, device=device) * (t_domain[1] - t_domain[0]) + t_domain[0]

    # 合併所有邊界點
    bc_x = torch.cat([x0, xL, x_b0, x_bH], dim=0)
    bc_y = torch.cat([y0, yL, y_b0, y_bH], dim=0)
    bc_t = torch.cat([t0, tL, t_b0, t_bH], dim=0)

    # --- !!! 定義你的邊界條件目標值 !!! ---
    # 範例: 所有邊界速度為 0 (Dirichlet BC)
    target_ux = torch.zeros_like(bc_x)
    target_uy = torch.zeros_like(bc_y)
    # 如果有壓力，可能需要壓力的 BC，例如出口 p=0
    # target_p = torch.zeros_like(bc_x)

    bc_targets = torch.cat([target_ux, target_uy], dim=1) # Shape: [N_bc_total, 2]
    # if output_dim == 3:
    #     bc_targets = torch.cat([target_ux, target_uy, target_p], dim=1) # Shape: [N_bc_total, 3]

    return bc_x, bc_y, bc_t, bc_targets

def sample_initial(N, device):
    """在初始時刻 t=0 採樣"""
    x = torch.rand(N, 1, device=device) * (x_domain[1] - x_domain[0]) + x_domain[0]
    y = torch.rand(N, 1, device=device) * (y_domain[1] - y_domain[0]) + y_domain[0]
    t = torch.zeros_like(x) + t_domain[0] # t=0

    # --- !!! 定義你的初始條件目標值 !!! ---
    # 範例: u_x = sin(pi*x)*sin(pi*y), u_y = 0
    target_ux = torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    target_uy = torch.zeros_like(y)
    # 如果有壓力，定義初始壓力場
    # target_p = torch.zeros_like(x) # e.g., p=0 initially

    ic_targets = torch.cat([target_ux, target_uy], dim=1) # Shape: [N, 2]
    # if output_dim == 3:
    #     ic_targets = torch.cat([target_ux, target_uy, target_p], dim=1) # Shape: [N, 3]

    return x, y, t, ic_targets


# --- 5. 訓練設定 ---
# 網路參數
INPUT_FEATURES = 3
HIDDEN_FEATURES = 50
NUM_HIDDEN_LAYERS = 4
OUTPUT_FEATURES = 2 # 改成 3 如果需要求解壓力
ACTIVATION = nn.Tanh # 或 nn.SiLU

# 物理參數 (範例)
NU = 0.01 / torch.pi # 根據 Burgers' equation 常見設定

# 訓練超參數
LEARNING_RATE = 1e-3
EPOCHS = 10000 # 迭代次數，可能需要更多
N_PDE = 20000  # PDE 配置點數量
N_BC = 2500    # 邊界條件點數量
N_IC = 2500    # 初始條件點數量

# 損失函數權重 (需要仔細調整)
W_PDE = 1.0
W_BC = 10.0 # BC 通常需要較高權重
W_IC = 10.0 # IC 也相對重要

# --- 6. 初始化模型和優化器 ---
pinn_net = PINN(
    input_features=INPUT_FEATURES,
    hidden_features=HIDDEN_FEATURES,
    num_hidden_layers=NUM_HIDDEN_LAYERS,
    output_features=OUTPUT_FEATURES,
    activation=ACTIVATION
).to(device)

optimizer = torch.optim.Adam(pinn_net.parameters(), lr=LEARNING_RATE)
# 可選: 學習率調度器
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

# 均方誤差損失函數
criterion = nn.MSELoss()

# --- 7. 訓練迴圈 ---
loss_history = []
start_time = time.time()

for epoch in range(EPOCHS):
    pinn_net.train() # 設定為訓練模式

    # 1. 採樣配置點
    x_pde, y_pde, t_pde = sample_domain(N_PDE, device)
    x_bc, y_bc, t_bc, target_bc = sample_boundary(N_BC, device)
    x_ic, y_ic, t_ic, target_ic = sample_initial(N_IC, device)

    # 2. 計算 PDE 損失
    residuals = pde_residual(pinn_net, x_pde, y_pde, t_pde, nu=NU, output_dim=OUTPUT_FEATURES)
    # 對所有 PDE 殘差計算 MSE 損失
    loss_pde = 0
    for res in residuals:
        loss_pde += criterion(res, torch.zeros_like(res))
    loss_pde /= len(residuals) # 取平均 (如果有多個 PDE)

    # 3. 計算 BC 損失
    pred_bc = pinn_net(x_bc, y_bc, t_bc)
    loss_bc = criterion(pred_bc, target_bc)

    # 4. 計算 IC 損失
    pred_ic = pinn_net(x_ic, y_ic, t_ic)
    loss_ic = criterion(pred_ic, target_ic)

    # 5. 計算總損失
    loss = W_PDE * loss_pde + W_BC * loss_bc + W_IC * loss_ic

    # 6. 反向傳播與優化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # if scheduler:
    #     scheduler.step()

    # 7. 記錄與打印損失
    loss_history.append(loss.item())
    if (epoch + 1) % 100 == 0:
        elapsed_time = time.time() - start_time
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}, '
              f'Loss PDE: {loss_pde.item():.6f}, Loss BC: {loss_bc.item():.6f}, Loss IC: {loss_ic.item():.6f}, '
              f'Time: {elapsed_time:.2f}s')
        start_time = time.time() # Reset timer for next block

# --- 8. 訓練後處理 (可選) ---
# 保存模型
# torch.save(pinn_net.state_dict(), 'pinn_model.pth')

# 繪製損失歷史
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.yscale('log')
plt.title('Training Loss History')
plt.xlabel('Epoch')
plt.ylabel('Log Loss')
plt.grid(True)
plt.show()

# 評估與可視化 (範例: 在 t=0.5 時刻的可視化)
pinn_net.eval() # 設定為評估模式
with torch.no_grad():
    t_eval = 0.5
    n_points_eval = 100
    x_eval = torch.linspace(x_domain[0], x_domain[1], n_points_eval, device=device)
    y_eval = torch.linspace(y_domain[0], y_domain[1], n_points_eval, device=device)
    x_grid, y_grid = torch.meshgrid(x_eval, y_eval, indexing='ij')
    t_grid = torch.ones_like(x_grid) * t_eval

    x_flat = x_grid.flatten().unsqueeze(1)
    y_flat = y_grid.flatten().unsqueeze(1)
    t_flat = t_grid.flatten().unsqueeze(1)

    predictions_flat = pinn_net(x_flat, y_flat, t_flat)
    u_x_pred = predictions_flat[:, 0].reshape(n_points_eval, n_points_eval).cpu().numpy()
    u_y_pred = predictions_flat[:, 1].reshape(n_points_eval, n_points_eval).cpu().numpy()
    # if OUTPUT_FEATURES == 3:
    #     p_pred = predictions_flat[:, 2].reshape(n_points_eval, n_points_eval).cpu().numpy()

    plt.figure(figsize=(18, 5))
    plt.subplot(1, 2, 1)
    plt.contourf(x_grid.cpu().numpy(), y_grid.cpu().numpy(), u_x_pred, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title(f'Predicted u_x at t={t_eval}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    plt.contourf(x_grid.cpu().numpy(), y_grid.cpu().numpy(), u_y_pred, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title(f'Predicted u_y at t={t_eval}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')

    # if OUTPUT_FEATURES == 3:
    #   plt.figure(figsize=(6, 5))
    #   plt.contourf(x_grid.cpu().numpy(), y_grid.cpu().numpy(), p_pred, levels=50, cmap='viridis')
    #   plt.colorbar()
    #   plt.title(f'Predicted Pressure p at t={t_eval}')
    #   plt.xlabel('x')
    #   plt.ylabel('y')
    #   plt.axis('equal')

    plt.tight_layout()
    plt.show()

    # 也可以繪製速度向量場
    # skip = 5 # 每隔幾個點繪製一個向量
    # plt.figure(figsize=(8, 8))
    # plt.quiver(x_grid.cpu().numpy()[::skip, ::skip], y_grid.cpu().numpy()[::skip, ::skip],
    #            u_x_pred[::skip, ::skip], u_y_pred[::skip, ::skip])
    # plt.title(f'Predicted Velocity Field at t={t_eval}')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.axis('equal')
    # plt.show()

```

**程式碼說明:**

1. **設定與裝置:** 導入必要的庫，設定計算裝置 (GPU 或 CPU)。
2. **神經網路模型定義 (`PINN` class):**
    - 使用 `nn.Module` 構建標準的 PyTorch 模型。
    - `__init__` 中根據輸入參數動態創建網路層，使用 `nn.Sequential` 方便地組合起來。
    - `forward` 方法定義了數據如何通過網路（輸入 `x, y, t`，輸出 `u_x, u_y, [p]`）。
3. **PDE 殘差計算 (`pde_residual` function):**
    - **核心部分**，計算物理損失。
    - 輸入網路模型和需要計算梯度的座標點 `x, y, t`。
    - **關鍵:** 使用 `torch.autograd.grad` 來自動計算網路輸出關於輸入的各階導數。注意 `create_graph=True` 對於計算高階導數和確保梯度能流回網路參數是必需的。
    - **替換區域:** 將計算得到的導數代入你的具體 PDE 方程，得到每個方程的殘差。
4. **資料採樣:**
    - `sample_domain`: 在計算域內部隨機生成點用於計算 PDE 損失。
    - `sample_boundary`: 在計算域的邊界上生成點，並 **定義相應的邊界目標值**。
    - `sample_initial`: 在初始時刻 `t=0` 生成點，並 **定義相應的初始條件目標值**。
5. **訓練設定:** 定義網路結構參數、物理參數、學習率、迭代次數、採樣點數量和損失權重等超參數。
6. **初始化模型和優化器:** 創建 PINN 網路實例，選擇優化器（常用 Adam），定義損失函數（MSE）。
7. **訓練迴圈:**
    - 迭代指定次數 (epochs)。
    - 每次迭代：採樣數據點 -> 計算 PDE、BC、IC 損失 -> 加權求和得到總損失 -> 清空梯度 -> 反向傳播計算梯度 -> 更新網路權重。
    - 定期打印損失值，監控訓練進程。
8. **訓練後處理:**
    - 可選：保存訓練好的模型。
    - 繪製損失下降曲線，檢查訓練是否收斂。
    - 在網格點上評估訓練好的模型，並將結果（例如某個時刻的流場）可視化。

這個框架提供了 PINN 的基本結構。你需要根據你的具體 PDE 問題，填充 `pde_residual` 函數中的方程，以及 `sample_boundary` 和 `sample_initial` 中的目標值。訓練過程可能需要大量的超參數調整才能獲得滿意的結果。