
感謝您註冊 AWS Support (Basic)。您現在可以存取 AWS Support (Basic)。

若您使用軟體開發套件、命令列介面 (CLI) 或 API 以程式方式與 Amazon Web Services 互動，您必須提供存取金鑰，以驗證您的身分以及您是否擁有存取您所請求資源的許可。若要管理您帳戶的存取金鑰，請前往 [https://console.aws.amazon.com/iam/home?#security_credential](https://p6li1chk.r.us-east-1.awstrack.me/L0/https:%2F%2Fconsole.aws.amazon.com%2Fiam%2Fhome%3F%23security_credential/1/0100019591e6d39b-9b37e50d-8a8e-4236-975a-d3aba4be9ac7-000000/knR4QJGtRjD5krlTxi-Zf7ISOAg=417)。


Example:
文件、範例程式碼、文章、教學等可在 Amazon Web Services 資源中心的 [https://aws.amazon.com/resources/](https://p6li1chk.r.us-east-1.awstrack.me/L0/https:%2F%2Faws.amazon.com%2Fresources%2F/1/0100019591e6d39b-9b37e50d-8a8e-4236-975a-d3aba4be9ac7-000000/AoDdKuPKRf3dwJcN4OwjZEFFVYQ=417) 中找到。

您可透過免費試用方案來探索超過 100 種的產品，並享有一律免費的優惠，使用 [免費方案](https://p6li1chk.r.us-east-1.awstrack.me/L0/https:%2F%2Faws.amazon.com%2Ffree%2F/1/0100019591e73ae1-9f2bbe5b-cd7c-47d4-b0c0-5bb602b46b51-000000/4r_WcsgJObTvf-ew1NPdAP_FKLw=417) 在 Amazon Web Services 上開始建置。某些優惠僅在您的 Amazon Web Services 註冊日期後的 12 個月內提供給新客戶。

使用存取金鑰，從 AWS CLI、適用於 AWS Tools for PowerShell、AWS 軟體開發套件或直接的 AWS API 呼叫，將程式化呼叫傳送至 AWS。您一次最多可以有兩種存取金鑰 (作用中或非作用中)。
Use access keys to send programmatic calls to AWS from the AWS CLI, AWS Tools for PowerShell, AWS SDKs, or direct AWS API calls. You can have a maximum of two access keys (active or inactive) at a time. [Learn more](https://docs.aws.amazon.com/console/iam/self-accesskeys)


|                                          |     |
| ---------------------------------------- | --- |
| [[###AWS Identity Center 建立使用者]]         |     |
| [[###AWS 的 Security Credentials (安全憑證)]] |     |
| [[###如何在 AWS 上訓練一個簡單的 AI 模型]]            |     |
|                                          |     |


### AWS Identity Center 建立使用者
## **1️⃣ Specify a user in Identity Center - Recommended（使用 AWS Identity Center 建立使用者，推薦選項）**

- **適用對象**：這是 **AWS 官方建議** 的方式，適合 **個人開發者** 或 **企業管理 AWS 訪問權限**。
- **管理方式**：
    - 透過 **AWS IAM Identity Center（以前叫 AWS SSO）** 集中管理使用者存取權限。
    - **適合多個 AWS 帳戶和應用程式**（例如允許某個人存取特定 AWS 服務，卻不能存取其他服務）。
- **存取方式**：
    - **允許 Web 登入 AWS 管理主控台**（適合開發者和管理者）。
    - 透過 Identity Center，**不需要手動設定 IAM 使用者與權限**，系統會自動管理。
- **何時選擇？**
    - **你想讓某人透過 AWS Management Console 操作 AWS 服務**。
    - **你希望之後輕鬆管理權限（例如公司 IT 團隊管理員）**。
    - **適合長期使用**，並支援 SSO（單一登入）。

---

## **2️⃣ I want to create an IAM user（建立 IAM 使用者）**

- **適用對象**：適合 **需要程式化存取（Programmatic Access）** 的使用者，而非主要用來登入 AWS Console。
- **管理方式**：
    - 需要手動設定 IAM Policies（權限）。
    - 適用於特定場景，例如 AWS API、CLI（命令列工具）操作。
- **存取方式**：
    - 可設定 **Access Key & Secret Key**，允許透過 AWS CLI、SDK、API 存取 AWS 服務。
    - 也可設定 **AWS Management Console 存取**，但這不是 AWS 推薦的方法，因為使用 Identity Center 會更容易管理。
- **何時選擇？**
    - 你 **需要讓程式或應用程式存取 AWS**（例如 EC2 內部的應用程式）。
    - 你 **需要 API 或 CLI 存取 AWS**。
    - 你 **需要備用帳號作為緊急存取方案（Emergency Access）**。
    - **不適合一般日常 AWS Console 存取**，AWS 建議使用 **Identity Center** 來管理 Web 存取。

---

### **🟢 你應該選哪個？**

- **如果你只是想要創建一個 Web 登入 AWS Console 的使用者 → 選擇** ✅ **「Specify a user in Identity Center」**（推薦）。
- **如果你是為程式化存取（API、CLI、SDK）建立帳號 → 選擇** ✅ **「I want to create an IAM user」**。

**💡 簡單來說：** ✔ **要管理 AWS Web 登入** → **選擇 Identity Center**（比較方便）。 ✔ **要讓應用程式或指令碼存取 AWS API** → **選擇 IAM 使用者**（需設定 Access Key）。 ✔ **要兩者都能用** → **可以在 Identity Center 設定 Web 存取，然後額外手動建立 IAM 使用者來存取 API**。

如果只是剛開始學習 AWS，建議選擇 **Identity Center**，以後需要 API 存取時，再來新增 **IAM 使用者**。




## **什麼是 IAM Roles？**

**IAM 角色（Roles）** 是 AWS 提供的一種 **授權機制**，允許不同的 AWS 服務或使用者臨時獲得特定權限，而 **不需要直接使用 IAM 使用者的 Access Key**。  
這對於 **AWS 內部服務之間的互動**（例如讓 EC2 存取 S3，或讓 SageMaker 訓練模型）特別重要。

---

## **你看到的四個 IAM Roles 是什麼？**

當你開啟 IAM Roles，會看到 AWS **自動建立** 的角色，這些都是 **AWS 內建的服務角色（Service-linked Roles）**，專門用來讓 AWS 內部服務運作：

1. **AWSServiceRoleForOrganizations**
    
    - **作用**：允許 AWS Organizations 管理多個 AWS 帳戶（如果你的帳戶屬於一個 Organizations）。
    - **是否需要理會？** ❌ 不需要，除非你在管理多個 AWS 帳戶。
2. **AWSServiceRoleForSSO**
    
    - **作用**：允許 **AWS IAM Identity Center（SSO）** 管理你的 AWS 帳戶存取權限。
    - **是否需要理會？** ❌ 不需要，除非你在管理多個使用者。
3. **AWSServiceRoleForSupport**
    
    - **作用**：允許 AWS Support 服務在你請求技術支援時存取你的帳戶資訊。
    - **是否需要理會？** ❌ 不需要，這是 AWS 內建的技術支援角色。
4. **AWSServiceRoleForTrustedAdvisor**
    
    - **作用**：允許 AWS Trusted Advisor 分析你的 AWS 環境，提供安全性和最佳實踐建議。
    - **是否需要理會？** ❌ 不需要，這只是 AWS 的最佳實踐建議工具。

### **🟢 結論：這些角色不影響你的 AI 模型訓練，你不需要刪除或修改它們。**

---

## **IAM Roles 與 AI 模型訓練的關係**

如果你要管理 **AI 模型的訓練** 和 **存取所需的數據集（S3 儲存桶）**，你會需要建立新的 IAM 角色，讓 AWS SageMaker 或 EC2 有權限存取 S3、EFS 或其他 AWS 服務。

你通常會用到 **以下 IAM 角色：**

1. **SageMaker Execution Role（SageMaker 執行角色）**
    
    - **作用**：允許 AWS SageMaker 訓練 AI 模型時存取 S3、ECR（Docker 容器）、CloudWatch Logging 等服務。
    - **如何設定？**
        - 你可以讓 AWS 自動建立，當你第一次開啟 SageMaker Studio 或 Notebook Instance 時，它會詢問你是否要建立一個 **SageMaker Execution Role**，你可以讓 AWS 幫你完成。
        - 或者，你也可以手動建立 IAM 角色，並授權它存取 S3（包含你的資料集）。
2. **EC2 Role（如果你用 EC2 訓練 AI）**
    
    - **作用**：如果你手動在 EC2 訓練 AI，你需要一個角色來授權 EC2 存取 S3（用來讀寫數據）。
    - **如何設定？**
        - 在 IAM 中建立一個 **EC2 專用角色**，並賦予 S3 `ReadWrite` 權限。

---

## **🛠 你應該怎麼做？**

### **1️⃣ 檢查 SageMaker 是否已經有 IAM Role**

- 如果你已經開啟過 **SageMaker Studio**，可以去 **IAM > Roles** 搜尋 `AmazonSageMaker-ExecutionRole` 之類的名稱。
- 如果 **已經有角色**，你可以直接使用它來訓練 AI 模型。
- 如果 **沒有角色**，你需要建立一個。

### **2️⃣ 如果要手動建立 SageMaker IAM Role**

- **步驟**：
    1. 打開 **IAM > Roles > Create Role**
    2. **選擇 AWS 服務**，然後選擇 **SageMaker**
    3. 選擇 **SageMaker - Full Access（完整存取）**
    4. **新增 S3 權限**（讓它能讀寫 S3 儲存桶）
    5. 建立角色，名稱可以取 `SageMakerExecutionRole`
    6. 回到 **SageMaker Studio**，選擇這個角色來運行訓練

---

## **🔹 總結**

- **AWS 自動建立的 IAM Roles（你看到的四個）是內建服務角色，不影響 AI 訓練，你不用管它們**。
- **要讓 AWS SageMaker 或 EC2 訓練 AI 模型，你需要一個 SageMaker 或 EC2 專用的 IAM Role，並授權它存取 S3（你的資料集）**。
- **建議：**
    - **如果 SageMaker 已經有自動建立的 Role，就直接使用它**。
    - **如果沒有，你可以手動建立一個 IAM Role，並授權 SageMaker 存取 S3**。

你可以先試試開啟 SageMaker Notebook，看它是否要求你建立角色，這樣 AWS 會自動幫你設置最適合的權限。



### AWS 的 Security Credentials (安全憑證)

**AWS 的 Security Credentials (安全憑證)** 中，你會看到 **Access Keys** 和 **SSH Public Keys for AWS CodeCommit** 兩種不同的憑證，它們的用途和工作方式完全不同。讓我們詳細解釋它們的差異、用途以及可以應用的場景。

---

## **1. Access Keys（存取金鑰）**

**Access Keys** 主要用於 **AWS API、CLI（Command Line Interface）、SDK 及其他程式化存取 AWS 服務**，可以讓應用程式與 AWS 互動。

### **包含的內容**

- **AWS Access Key ID**（例如：`AKIAIOSFODNN7EXAMPLE`）
- **AWS Secret Access Key**（例如：`wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`）

### **用途**

1. **AWS CLI**：允許你透過終端機（Terminal）或命令提示字元（CMD）使用 AWS 命令，例如：
    
```sh
aws s3 ls
```
    
2. **AWS SDK**：當你使用程式（如 Python、JavaScript、Java）與 AWS 服務互動時，需要 Access Keys 來驗證身份，例如：

```python
import boto3

s3 = boto3.client('s3', aws_access_key_id='你的Access Key', aws_secret_access_key='你的Secret Key')
buckets = s3.list_buckets()
print(buckets)

```

3. **AWS API**：如果你開發應用程式，需要透過 AWS API 存取資源，也需要使用 Access Keys。
4. **Terraform / Ansible / AWS CDK** 等基礎設施自動化工具也依賴 Access Keys 來與 AWS 互動。

### **安全性注意事項**

- **不要將 Access Key 存在公開的地方（例如 GitHub）**，可能會導致 AWS 帳戶被盜用，甚至產生高額費用。
- **建議使用 IAM 角色（Roles）而非 Access Keys**，這樣可以減少明文存放憑證的風險。
- **可以設定 Access Key 的權限**（例如僅允許存取 S3 而不能操作 EC2）。
- **每個 IAM 使用者最多只能有 2 個 Access Keys**，如果超過，必須刪除舊的再建立新的。

---

## **2. SSH Public Keys for AWS CodeCommit（AWS CodeCommit 的 SSH 公鑰）**

這部分的 SSH 公鑰 **僅用於 AWS CodeCommit**，它允許你使用 **SSH 金鑰** 來安全地存取 AWS CodeCommit（AWS 提供的 Git 版本控制服務）。

### **包含的內容**

- **SSH Public Key**（公開金鑰），你必須提供自己的公鑰，例如：
```
ssh-rsa AAAAB3NzaC1yc2EAAAABIwAAAQEAr7...
```
- **SSH Key ID**（AWS 生成的唯一 ID）

### **用途**

1. **使用 Git SSH 方式存取 AWS CodeCommit**
    - 你可以透過 `git clone`、`git push`、`git pull` 來操作 AWS CodeCommit 儲存庫，而不需要 AWS Access Key。
    - 例如：
```sh
    git clone ssh://git-codecommit.us-east-1.amazonaws.com/v1/repos/MyRepo
```

2. **適用於開發者**：如果你熟悉 Git 並想透過 SSH 存取 AWS CodeCommit，這種方式比使用 HTTPS 和 Access Key 更方便。

### **如何設定 SSH Public Key？**

1. **本地端生成 SSH 金鑰**
    
    - 在終端機輸入：
```sh
ssh-keygen -t rsa -b 4096 -C "your-email@example.com"
```
    - 這會在 `~/.ssh/` 目錄下產生：
        - **私鑰 (`id_rsa`)**：請勿公開，存放在本機。
        - **公鑰 (`id_rsa.pub`)**：可以提供給 AWS CodeCommit。
2. **上傳 SSH 公鑰到 AWS**
    
    - 進入 **AWS IAM 控制台** > **安全憑證 (Security credentials)**
    - 找到 **SSH public keys for AWS CodeCommit**
    - 點選 **Upload SSH Public Key**，然後貼上 `id_rsa.pub` 內容。
3. **設定本機 SSH 設定**
    
    - 編輯 `~/.ssh/config`：
```
Host git-codecommit.*.amazonaws.com
  User SSH-Key-ID
  IdentityFile ~/.ssh/id_rsa
```
        
    - 確認連線：
```sh
ssh -T git-codecommit.us-east-1.amazonaws.com
```
    - 如果成功，表示你可以用 SSH 操作 CodeCommit。

### **安全性注意事項**

- **SSH 金鑰只能用於 AWS CodeCommit，不能用來存取 AWS 其他服務。**
- **你可以管理 SSH 金鑰（新增/刪除），但不能直接查看已上傳的 SSH 公鑰內容**。

---

## **Access Keys vs. SSH Public Keys 的主要差異**

|功能|Access Keys|SSH Public Keys for CodeCommit|
|---|---|---|
|**用途**|透過 AWS API、CLI、SDK 存取 AWS 服務|透過 SSH 存取 AWS CodeCommit|
|**包含內容**|Access Key ID + Secret Access Key|SSH 公鑰|
|**適用於**|AWS CLI、SDK、Terraform、Ansible|Git (SSH) 操作 AWS CodeCommit|
|**如何設定**|在 IAM 使用者介面創建並下載|本地產生 SSH Key，並手動上傳|
|**安全性**|Secret Key 需妥善保管，避免洩漏|SSH 私鑰需妥善保管，避免洩漏|
|**存取方式**|需要明文輸入 `aws_access_key_id`|透過 SSH 金鑰驗證，無需輸入密碼|
|**適用場景**|程式化存取 AWS（例如 EC2、S3、DynamoDB）|透過 SSH 存取 AWS CodeCommit|

---

## **結論**

1. **如果你需要程式化存取 AWS（CLI、SDK、API）→ 使用 Access Keys**
    
    - 例如 `aws s3 ls`、`boto3`、Terraform、Ansible、Lambda 等
    - 注意安全性，避免明文存放 Secret Key
    - 最佳做法是使用 **IAM 角色 (Roles) 而非 Access Keys**
2. **如果你使用 Git 並希望透過 SSH 操作 AWS CodeCommit → 使用 SSH 公鑰**
    
    - 例如 `git clone`、`git push`、`git pull`
    - 設定 SSH 金鑰後，不需要再手動輸入密碼或 Access Key
    - 這種方式更安全，適合 Git 版本控制開發者




### 如何在 AWS 上訓練一個簡單的 AI 模型

（例如使用 TensorFlow 或 PyTorch 訓練 MNIST 數據集）。

---

# **第一步：註冊 AWS 並設定 IAM 使用者**

AWS 的 **root account** 權限非常高，不建議直接用來開發，因此我們需要創建一個 IAM 使用者。

### **1. 註冊 AWS 並登錄**

- 你應該已經完成註冊，可以使用 root 帳戶登入 AWS 管理控制台：[https://aws.amazon.com/](https://aws.amazon.com/)
- 確保你已經設置 **帳單資訊** 並啟用 **AWS Free Tier**（如果適用）。

### **2. 創建 IAM 使用者**

1. 打開 **AWS 管理控制台**，在搜尋框輸入 **IAM**（身份與存取管理）。
2. 選擇 **使用者 (Users) > 新增使用者 (Add User)**。
3. 設定使用者名稱（例如 `ai-training-user`）。
4. 選擇存取類型：
    - ✅ **AWS 管理控制台存取 (AWS Management Console access)**（用來登入 Web 控制台）
    - ✅ **程式設計存取 (Programmatic access)**（用來用 CLI 操作）
5. **設定權限**
    - 點擊 **直接附加現有政策 (Attach existing policies directly)**
    - 選擇 `AdministratorAccess`（或更細緻的 `AmazonS3FullAccess`、`AmazonSageMakerFullAccess`、`AWSLambdaFullAccess` 等）
6. **設定登入憑證**
    - 記下 `AWS Access Key ID` 和 `AWS Secret Access Key`，這稍後會用於 CLI。

---

# **第二步：安裝 AWS CLI 並配置**

為了方便與 AWS 互動，我們需要安裝 **AWS Command Line Interface (AWS CLI)**。

### **1. 安裝 AWS CLI**

請根據你的系統選擇安裝方式：

- Windows：[下載 AWS CLI](https://aws.amazon.com/cli/)
- Mac (Homebrew)：`brew install awscli`
- Linux (Ubuntu/Debian)：`sudo apt install awscli`

### **2. 配置 AWS CLI**

打開終端機 (Terminal) 或命令提示字元 (Command Prompt)，輸入：

`aws configure`

系統會要求輸入：

1. **AWS Access Key ID**（來自 IAM 使用者）
2. **AWS Secret Access Key**（來自 IAM 使用者）
3. **預設區域 (Region)**，可以選擇：
    - `us-east-1`（美東 1）
    - `us-west-2`（美西 2）
    - `ap-northeast-1`（東京）
    - `ap-southeast-1`（新加坡）
4. **輸出格式 (Output format)**，通常選 `json`

驗證安裝：

`aws s3 ls`

如果沒有錯誤，表示 AWS CLI 設定成功！

---

# **第三步：啟動 SageMaker Jupyter Notebook 進行開發**

AWS **SageMaker** 提供一個 Jupyter Notebook 環境，讓你可以在雲端訓練 AI 模型。

### **1. 進入 SageMaker**

1. 登入 AWS 管理控制台
2. 搜尋 `SageMaker` 並進入 **Amazon SageMaker**
3. 點選 **筆記本執行個體 (Notebook Instances)** > **創建筆記本執行個體**
4. 設定：
    - 名稱：`ai-training-instance`
    - 執行角色：選擇 `Create a new role`，並允許對 S3 的存取
    - 執行個體類型（Instance Type）：
        - `ml.t2.medium`（適合免費方案）
        - `ml.p2.xlarge`（GPU 訓練，更快但需付費）
    - 存儲：5GB 即可
5. 點擊 **創建執行個體 (Create Notebook Instance)**

### **2. 開啟 Jupyter Notebook**

1. 等待 **狀態 (Status) = InService**
2. 點擊 **Open Jupyter**
3. 創建一個新的 **Python 3 Notebook**

---

# **第四步：下載數據集並訓練 AI 模型**

現在我們使用 **PyTorch** 訓練一個簡單的 MNIST 手寫數字分類模型。

### **1. 安裝 PyTorch**

在 Jupyter Notebook 中運行：

python

複製編輯

`!pip install torch torchvision matplotlib`

### **2. 下載 MNIST 數據集**

python

複製編輯

`import torch import torchvision import torchvision.transforms as transforms  # 設定數據轉換 transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])  # 下載訓練數據 trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform) trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)  # 下載測試數據 testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform) testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)`

### **3. 建立簡單的神經網路**

python

複製編輯

`import torch.nn as nn import torch.optim as optim  # 定義神經網路模型 class SimpleNN(nn.Module):     def __init__(self):         super(SimpleNN, self).__init__()         self.fc1 = nn.Linear(28*28, 128)         self.fc2 = nn.Linear(128, 10)      def forward(self, x):         x = x.view(-1, 28*28)  # 攤平         x = torch.relu(self.fc1(x))         x = self.fc2(x)         return x  model = SimpleNN()`

### **4. 設定損失函數與優化器**

python

複製編輯

`criterion = nn.CrossEntropyLoss() optimizer = optim.Adam(model.parameters(), lr=0.001)`

### **5. 訓練模型**

python

複製編輯

`epochs = 5  for epoch in range(epochs):     running_loss = 0.0     for images, labels in trainloader:         optimizer.zero_grad()         outputs = model(images)         loss = criterion(outputs, labels)         loss.backward()         optimizer.step()         running_loss += loss.item()          print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")  print("訓練完成！")`

---

# **第五步：測試與部署模型**

### **1. 測試模型準確率**

python

複製編輯

`correct = 0 total = 0  with torch.no_grad():     for images, labels in testloader:         outputs = model(images)         _, predicted = torch.max(outputs, 1)         total += labels.size(0)         correct += (predicted == labels).sum().item()  print(f"測試準確率: {100 * correct / total}%")`

### **2. 部署模型到 S3**

如果你要將模型存儲到 AWS S3：

python

複製編輯

`import boto3  s3 = boto3.client('s3') model_path = "model.pth"  # 保存模型 torch.save(model.state_dict(), model_path)  # 上傳到 S3 s3.upload_file(model_path, 'your-s3-bucket-name', model_path) print("模型已上傳至 S3！")`

---

這樣，你就成功在 AWS 上 **訓練了一個簡單的 AI 模型**！🚀

如果你要進一步 **部署到 AWS Lambda 或 SageMaker Endpoint**，可以讓我知道，我可以提供更多的教學！