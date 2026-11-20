
|                                                            |     |
| ---------------------------------------------------------- | --- |
| [[#### 整理在這App裡面一些api jey, certificate, 或密碼]]              |     |
| [[#### LLM gateway解決`ANTHROPIC_API_KEY`、`OPENAI_API_KEY`]] |     |
|                                                            |     |
|                                                            |     |

#### 整理在這App裡面一些api jey, certificate, 或密碼
```
請幫我先整理在這App裡面使用到的一些api jey, certificate, 或密碼等有哪些, 以及有哪些在那考量到App的安全, 以及App會deploy到多個站點等因素, 哪些有風險? 以及如果用LLM gateway是否就可以解決風險問題. 或要搭配其他方式? 以及從目前的code要增加這些風險防護是否要做很多改動, 請深入評估
```
目前最大的問題不只 `ANTHROPIC_API_KEY`，而是 repo 內已經同時存在：

- 進入 Git 歷史的憑證、private key、加密金鑰與 vendor token
- 寫死在程式裡的預設帳號密碼
- 儲存在 Windows 使用者環境變數、YAML 或一般檔案裡的 secrets
- 對區網開放但沒有驗證的分析服務
- 未加密的硬體控制通訊
- AWS IoT 指令驗證繞過與 OTA 更新信任問題

LLM gateway 很值得做，而且現有架構改造成本不算高；但它只能解決「LLM vendor API key 不落地到各站點」這一類問題，不能取代站點身分、TLS/mTLS、Windows secret storage、網路服務驗證、IoT 憑證與 OTA 簽章。

以下檢查沒有輸出或重複任何實際 secret 值，也沒有修改程式。

## Secrets 與憑證盤點

|項目|目前位置／用途|現況|風險|
|---|---|---|---|
|`ANTHROPIC_API_KEY`|[LLM_config.yaml (line 34)](D:/Provenance Laboratories projects/ImagingLibWatch/config/LLM_config.yaml:34) → [LLM_models.py (line 78)](D:/Provenance Laboratories projects/ImagingLibWatch/algorithms/LLM_models.py:78)|從 process environment 取得，用於 Claude image/OCR 分析|高|
|`OPENAI_API_KEY`|[LLM_config.yaml (line 20)](D:/Provenance Laboratories projects/ImagingLibWatch/config/LLM_config.yaml:20)|同樣由環境變數取得|高|
|Windows user environment secrets|[update_env_ui.py (line 110)](D:/Provenance Laboratories projects/ImagingLibWatch/config/update_env_ui.py:110)|App 設定 UI 會把值寫進 `HKCU\Environment`；畫面遮罩不等於加密|高|
|Zaber IoT token|[hardware_config.yaml (line 88)](D:/Provenance Laboratories projects/ImagingLibWatch/config/hardware_config.yaml:88)、[hardware_managers.py (line 203)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/hardware_managers.py:203)|非空 token 直接寫在已追蹤 YAML|嚴重|
|AWS access key／secret key／session token|`aws.*` config、`AWS_*` environment、AWS profile|程式支援 static credentials；目前沒發現非空 access key，`aws_credentials.yaml` 是空檔|潛在高風險|
|AWS IoT private key|`config/certs/private.pem.key`|主工作目錄目前沒有，但在三個 `.claude/worktrees` 和 Git 歷史中找到相同 private key|嚴重|
|AWS IoT client certificate|`certificate.pem.crt`|certificate 本身不是秘密，但需能撤銷及對應唯一站點|中|
|Amazon Root CA|[AmazonRootCA1.pem](D:/Provenance Laboratories projects/ImagingLibWatch/config/certs/AmazonRootCA1.pem)|公開 CA，不是 secret|低|
|Audit HMAC key|`config/keys/hmac.key`、[audit_logger.py (line 41)](D:/Provenance Laboratories projects/ImagingLibWatch/logging_system/audit_logger.py:41)|用來保護 audit log 完整性；副本進過 Git 歷史|高|
|AES-GCM data key|`config/keys/aesgcm.key`、[encryption.py (line 14)](D:/Provenance Laboratories projects/ImagingLibWatch/data_manager/encryption.py:14)|可加密圖片／報告；副本進過 Git 歷史|高|
|Fernet key|`config/keys/fernet.key`|選擇性加密 audit 欄位|高|
|`config/secret.key`|[secret.key](D:/Provenance Laboratories projects/ImagingLibWatch/config/secret.key)|已追蹤，三個 worktree 也有相同內容；目前找不到程式引用，可能是 legacy|中至高|
|Workflow machine token|`network.auth_token`|用於 workflow HTTP API 與 AWS IoT HMAC；目前 local config 的值偏弱|高|
|Controller token|`MOONLIGHT_CONTROLLER_AUTH_TOKEN`|用於 TCP hardware controller|高|
|使用者密碼|[account_manager.py (line 99)](D:/Provenance Laboratories projects/ImagingLibWatch/core/account_manager.py:99)、[setup_global_admins.py (line 87)](D:/Provenance Laboratories projects/ImagingLibWatch/setup_global_admins.py:87)|四組 plaintext bootstrap 密碼寫死於 source|嚴重|
|Password hashes|SQLite／DynamoDB|PBKDF2-HMAC-SHA256、隨機 salt、600k iterations，雜湊方法本身合理|中|

## 最嚴重的實際發現

### 1. 已有 credentials 進入 Git 歷史

在三個 `.claude/worktrees` 與 Git object history 中發現相同的：

- AWS IoT private key
- client certificate
- AES-GCM key
- audit HMAC key
- `secret.key`

如果這個 repo 或任何 branch 曾推送到遠端、備份、CI 或分享給其他人，應把它們視為已外洩。

只刪除目前檔案不夠，正確順序是：

1. 先撤銷／輪替 AWS IoT certificate 和所有相關金鑰。
2. 確認 AES/HMAC key 曾保護哪些既有資料。
3. 清理 branch、worktree 與 Git 歷史。
4. 要求既有 clone 重新同步或重新 clone。
5. 在 CI/pre-commit 加 secret scanning。

### 2. 寫死的預設密碼存在可被持續繞回的風險

更嚴重的是，[AccountManager 的 fallback (line 184)](D:/Provenance Laboratories projects/ImagingLibWatch/core/account_manager.py:184) 在資料庫密碼不符時，還會嘗試 source code 中的預設密碼，成功後可能把本機記錄重新寫回預設值。

所以這不只是「初次安裝有預設密碼」，而可能導致使用者改密碼後，舊的 source-code 密碼在重啟後仍成為可用 fallback。這應優先移除。

建議改成：

- 首次部署隨機產生一次性 bootstrap credential。
- 首次登入強制改密碼。
- 不在 binary/source 保留 fallback 密碼。
- 增加登入失敗 rate limit、lockout 與 audit。
- 管理者密碼不要由部署 script 寫成固定值。

### 3. 多個服務直接暴露到區網

我找到 14 個 `uvicorn` server 以 `0.0.0.0` 綁定，其中約 13 個影像分析 server 沒有明顯 authentication middleware。

這些 endpoint 能接收 `image_path`、`output_dir`、`file_path` 等本機路徑，風險包含：

- 未授權觸發 GPU/CPU 密集任務
- 本機檔案存在性探測
- 寫入非預期輸出位置
- DoS
- 對後續硬體或 workflow 的橫向移動

如果服務只供同一台機器使用，應直接 bind `127.0.0.1`。如果必須跨主機，則需要 TLS/mTLS、per-service authorization、輸入路徑 allowlist 與 rate limiting。

### 4. Hardware controller token 會以明文網路資料傳送

[controller_server.py (line 29)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/controller_server.py:29) 使用 static token 驗證，但底層是 raw TCP JSON，沒有 TLS。

對硬體控制來說風險特別高：區網中若能攔截 token，可能重播 stage movement、capture 等命令。

至少需要：

- TLS 或 mTLS
- 每站唯一憑證
- nonce／timestamp／replay protection
- 指令層級 authorization
- 緊急停止與 hardware safety interlock 不依賴純軟體 token

### 5. AWS IoT 指令存在開發用 bypass

[aws_agent.py (line 165)](D:/Provenance Laboratories projects/ImagingLibWatch/cloud_relay/aws_agent.py:165) 允許特殊開發 signature 直接跳過驗證。

另外，目前 HMAC 只簽部分欄位，沒有完整涵蓋所有 command parameters。攻擊者若能修改未簽欄位，訊息仍可能通過。

這應改為：

- production build 完全移除 bypass，而不是只靠設定關閉。
- 對 canonicalized 完整 payload 簽章。
- 加入 expiry、nonce、CommandID replay ledger。
- AWS IoT policy 嚴格限制誰能 publish 到每台 device topic。

### 6. OTA 更新信任鏈不足

AWS IoT Jobs 更新流程會下載 ZIP 並解壓到 project root，但目前沒有看到完整的 artifact signature/hash 驗證與安全解壓路徑檢查。

這是高影響供應鏈入口。需要：

- manifest + SHA-256
- KMS 非對稱簽章驗證
- ZipSlip 路徑驗證
- staging directory
- atomic switch／rollback
- 更新執行前驗證 publisher 與版本 policy

現有 KMS signing architecture 是很好的基礎，應把同一信任模型套用到 OTA artifact。

## LLM gateway 能解決什麼

值得做。由於 Claude/OpenAI 的使用集中在 [LLM_models.py (line 529)](D:/Provenance Laboratories projects/ImagingLibWatch/algorithms/LLM_models.py:529)，不需要重寫整個 image pipeline。

理想流程是：

```
Station App
    │  站點憑證／短效 token
    ▼
Company LLM Gateway
    │  從 Secrets Manager 取得 vendor key
    ▼
Anthropic / OpenAI
```

它可以解決：

- `ANTHROPIC_API_KEY`、`OPENAI_API_KEY` 不再落到每個 Windows 站點。
- 單一位置輪替 vendor key。
- 各站點 quota、模型 allowlist、費用限制。
- 集中 audit、timeout、retry、內容大小限制。
- 可阻擋任意 prompt/model 或濫用。
- 每個站點被撤銷時，不需要重發 Anthropic key。
- 避免 App key 與 Claude Code/Remote Control 使用者環境互相污染。

Anthropic 也建議使用環境變數、分離用途、定期輪替並監控使用量；但對大量部署的桌面 App，gateway 通常比把共同 vendor key 放到每台 station 更合適。[Anthropic API key best practices](https://support.claude.com/en/articles/9767949-api-key-best-practices-keeping-your-keys-safe-and-secure)

## LLM gateway 不能解決什麼

Gateway 不會解決：

- AWS IoT private key
- Zaber token
- controller token
- 使用者帳號密碼
- 本機 AES/HMAC/Fernet keys
- 沒有 authentication 的 LAN services
- controller 明文 TCP
- IoT HMAC bypass
- OTA 更新信任
- 本機圖片與資料庫未加密
- station 本身被入侵

而且 gateway 會成為新的高價值目標，因此仍需要：

- Gateway vendor key 放 AWS Secrets Manager，開 rotation、CloudTrail 與最小權限。[AWS Secrets Manager best practices](https://docs.aws.amazon.com/secretsmanager/latest/userguide/best-practices.html)
- Station 使用唯一身分；優先 mTLS certificate 或硬體綁定 private key。
- Gateway 發短效 access token，不使用所有站點共用的長效 shared secret。
- 每站點 rate limit、cost quota 與 revocation。
- 限制影像大小、格式與 metadata。
- 明確決定圖片是否允許送到第三方 LLM。

Gateway 只隱藏 key，不會讓影像停止離開公司環境。標準 Anthropic API 的 inputs/outputs 一般會有最長 30 天的保存政策及例外，因此仍須做資料分類、必要區域裁切、EXIF 移除及合約／資料保留設定確認。[Anthropic data retention](https://privacy.claude.com/en/articles/7996866-how-long-do-you-store-my-organization-s-data)

## Windows 站點應如何保存仍然必須存在的秘密

不要把長期 secrets 放在：

- tracked YAML
- `.env`
- `HKCU\Environment`
- executable 旁的一般 key file
- 所有站點共用的 config

建議依用途分開：

- 一般 App credential：Windows Credential Locker／Credential Manager。
- 綁定使用者或機器的資料：Windows DPAPI。[Microsoft DPAPI](https://learn.microsoft.com/windows/uwp/security/data-protection)
- AWS IoT private key：Windows Certificate Store 或 TPM-backed key；至少用 NTFS ACL 限制到專用 service account。
- 本機 data-encryption master key：由 DPAPI/TPM 保護，不與被加密資料放在同一目錄的裸檔案。
- AWS API：IAM role、IoT credentials provider 或 Roles Anywhere，避免 static AWS access key。
- 每站點都要有不同身分，才能單獨撤銷。

目前 deployment script 明確沒有設定 NTFS ACL；`D:\Moonlight\Protected` 只是路徑名稱，不是實際安全邊界。雖然 [deploy_moonlight.py (line 64)](D:/Provenance Laboratories projects/ImagingLibWatch/helper/deployment/deploy_moonlight.py:64) 會預設清除 secrets 並產生隨機 token/key，仍需要補 ACL、service account 與 secure provisioning。

## 改造成本評估

### LLM gateway：中低程度 repo 改動

若維持既有 `LLMInferenceResult` 介面，預估只需改約 4–8 個檔案：

- `algorithms/LLM_models.py`：新增 gateway provider/client
- `config/LLM_config.yaml`：加入 gateway URL、timeout、model mapping
- `config/update_env_ui.py`：移除 vendor key 的站點輸入
- deployment/config sanitizer
- requirements、tests、文件
- OCR helper 理論上可保持不變

粗估：

- Repo client 改造：2–5 engineer-days
- Gateway MVP + Secrets Manager + auth + logging：5–10 engineer-days
- Production-grade mTLS、HA、quota、monitoring、privacy policy：2–4 週

主要工作量會在 gateway infrastructure 與 station authentication，不是 image analysis pipeline。

### 全面安全強化：中至高程度

|工作|粗估|
|---|---|
|旋轉洩漏憑證、清理 Git history|1–3 天，另有部署協調|
|分析服務改 loopback／共用 auth middleware|1–5 天|
|移除預設密碼 fallback、首次啟用流程、lockout|3–6 天|
|DPAPI/Credential Manager secret abstraction|3–7 天|
|專用 Windows service account + ACL installer|2–5 天|
|Controller TLS/mTLS、replay protection|1–2 週，需硬體測試|
|AWS IoT 完整 payload 簽章與 bypass 移除|3–7 天|
|OTA 簽章、safe extraction、rollback|1–2 週|
|全部整合、現場部署與回歸測試|額外 1–2 週|

整體建議分階段，不需要一次重寫 App：

1. 立即輪替已進 Git 的 token/key/certificate，移除預設密碼 fallback 與 IoT bypass。
2. 將本機分析服務改成 loopback，封鎖不必要的 inbound ports。
3. 導入 LLM gateway，從 station 移除 Anthropic/OpenAI key。
4. 建立統一 Station Secret Store abstraction。
5. 補上 controller mTLS、完整 IoT 簽章與安全 OTA。
6. 最後處理資料落地加密、稽核、集中部署與憑證生命週期。

就現況而言，LLM gateway 是一個改動集中、效益高的第一階段，但不能單獨當成整個 App 的安全解法。




#### LLM gateway解決`ANTHROPIC_API_KEY`、`OPENAI_API_KEY`
```
那請幫我用LLM gateway解決`ANTHROPIC_API_KEY`、`OPENAI_API_KEY` 或其他問題. 那除了修改code之外, 我需要做甚麼嗎
```
已完成 LLM Gateway 的程式改造；App 預設不再直接讀取 `ANTHROPIC_API_KEY` 或 `OPENAI_API_KEY`。不過 Gateway 尚未部署到 AWS，部署與每站 provisioning 仍需要你完成。

## 已完成的修改

- OpenAI、Claude 預設都改走 Gateway：[LLM_config.yaml (line 20)](D:/Provenance Laboratories projects/ImagingLibWatch/config/LLM_config.yaml:20)
- Gateway client、HTTPS 強制、station authentication：[LLM_models.py (line 454)](D:/Provenance Laboratories projects/ImagingLibWatch/algorithms/LLM_models.py:454)
- 中央 FastAPI Gateway：
    - OpenAI／Anthropic model allowlist
    - 圖片數量、大小、prompt、output token 限制
    - 每站 rate limit
    - per-station token authentication
    - Gateway 只保存 token SHA-256 hash
    - 可用 DynamoDB 管理多站點
    - 不記錄圖片、prompt、token 或 vendor key  
        見 [app.py (line 99)](D:/Provenance Laboratories projects/ImagingLibWatch/llm_gateway/app.py:99)
- Station token 存 Windows Credential Manager：[credential_store.py (line 53)](D:/Provenance Laboratories projects/ImagingLibWatch/core/credential_store.py:53)
- 自動產生每站 token：[provision_station.py](D:/Provenance Laboratories projects/ImagingLibWatch/llm_gateway/provision_station.py)
- Docker deployment：[Dockerfile](D:/Provenance Laboratories projects/ImagingLibWatch/llm_gateway/Dockerfile)
- API key 設定 UI 已移除 vendor key，改成 Gateway URL 與 station ID：[update_env_ui.py (line 100)](D:/Provenance Laboratories projects/ImagingLibWatch/config/update_env_ui.py:100)
- 完整部署說明：[README.md (line 9)](D:/Provenance Laboratories projects/ImagingLibWatch/llm_gateway/README.md:9)

驗證結果：5 個新增 Gateway 測試及 12 個既有 deployment/config 測試，共 17 個通過；相關 Python 檔案也通過語法編譯。

## 你還需要做的事

### 一次性的中央 Gateway 建置

我建議沿用現有 AWS 架構：

1. 建立 Gateway 網域，例如：
    
    ```
    llm-gateway.yourcompany.com
    ```
    
2. 將 Docker image 部署到 ECS Fargate。
    
3. 在前面放 Application Load Balancer：
    
    - 只開 HTTPS 443
    - 憑證放 ACM
    - ECS 8080 不直接對 Internet 開放
    - 加 AWS WAF
    - 在 ALB/WAF 設約 30 MB request-body 上限
    
    AWS 建議使用 ACM 管理 ALB HTTPS certificate，也支援 mTLS。[AWS ALB HTTPS 文件](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/create-https-listener.html)
    
4. 將這兩個 vendor keys 建立在 AWS Secrets Manager：
    
    ```
    OPENAI_API_KEY
    ANTHROPIC_API_KEY
    ```
    
    ECS task definition 要用 `secrets` 注入，不能放在一般 `environment`、Docker image 或 task definition 明文欄位。[AWS ECS Secrets Manager 文件](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/secrets-envvar-secrets-manager.html)
    
    注意：Secrets Manager 旋轉後，既有 ECS task 不會自動收到新值，需要 force new deployment。
    
5. 建立 DynamoDB station registry：
    
    Partition key：
    
    ```
    station_id  String
    ```
    
    每筆資料：
    
    ```
    {
      "station_id": "site01-station01",
      "token_hash": "<sha256>",
      "enabled": true
    }
    ```
    
    Gateway ECS task role 只需要該 table 的 `dynamodb:GetItem`。
    
6. 設定 Gateway 非秘密環境變數：
    
    ```
    LLM_GATEWAY_STATION_TABLE=MoonlightLLMGatewayStations
    LLM_GATEWAY_OPENAI_MODELS=gpt-4.1-mini
    LLM_GATEWAY_ANTHROPIC_MODELS=claude-sonnet-4-5
    LLM_GATEWAY_REQUESTS_PER_MINUTE=30
    LLM_GATEWAY_MAX_IMAGES=8
    LLM_GATEWAY_MAX_TOTAL_IMAGE_BYTES=20000000
    ```
    

### 每一個 Windows 站點都要做

1. 指定唯一 station ID：
    
    ```
    site01-station01
    site01-station02
    site02-station01
    ```
    
2. 使用實際執行 App 的同一個 Windows 帳號，執行：
    
    ```
    conda run -n watch_env python -m llm_gateway.provision_station --station-id site01-station01
    ```
    
    這會：
    
    - 產生高強度隨機 token
    - 將 token 寫進 Windows Credential Manager
    - 只輸出 SHA-256 hash，不輸出 plaintext token
3. 管理員把輸出的 hash 加入 DynamoDB。
    
4. 設定兩個非秘密變數：
    
    ```
    MOONLIGHT_STATION_ID=site01-station01
    LLM_GATEWAY_URL=https://llm-gateway.yourcompany.com
    ```
    
    可以透過更新後的 environment UI 設定。
    
5. 更新站點上的：
    
    ```
    D:\Moonlight\Protected\config\LLM_config.yaml
    ```
    
    這點很重要：現有 deployment 預設不覆寫已存在的 station config，所以舊站點不會因為換新版 EXE 就自動切換 Gateway。需要手動更新，或部署時明確覆寫這個設定。
    
6. 重啟 App，測試 Claude 與 OpenAI image analysis。
    
7. 確認 Gateway 成功後，刪除 station 上舊的：
    
    ```
    ANTHROPIC_API_KEY
    OPENAI_API_KEY
    ANTHROPIC_BASE_URL
    OPENAI_BASE_URL
    ```
    
    包含 User environment、Machine environment、啟動 script、舊 `.env` 和 YAML。刪除後要重新啟動 App／Windows service。
    

## Gateway 沒有解決的項目

以下不能放進 LLM Gateway，仍需另外處理：

- repo 內的 Zaber IoT token
- Git 歷史中的 AWS IoT private key
- 寫死的管理員預設密碼與 fallback
- hardware controller 的明文 TCP token
- AWS IoT `SKIP_VERIFY_DEV` bypass
- 沒有 authentication 的區網分析 API
- OTA ZIP 簽章與安全解壓

這些需要分別使用憑證輪替、Windows Credential Manager／Certificate Store、TLS/mTLS、帳號 bootstrap 重構及 OTA artifact signing。LLM Gateway 只負責 LLM vendor key 與 LLM 使用治理。

目前剩下的主要阻礙不是程式，而是 AWS account、region、網域、VPC 與憑證資訊；有這些資訊後，下一步可以把 Gateway 的 ECS、ALB、ACM、Secrets Manager、DynamoDB 與 WAF 做成 CloudFormation/Terraform，避免手動建立。





```

```
