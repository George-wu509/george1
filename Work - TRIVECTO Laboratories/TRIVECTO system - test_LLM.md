

|     |     |
| --- | --- |
|     |     |
|     |     |
|     |     |
|     |     |

```

```



```
請詳細解釋LLM_config.yaml以下的parameters要如何設定? API key是輸入在api_key? 那 api_key_env跟base_url各是甚麼因為我在openai跟claude的api設定page沒看到這些 openai: enabled: true provider: "openai" model: "gpt-4.1-mini" api_key: "" api_key_env: "OPENAI_API_KEY" base_url: "" max_output_tokens: 1200 temperature: 0.1 response_format: "text" claude: enabled: true provider: "anthropic" model: "claude-3-5-sonnet-latest" api_key: "" api_key_env: "ANTHROPIC_API_KEY" base_url: "" max_output_tokens: 1200 temperature: 0.1 llama_local: enabled: true provider: "llama_local" # Examples: # - Ollama with llava/llama vision model: endpoint_type=ollama, endpoint_url=http://localhost:11434/api/chat # - llama.cpp/OpenAI-compatible server: endpoint_type=openai_compatible, endpoint_url=http://localhost:8000/v1/chat/completions model: "llava:latest" endpoint_type: "ollama" endpoint_url: "http://localhost:11434/api/chat" api_key: "" api_key_env: "" max_output_tokens: 1200 temperature: 0.1 stream: false
```

可以，重點先講結論：

`api_key` 可以直接放 API key，但比較不建議。  
`api_key_env` 不是 API key，而是「環境變數名稱」。  
`base_url` 不是 OpenAI/Claude 後台要設定的東西，而是程式端用來覆蓋 API 伺服器網址的設定；用官方 OpenAI / Anthropic API 時通常留空。

我看了 repo 的實作，key 的讀取邏輯在 [LLM_models.py (line 73)](/d:/Provenance Project/ImagingLibWatch/algorithms/LLM_models.py:73)：

```
api_key_env -> 如果環境變數有值，優先使用
api_key     -> 如果環境變數沒值，才使用 YAML 裡的 api_key
```

所以推薦設定是這樣：

```
openai:
  api_key: ""
  api_key_env: "OPENAI_API_KEY"
  base_url: ""

claude:
  api_key: ""
  api_key_env: "ANTHROPIC_API_KEY"
  base_url: ""
```

然後在 Windows PowerShell 設環境變數：

```
setx OPENAI_API_KEY "你的 OpenAI API key"
setx ANTHROPIC_API_KEY "你的 Anthropic API key"
```

設完 `setx` 之後要重開 terminal / app，因為它不會更新目前已開啟的程序。

如果只是臨時測試目前 PowerShell 視窗，可以用：

```
$env:OPENAI_API_KEY="你的 OpenAI API key"
$env:ANTHROPIC_API_KEY="你的 Anthropic API key"
```

各欄位意思如下：

`enabled`  
是否啟用這個服務。`true` 代表 UI / client 會把它當成可用服務。

`provider`  
告訴程式要走哪一種 API 呼叫邏輯。  
OpenAI 用：

```
provider: "openai"
```

Claude / Anthropic 用：

```
provider: "anthropic"
```

本機 Llama / Ollama 用：

```
provider: "llama_local"
```

`model`  
要使用的模型名稱。例如：

```
model: "gpt-4.1-mini"
model: "claude-3-5-sonnet-latest"
model: "llava:latest"
```

這個名稱必須是該 provider 支援的模型名稱。OpenAI 和 Anthropic 的模型名稱不共用。

`api_key`  
直接把 key 寫在 YAML 裡。可以用，但不推薦，因為容易被 git commit 或分享出去。

例如：

```
api_key: "sk-..."
api_key_env: ""
```

這樣會直接用 YAML 裡的 key。

`api_key_env`  
環境變數的名稱，不是 key 本身。

例如：

```
api_key_env: "OPENAI_API_KEY"
```

意思是程式會去讀 Windows / shell 裡的：

```
$env:OPENAI_API_KEY
```

Claude 同理：

```
api_key_env: "ANTHROPIC_API_KEY"
```

這就是為什麼你在 OpenAI 或 Claude 的 API settings page 沒看到 `api_key_env`，因為它不是 provider 的設定，而是你這個程式自己的設定方式。

`base_url`  
API 伺服器的基底網址。用官方 OpenAI / Anthropic API 時留空：

```
base_url: ""
```

留空時，OpenAI SDK / Anthropic SDK 會自己使用官方預設 API endpoint。

只有這些情況才需要填 `base_url`：

```
base_url: "http://localhost:8000/v1"
```

或：

```
base_url: "https://your-proxy.example.com/v1"
```

常見用途包括：

- 公司內部 API gateway
- LiteLLM / OpenRouter / proxy server
- OpenAI-compatible local server
- 測試 mock server

一般你從 OpenAI 或 Anthropic 官方網站拿 API key 時，不需要管 `base_url`。

`max_output_tokens`  
限制模型最多輸出多少 token。這不是圖片大小，也不是輸入 prompt 大小，而是回答長度上限。

```
max_output_tokens: 1200
```

想要更長回答可以調高，例如 `2000`、`4000`。但會花更多時間和費用。

`temperature`  
控制回答隨機性。

```
temperature: 0.1
```

影像分析、OCR、缺陷描述這種任務建議低一點，像 `0` 到 `0.2`。  
創意描述才會用高一點，例如 `0.7`。

`response_format`  
你 YAML 裡 OpenAI 有：

```
response_format: "text"
```

但以目前 repo 實作來看，這個欄位沒有直接被 `_run_openai()` 使用。實際決定輸出 text 或 JSON 的是 task 裡的：

```
output_mode: "text"
```

或：

```
output_mode: "json"
```

相關邏輯在 [LLM_models.py (line 244)](/d:/Provenance Project/ImagingLibWatch/algorithms/LLM_models.py:244)。

`llama_local.endpoint_type`  
本機模型服務類型。

Ollama 用：

```
endpoint_type: "ollama"
endpoint_url: "http://localhost:11434/api/chat"
model: "llava:latest"
```

OpenAI-compatible server 用：

```
endpoint_type: "openai_compatible"
endpoint_url: "http://localhost:8000/v1/chat/completions"
```

注意：`llama_local.endpoint_url` 是完整 endpoint；但 OpenAI / Claude 的 `base_url` 是 base URL，不是完整 `/chat/completions` endpoint。

`llama_local.api_key` / `api_key_env`  
本機 Ollama 通常不需要 key，所以可以空白：

```
api_key: ""
api_key_env: ""
```

如果你接的是需要 Bearer token 的本機或 proxy server，才填。

最建議你現在用這版：

```
openai:
  enabled: true
  provider: "openai"
  model: "gpt-4.1-mini"
  api_key: ""
  api_key_env: "OPENAI_API_KEY"
  base_url: ""
  max_output_tokens: 1200
  temperature: 0.1
  response_format: "text"

claude:
  enabled: true
  provider: "anthropic"
  model: "claude-3-5-sonnet-latest"
  api_key: ""
  api_key_env: "ANTHROPIC_API_KEY"
  base_url: ""
  max_output_tokens: 1200
  temperature: 0.1
```

一句話版：  
API key 建議不要寫進 `api_key`，而是放到系統環境變數；`api_key_env` 填環境變數名稱；`base_url` 用官方 API 時留空。