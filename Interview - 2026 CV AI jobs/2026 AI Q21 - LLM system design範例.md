
|                              |     |
| ---------------------------- | --- |
| [[#### LLM system design範例]] |     |
|                              |     |
|                              |     |

#### LLM system design範例
```
如果你是面試者要回答以下面試題目, 要精確但完整深入回答給面試官
 假設面試官問：

> Design an enterprise AI assistant that uses internal documents, calls business tools, and supports 10,000 users while enforcing document access permissions.

你應能設計出類似：

User / Authentication / Authorization
API Gateway / Agent Orchestrator
RAG Pipeline
ACL Filter · Retrieval · Rerank
Tool Execution
Schema · Approval · Sandbox
LLM / Model Routing / Inference
Verification / Response / Citations
Evaluation / Tracing / Feedback / Monitoring

面試官通常會追問：如果文件每天更新怎麼辦？如果 Vector Search 找錯資料呢？如果 Agent 呼叫的 Tool 失敗？如果使用者要求未授權操作？如果 Token Cost 增加三倍如何處理？

Senior Engineer 需要從 Architecture、Failure Recovery、Cost、Security、Latency 與 Evaluation 同時回答，而不是只畫出 LLM API 和 Vector Database。
```
