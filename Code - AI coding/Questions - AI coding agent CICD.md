

|                                                     |     |
| --------------------------------------------------- | --- |
| [[#### 在interview是否會問相關問題?]]                        |     |
| [[#### 1. AI-Assisted Coding & Development  ]]      |     |
| [[#### 2  Code Review、Testing & Debugging]]         |     |
| [[#### 3  CI/CD、DevOps & Release Engineering ]]     |     |
| [[#### 4  DevSecOps & AI Coding Security]]          |     |
| [[#### 5  Agent Architecture、Tool Calling & Evals]] |     |
| [[#### 6  MLOps、CV Production & AI Deployment]]     |     |
| [[#### 7  System Design、Reliability & Leadership]]  |     |
| [[#### 最建議你先精通的 15 題]]                              |     |
|                                                     |     |

#### 在interview是否會問相關問題?
```
所以今年開始在interview是否會問相關問題, 頻率大嗎? 請幫我整理成幾個大類, 各大類各提供30題常在interview會問的問題.
```

### 2026 美國 Senior Software / AI Engineer 面試題庫：7 大類、210 題

結論：會，而且從 2025 年開始，AI Coding、AI Code Review、Agentic Development、CI/CD、Production Engineering 已經逐步進入技術面試。但不同職位的提問頻率差異很大。

如果你準備的是美國 Senior / Staff Software Engineer、Senior Computer Vision Engineer、Senior Applied AI Engineer 或 AI Agent Engineer，我建議你系統化準備以下 7 大類，共 210 題。

這份題庫以英文呈現面試問題、中文輔助理解，並標示優先級。題目根據 2026 年招聘資料、公司官方面試指南、職缺技能要求及常見實務情境整理；它是高實用性的模擬題庫，不代表這 210 題都有可核實的逐題出現頻率。

## 一、2026 年，這類問題真的問得多嗎？

### 1. 實際招聘數據

2026 年技術面試中的 AI 相關問題

# 47%

ML / AI Engineer

面試回報提及 AI、LLM 或 GenAI

# 9%

Software Engineer

面試回報提及 AI、LLM 或 GenAI

Interview Query 2026 年自行提交的第一手面試回報：ML/AI Engineer n=81，Software Engineer n=681。這些是樣本中的「AI 相關提及率」，並非所有美國職缺的實際提問機率，也不是 AI Coding 專項比例。

Interview Query

2026 年 CoderPad 的招聘調查也提供了另外兩項重要資訊：

# 46%

允許使用 AI 的面試

29% 有限制，17% 廣泛允許

# 34%

禁止 AI 的面試

另外 20% 依個案決定

CoderPad

這代表兩件事：

第一，AI 相關能力越來越重要，但不能認為現在每家公司都會直接問 Codex / Claude Code，或允許你在 Coding Interview 使用 AI。

第二，傳統 Software Engineering 基礎並沒有消失。CoderPad 2026 調查指出，招聘方使用的技術測驗仍包含 Algorithmic Questions（43%）、System Design（38%）、Code Review（21%）和 Debugging（19%）。這些比例來自同一份問卷的多選題。

CoderPad

### 2. 哪些公司已經改變面試方式？

|公司|2026 年可確認的方向|
|---|---|
|Canva|2025 年 6 月就宣布 Backend、Frontend、ML 候選人應使用 AI 工具參與技術面試，更強調理解及修正 AI-generated Code|
|OpenAI|面試依職位及輪次決定是否允許 AI；依然重視 Coding、Performance、Testing、Design|
|Microsoft|官方面試指南強調 Coding、Testing、System Design、Security、Distributed Systems 與 ML|
|Anthropic|官方仍列出 Live Coding 等評估，不能假設所有技術面試都已改用 AI-assisted Coding|

另外，OpenAI 的 Codex Core Agent 職缺明確要求 Agent Execution、Tool Use、Sandboxing、Orchestration、Evals、Production Reliability、Latency 與 Cost Optimization。這是 AI Agent 類職位會深入考察這些能力的重要證據。

##### 如果準備時間有限，應該先讀哪幾類？

依照你的履歷背景與求職方向，我建議如下。

|目標職位|第一優先|第二優先|第三優先|
|---|---|---|---|
|Senior Software Engineer|B、G、C|A、D|E、F|
|Staff / Lead Software Engineer|G、C、B|D、A|E、F|
|Senior Computer Vision Engineer|F、B、G|C、A|D、E|
|Senior Applied AI Engineer|E、G、B|A、F、C|D|
|Senior AI Agent Engineer|E、A、D|B、G|C、F|
|Senior ML / AI Infrastructure Engineer|F、C、G|B、D|A、E|

此排序是針對上述職位的常見責任進行判斷，具體還需要依據每家公司 Job Description 調整。



## 二、7 大類與面試優先級

以下的「高、中、低」是我針對不同職位的準備優先級評估，不是統計上的實際出題率。

| 類別  | 面試方向                                    | Senior SWE | Senior CV / AI | AI Agent Engineer |
| --- | --------------------------------------- | ---------- | -------------- | ----------------- |
| A   | AI-Assisted Coding & Development        | 中高         | 中              | 極高                |
| B   | Code Review、Testing & Debugging         | 極高         | 高              | 極高                |
| C   | CI/CD、DevOps & Release Engineering      | 高          | 高              | 高                 |
| D   | DevSecOps & AI Coding Security          | 中高         | 中高             | 極高                |
| E   | Agent Architecture、Tool Calling & Evals | 中          | 中              | 極高                |
| F   | MLOps、CV Production & AI Deployment     | 中          | 極高             | 高                 |
| G   | System Design、Reliability & Leadership  | 極高         | 極高             | 極高                |

針對你的 Computer Vision、Production C++ / Python、GPU Inference、AWS 和 MLOps 經驗，我尤其建議優先準備 B、C、F、G，再補強 A、D、E。

以下每個大類有 30 題，前面標註 ★ 的題目是我建議你優先準備的 10 題。這些題目不只是背定義，還應能回答設計、Tradeoffs、Failure Modes 及實際案例。



#### 1. AI-Assisted Coding & Development  
```
### A. AI-Assisted Coding & Agentic Software Development
```
30 題｜Codex、Claude Code、Copilot、Agent Workflow、Context Engineering、AI-generated Code

這一類是 2025–2026 年快速增加的面試方向，尤其適合會實際使用 AI Coding Tools 的團隊，以及 Applied AI / Developer Tools 相關職位。

##### A1. AI Coding Fundamentals（1–10）

1. ★ How do you use AI coding assistants in your daily software development workflow?  
你平常如何把 Codex、Claude Code 或 Copilot 整合到 Software Development Lifecycle？

2. What are the differences between AI code completion, AI-assisted coding, and autonomous coding agents?  
三種 AI Coding 方式的自主程度與使用情境有何不同？

3. ★ How do you verify that AI-generated code is correct, secure, and maintainable?  
如何證明 AI 產生的程式碼符合 Production Quality？

4. How do you decide which coding tasks should be delegated to an AI agent?  
哪些工作可以交給 Agent？哪些必須由工程師主導？

5. How do you write effective prompts for complex software engineering tasks?  
如何提供 Context、Requirements、Constraints 與 Acceptance Criteria？

6. How do you prevent AI coding assistants from introducing unnecessary dependencies?  
如何控制套件、版本及相依性？

7. How do you use AI to refactor legacy code without changing existing behavior?  
如何透過 Characterization Tests 保持原有功能？

8. ★ How would you use an AI agent to implement a feature in an unfamiliar codebase?  
如何探索 Repository、了解 Architecture、規劃修改與驗證？

9. What are the limitations of using LLMs for code generation?  
如何處理 Hallucination、錯誤 API、缺乏 Context、Non-determinism？

10. How do you measure productivity improvements from AI-assisted development?  
如何衡量 Development Time、Review Cost、Rework、Defect Rate，而非只計算產生多少程式碼？

### A2. Agentic Development Workflow（11–20）

11. ★ Design an end-to-end AI-assisted software development workflow from GitHub Issue to merged Pull Request.  
設計 Issue → Planning → Coding → Testing → Review → Merge 的完整流程。

12. What is the purpose of AGENTS.md or repository-level AI coding instructions?  
如何管理 Coding Standards、Architectural Constraints、Test Requirements？

13. How would you structure an AI coding task to minimize hallucinations and unnecessary changes?  
如何拆分任務並限制修改範圍？

14. ★ How do you manage context windows when an AI agent works on a large monorepo?  
如何進行 Repository Search、Relevant File Selection、Context Compression？

15. How do you prevent an AI agent from modifying unrelated files?  
如何設計 Filesystem Scope、Diff Validation 和 Approval Gates？

16. How would you coordinate multiple coding agents working on the same repository?  
如何處理 Branch Isolation、Merge Conflicts、Shared State？

17. How do you decide whether a coding task requires one agent or multiple agents?  
Single-Agent 與 Multi-Agent 的品質、Latency、Cost Tradeoffs 是什麼？

18. How would you design a Planner–Coder–Reviewer–Verifier workflow?  
四種角色的 Responsibilities、Interfaces、Failure Handling 應如何定義？

19. ★ What should happen if an AI coding agent repeatedly fails to fix a test?  
如何設定 Retry Limits、Failure Classification、Escalation？

20. How do you preserve context and progress across long-running coding tasks?  
Checkpoints、Task State、Summaries、Artifact Persistence 如何設計？

### A3. Advanced Coding Agent Engineering（21–30）

21. How would you use AI agents to migrate a large Python codebase to a newer API version?  
如何保證 Backward Compatibility？

22. How would you integrate an AI coding agent with GitHub Actions?  
Agent 應在哪個 CI 階段執行？如何控制 Permissions？

23. ★ How would you design a coding agent that can run tests, debug failures, and propose patches?  
請設計 Tool Calling、Execution Loop、Verification 及 Patch Generation。

24. How do you prevent an AI coding agent from overfitting to the visible test cases?  
如何使用 Independent Tests 或 Hidden Regression Cases？

25. How would you evaluate two coding agents on the same benchmark?  
如何比較 Solve Rate、Patch Quality、Latency、Token Cost？

26. How would you diagnose a coding agent that succeeds on small repositories but fails on large ones?  
如何分析 Context Retrieval、Tool Use、Planning 與 Environment Failure？

27. How do you optimize token usage and cost in agentic coding workflows?  
如何做 Context Reduction、Caching、Model Routing？

28. ★ How would you design a safe autonomous coding agent for a production repository?  
如何設計 Sandboxing、Least Privilege、Approval、Audit Logs？

29. What is the difference between building a coding agent with a general agent framework versus a custom execution harness?  
何時需要自行設計 Agent Execution Loop？

30. ★ Your team claims AI coding has doubled productivity but production defects have increased. How would you investigate?  
如何區分 Coding Speed、Review Bottlenecks、Test Effectiveness 與 Change Failure Rate？




#### 2  Code Review、Testing & Debugging   
```

```
# . Code Review、Automated Testing & Debugging

30 題｜pytest、Quality Gates、Regression Testing、AI Code Review、Production Debugging

這是 Senior Software Engineer 最應準備的基礎領域之一，也與 AI Coding 密切相關。Microsoft 的官方面試指南明確要求候選人展示 Code Quality、Testing、Boundary Conditions 及 Error Handling。

Microsoft Careers

### B1. Automated Testing & Software Quality（1–10）

1. ★ How would you design a comprehensive testing strategy for a large software system?  
如何規劃 Unit、Integration、E2E、Regression 和 Performance Tests？

2. What is the difference between unit testing, integration testing, and end-to-end testing?  
各適合驗證什麼？如何決定 Testing Pyramid？

3. ★ How would you test a Python/C++ application that integrates computer vision models and hardware devices?  
如何設計 Mock Camera、Simulated Hardware、Model Tests 和 Hardware-in-the-loop Testing？

4. How do you determine what should be mocked versus tested with real dependencies?  
Mock 太多可能帶來什麼問題？

5. ★ A test is intermittently failing in CI but always passes locally. How would you debug it?  
如何排查 Race Conditions、Timing、Environment Differences、Shared State？

6. How do you validate AI-generated unit tests?  
如何避免 Agent 產生只會重複 Implementation Logic 的無效測試？

7. How would you detect missing edge cases in an existing test suite?  
如何運用 Boundary Analysis、Mutation Testing、Property-based Testing？

8. What is property-based testing, and when would you use Hypothesis?  
如何自動產生 Inputs，驗證 Invariants？

9. ★ Why is 100% test coverage not sufficient to guarantee software quality?  
Coverage、Assertions、Mutation Score、Risk-based Testing 有何不同？

10. How do you build regression tests for a legacy codebase without good documentation?  
如何建立 Characterization Tests 和 Golden Outputs？

### B2. Code Review & Quality Gates（11–20）

11. ★ Design a multi-layer automated code review pipeline for a large engineering team.  
如何整合 Formatting、Linting、Type Checking、Testing、Security、AI Review、Human Review？

12. What are the most important things you look for when reviewing a Pull Request?  
Correctness、Maintainability、Security、Performance、Testability 如何排序？

13. What is the difference between static analysis, AI code review, and human code review?  
各自能發現哪些問題？有哪些限制？

14. ★ Your AI code reviewer generates too many false positives. How would you improve it?  
如何設計 Review Rules、Severity Thresholds 和評估指標？

15. ★ How would you identify race conditions or deadlocks in a concurrent Python/C++ program?  
如何分析 Threads、Locks、Shared State、Synchronization？

16. How would you review a code change that improves speed but increases memory usage?  
如何判斷 Performance Tradeoffs？

17. What are CODEOWNERS, required reviewers, and branch protection rules?  
如何用它們管理 Code Ownership？

18. How do you decide which code review findings should block a PR?  
如何分級 Critical、Major、Minor、Informational Findings？

19. How would you detect security vulnerabilities introduced by AI-generated code?  
如何檢查 Input Validation、Injection、Secrets、Unsafe Dependencies？

20. How would you review a database schema migration without risking production data loss?  
如何設計 Backward-compatible Migrations？

### B3. Production Debugging & Reliability（21–30）

21. ★ A production service suddenly starts failing after a deployment. How do you investigate?  
如何結合 Logs、Metrics、Traces、Recent Changes、Rollback？

22. All automated tests pass, but users report incorrect results in production. What would you do?  
如何分析 Missing Test Scenarios、Environment Drift、Incorrect Assumptions？

23. ★ How would you profile a slow Python application and identify its performance bottlenecks?  
CPU、Memory、I/O、GPU、Lock Contention 如何分析？

24. How would you investigate a memory leak in a long-running C++ or Python service?  
如何使用 Profilers、Allocation Tracking、Resource Lifecycle Analysis？

25. How would you debug an intermittent deadlock in a multi-threaded application?  
如何使用 Thread Dumps、Lock Ordering 和 Reproduction Tests？

26. How do you make production bugs reproducible in a local development environment?  
如何保存 Inputs、Configuration、Version、Random Seeds？

27. How would you validate a bug fix without introducing new regressions?  
如何設計 Reproduction Test、Regression Suite、Canary Validation？

28. How do you use fault injection to test system resilience?  
如何模擬 Timeouts、Network Failures、Unavailable Dependencies？

29. What metrics would you track to evaluate code quality across a large team?  
Change Failure Rate、Escaped Defects、Review Latency、Flaky Tests、Maintainability 如何衡量？

30. ★ A bug only occurs when multiple cameras capture images concurrently. How would you debug it?  
如何檢查 Buffer Ownership、Memory Lifetime、Thread Safety、Synchronization、Backpressure？

對你特別重要：B3、B5、B11、B15、B23、B30。這些能直接結合既有 Python/C++ 與多相機系統經驗。





#### 3  CI/CD、DevOps & Release Engineering 
```

```
# C. CI/CD、DevOps & Release Engineering

30 題｜GitHub Actions、Docker、AWS、Build Automation、Deployment、Monitoring、Rollback

這一類是 Senior Backend / Production Software Engineer 的重要基礎，也是你履歷已經寫到 CI/CD、MLOps、AWS Infrastructure 後，面試官可能深入追問的部分。

### C1. CI/CD Fundamentals（1–10）

1. ★ Design a complete CI/CD pipeline for a Python-based AI application.  
如何設計 Push → Test → Security → Build → Staging → Production？

2. What is the difference between Continuous Integration, Continuous Delivery, and Continuous Deployment?  
三種模式如何區分？何時應採用 Manual Approval？

3. ★ How would you configure GitHub Actions to block a PR if tests or security checks fail?  
如何設計 Required Status Checks、Branch Protection、Workflow Permissions？

4. What Git branching strategy would you choose for a fast-moving engineering team?  
Trunk-based Development、GitFlow、Feature Branches 各有何優缺點？

5. How would you optimize a CI pipeline that takes 45 minutes to finish?  
如何使用 Caching、Parallel Jobs、Test Sharding、Incremental Builds？

6. ★ How do you ensure reproducible builds across developer machines, CI runners, and production servers?  
如何管理 Dependency Versions、Lockfiles、Containers、Immutable Artifacts？

7. What is software artifact provenance, and why does it matter?  
如何追蹤 Binary / Docker Image 是由哪個 Commit 和 Build 產生？

8. How do you test changes to CI/CD pipelines before applying them to production?  
如何測試 Workflow、Permissions、Secrets Access、Deployment Logic？

9. ★ How would you securely deploy to AWS from GitHub Actions without storing long-lived AWS credentials?  
如何使用 OIDC、IAM Roles、Short-lived Tokens、Least Privilege？

10. How do you handle flaky tests without allowing them to weaken CI quality gates?  
何時允許 Retry、Quarantine、Owner Assignment？

### C2. Docker、Infrastructure & Deployment（11–20）

11. How would you containerize a PyTorch inference service with Docker?  
如何處理 CUDA Runtime、Model Weights、Dependencies、Image Size？

12. What is the difference between Development, Staging, and Production environments?  
如何避免 Configuration Drift？

13. How would you manage cloud infrastructure using Terraform or another Infrastructure-as-Code tool?  
如何管理 State、Environment、Review 和 Rollback？

14. When would you choose AWS ECS, Kubernetes, or a simpler VM-based deployment?  
如何考慮 Complexity、Scaling、Cost、Team Expertise？

15. How do you maintain consistent configuration across multiple deployment environments?  
如何管理 Secrets、Environment Variables、Feature Flags？

16. ★ Compare rolling deployment, blue-green deployment, and canary deployment.  
各適合什麼系統？如何設定 Traffic Shifting 與 Rollback？

17. How do you safely deploy a backward-incompatible database schema change?  
如何執行 Expand-and-Contract Migration？

18. What does an idempotent deployment process mean? Why is it important?  
如何確保重複執行 Deployment 不會造成損壞？

19. What would you do if cloud credentials were accidentally exposed in CI logs?  
如何執行 Revocation、Rotation、Impact Assessment、Incident Review？

20. ★ A deployment succeeds, but error rates increase five minutes later. What should the pipeline do?  
如何結合 Post-deployment Monitoring、Automated Rollback、Release Gates？

### C3. Advanced Release Engineering（21–30）

21. ★ How would you design an automated rollback mechanism for a production AI service?  
如何保留 Previous Artifact、Configuration、Model Version？

22. What monitoring and alerting should be integrated into a CD pipeline?  
如何監控 Error Rate、Latency、Resource Usage、Business Metrics？

23. What is GitOps, and how does it differ from traditional CI/CD?  
Git 作為 Desired State 的優缺點是什麼？

24. How do you implement semantic versioning and release artifact management?  
如何管理 Releases、Tags、Compatibility、Dependencies？

25. How would you use feature flags to safely release a new AI feature?  
如何分階段啟用、灰度測試和緊急關閉？

26. How would you design CI/CD for a large monorepo containing multiple services and ML models?  
如何避免每次 PR 都重新 Build / Test 全部模組？

27. ★ How would you deploy and update software across 100 edge AI inspection machines?  
如何設計 Device Identity、Versioning、Canary Fleet、Rollback、Offline Recovery？

28. How do you update software on machines that may temporarily lose network connectivity?  
如何設計 Queued Updates、Version Checks、Transactional Installation？

29. ★ How would you measure whether a CI/CD system is actually improving engineering performance?  
如何衡量 Deployment Frequency、Lead Time、Change Failure Rate、Recovery Time？

30. ★ Design a complete CI/CD and MLOps pipeline for an automated computer vision inspection system.  
包含 Python Training、ONNX/TensorRT、C++ Integration、Model Validation、Cloud / Edge Deployment。

對你特別重要：C1、C6、C9、C16、C21、C27、C30。尤其 C27、C30 可以成為你的 Senior / Staff System Design 代表題。


#### 4  DevSecOps & AI Coding Security   
```

```
# D. DevSecOps、Software Security & AI Agent Safety

30 題｜Secure SDLC、Prompt Injection、Secrets、Supply Chain、Sandbox、Agent Permissions

這是近年 AI Coding 普及後特別值得注意的領域。傳統 Software Security 仍是基礎，但讓 Agent 自動讀取 Repository、執行程式或呼叫 Cloud Tools，會產生額外的信任邊界與權限風險。

### D1. AI Coding Security（1–10）

1. ★ What are the security risks of allowing an AI coding agent to access a private source code repository?  
如何避免 Sensitive Code、Secrets、Customer Data 外洩？

2. ★ What is prompt injection, and how can it affect an AI coding agent?  
如果 README、Issue、PR Comment 包含惡意指示，Agent 應如何處理？

3. How do you prevent secrets from leaking through AI prompts, logs, and tool outputs?  
如何設計 Redaction、Secret Scanning、Data Access Controls？

4. ★ How would you implement least-privilege access for an autonomous coding agent?  
如何限制 Filesystem、Network、Shell Commands、GitHub 和 Cloud Permissions？

5. How would you sandbox code execution performed by an AI agent?  
Containers、Filesystem Isolation、Resource Limits、Network Egress Controls 如何選擇？

6. How do you safely let an AI agent read external documentation or web pages?  
如何區分 Trusted Instructions 與 Untrusted Content？

7. Can repository-level instructions fully protect an AI agent from malicious code or prompts?  
為何 Prompt Rules 不能取代 Runtime Security Controls？

8. How would you secure a coding agent that uses OAuth-authenticated external tools?  
如何管理 Scopes、Delegated Permissions、Tokens？

9. ★ An AI coding agent reads an error log containing a malicious command. How do you prevent it from executing that command?  
如何處理 Tool-output Injection 與 Instruction Hierarchy？

10. What should be enforced through prompts versus deterministic security policies?  
哪些約束必須由 Sandbox、Permissions、Approval Gates 強制執行？

### D2. DevSecOps & Supply Chain Security（11–20）

11. What is the difference between SAST, DAST, and software composition analysis (SCA)?  
各適合偵測哪些風險？

12. How would you integrate CodeQL into a GitHub Actions CI pipeline?  
如何處理 Findings、Severity、False Positives？

13. What is a Software Bill of Materials (SBOM), and why would you generate one?  
如何支援 Dependency Traceability 和 Vulnerability Response？

14. How do you prevent vulnerable or malicious dependencies from entering production builds?  
如何使用 Version Pinning、Dependency Audits、Approved Registries？

15. ★ How would you secure a GitHub Actions workflow that has access to production deployment credentials?  
如何避免 Excessive Permissions、Unsafe Triggers 和 Untrusted Code Execution？

16. What are the security risks of running CI workflows for pull requests from external contributors?  
為什麼 Fork-based PR 需要特別設計隔離與權限？

17. Why is OIDC-based authentication often preferable to long-lived cloud credentials in CI/CD?  
如何設計 Short-lived Credentials 和 Trust Policies？

18. How would you investigate a compromised open-source dependency used in your AI application?  
如何定位受影響版本、停止發布、替換套件？

19. ★ What is software supply chain security, and how do you verify build artifact provenance?  
如何驗證 Build、Dependency、Artifact、Release 的完整來源？

20. How would you mitigate a malicious package introduced by an AI-generated code change?  
如何建立 Dependency Allowlist、Review、Scan、Build Isolation？

### D3. Production Security & Governance（21–30）

21. ★ An AI agent accidentally exposes a production API key. What is your incident response plan?  
如何處理 Revocation、Rotation、Investigation、Audit、Prevention？

22. How would you investigate whether sensitive source code was transmitted to an external LLM service?  
如何分析 Network Logs、Tool Traces、Data Retention Policies？

23. What audit information should be captured when coding agents make repository changes?  
如何記錄 Actor、Prompt/Task Reference、Tool Calls、Diff、Approval？

24. How do you implement privacy and data retention policies for enterprise coding agents?  
如何限制 Data Collection、Retention、Access、Deletion？

25. How do you review AI-generated code for insecure authentication and authorization logic?  
如何驗證 Server-side Authorization，而不只相信 AI 的建議？

26. What are the risks of using a container alone as the security boundary for untrusted agent execution?  
何時需要更強的 Isolation、Network Policy 或 VM Sandbox？

27. How would you handle a security vulnerability discovered immediately before a scheduled release?  
如何進行 Risk Assessment、Release Block、Exception Approval？

28. ★ How would you design a risk-based human approval system for AI coding agents?  
哪些操作可以自動執行？哪些需要人工核准？

29. ★ How do you test whether an AI agent can be manipulated into performing unauthorized actions?  
如何設計 Adversarial Tests、Prompt Injection Cases、Tool Authorization Tests？

30. ★ Design a secure enterprise AI-assisted software development platform.  
如何整合 SSO、RBAC、Sandbox、Audit Logging、Code Review、CI/CD、Secrets Management？

對 Senior / Staff，建議能夠畫出完整 Trust Boundaries，而不是只回答「使用 Secret Manager」或「加入 Security Scanner」。


#### 5  Agent Architecture、Tool Calling & Evals 
```

```
# E. AI Agent Architecture、Tool Calling & Agent Evaluation

30 題｜Agent Design、MCP、Workflow Orchestration、Evaluation、Reliability、Cost

這一類尤其針對 Senior Applied AI Engineer、Senior AI Agent Engineer，以及 Codex / Copilot 類型的產品團隊。

OpenAI 的 Applied AI Engineer — Codex Core Agent 職缺明確強調 Real-world Coding Tasks、Agent Evals、Tool-use Strategies、Context Construction、Failure Analysis 及 Production Robustness，因此這些問題對該類職位具有直接關聯。

OpenAI

### E1. Agent Architecture Fundamentals（1–10）

1. ★ What is the difference between a traditional LLM application, an automated workflow, and an autonomous AI agent?  
如何依自主程度、工具使用及執行迴圈區分？

2. Explain the complete lifecycle of an AI agent using function calling.  
LLM Decision → Tool Invocation → Tool Result → State Update → Next Action 如何運作？

3. When should an AI agent use RAG versus external tools versus model knowledge?  
如何決定 Retrieval、Tool Calling 和 Direct Generation？

4. What is MCP, and how does it differ from ordinary function calling?  
MCP Server、Tools、Resources、Prompts 在系統中扮演什麼角色？

5. ★ How would you design an agent capable of completing complex, long-horizon software engineering tasks?  
如何管理 Planning、Execution、Checkpoints、Verification、Recovery？

6. What is the role of planning in an agentic AI system?  
何時採用固定 Workflow、Dynamic Planning、Replanning？

7. How would you manage agent state, short-term context, and long-term memory?  
如何避免 Stale Context、Context Pollution、Unbounded Memory？

8. How do you design tool schemas so an LLM can call APIs reliably?  
如何設計 JSON Schema、Validation、Error Responses、Idempotency？

9. How would you handle tool call failures, timeouts, and invalid responses?  
Retry、Fallback、Circuit Breaker、Escalation 如何設計？

10. How do you prevent duplicate side effects when an agent retries a tool call?  
如何使用 Idempotency Keys、Transaction Boundaries、Execution IDs？

### E2. Agent Evaluation & Quality（11–20）

11. ★ How would you build a representative golden test dataset for a coding agent?  
如何收集真實任務、失敗案例、Edge Cases、難度分布？

12. How do you avoid data leakage between agent development and evaluation datasets?  
如何避免 Agent 記住 Benchmark 或依據答案調整行為？

13. What are the weaknesses of using LLM-as-a-Judge to evaluate agent outputs?  
如何控制 Bias、Judge Inconsistency、Position Effects？

14. ★ How would you implement human-in-the-loop approval for high-risk agent actions?  
如何設定 Approval Gates、Escalation、Permission Boundaries？

15. How do you evaluate whether an agent selected and executed the correct tools?  
如何驗證 Tool Selection、Arguments、Execution Sequence？

16. ★ How do you measure the task success rate of a coding agent?  
如何定義 Verified Completion、Partial Success、Failure、Regression？

17. ★ An agent fails a task. How do you determine whether the failure came from the model, context retrieval, tools, or orchestration?  
如何使用 Traces、Failure Taxonomy、Ablation Tests 找出 Root Cause？

18. What is the difference between offline evaluation and online evaluation for agents?  
何時使用 Golden Dataset、Shadow Testing、Production Metrics？

19. What information should be recorded in agent execution traces?  
如何記錄 Plans、Tool Calls、Latency、Cost、Errors，同時保護敏感資料？

20. How would you prevent an agent from repeatedly performing the same unsuccessful actions?  
如何設計 Loop Detection、Retry Budgets、State Transitions？

### E3. Production Agent Systems（21–30）

21. ★ Design a production AI agent service that supports 10,000 concurrent users.  
如何設計 Stateless API、State Store、Queues、Workers、Model Routing、Rate Limits？

22. How would you implement rate limiting and backpressure for tool-calling agents?  
如何避免外部 API 被過量呼叫？

23. How do you reduce agent latency without sacrificing task success?  
如何處理 Sequential Tools、Parallel Calls、Context Size、Model Choice？

24. How would you optimize the cost of an agent that performs multiple LLM calls per task?  
如何比較 Routing、Caching、Token Budget、Quality Tradeoffs？

25. How do you decide what information should be included in an agent's context window?  
如何處理 Relevant Retrieval、Summarization、Context Ranking？

26. How would you make an AI agent recoverable after its execution environment crashes?  
如何儲存 Durable State、Checkpoints、Tool Results、Resume Logic？

27. ★ How would you evaluate an AI code review agent before enabling it on enterprise pull requests?  
如何衡量 Precision、Recall、False Positives、Review Latency、Security Risks？

28. How would you run A/B testing for two different agent prompts or execution strategies?  
如何建立 Baseline、Randomization、Success Metrics？

29. ★ How do you balance task success rate, latency, token cost, and reliability when optimizing an AI agent?  
如何定義 SLO、Cost Budget、Evaluation Thresholds？

30. ★ Design an end-to-end agent that can read GitHub issues, modify code, execute tests, and create pull requests safely.  
請說明 Architecture、Tools、Permissions、State、Evaluation、Failure Recovery。

對你特別重要：E5、E11、E16、E17、E21、E27、E30。這些也是我先前建議你建立 PRGuard / Coding Agent Side Project 的核心理由。



#### 6  MLOps、CV Production & AI Deployment   
```

```
# F. MLOps、Computer Vision Production & AI Deployment

30 題｜PyTorch、ONNX、TensorRT、Model CI/CD、GPU、Multi-camera、Edge AI

這是對你最有優勢、也最值得深入準備的類別。

你已有 Production C++、Computer Vision、GPU Inference、Model Compression、AWS、Multi-camera Imaging 的經驗。因此，面試官可能不只問 AI Model，而是深入追問 Model Validation、Reproducibility、Performance、Deployment、Monitoring 和 Failure Recovery。

### F1. Model CI/CD & MLOps（1–10）

1. ★ Design an end-to-end MLOps pipeline from model training to production deployment.  
如何整合 Data Versioning、Training、Validation、Model Registry、CI/CD、Monitoring？

2. What is the difference between traditional software CI/CD and machine learning CI/CD?  
為什麼 ML Pipeline 必須額外處理 Dataset、Model Weights、Accuracy Regression？

3. ★ How would you validate that an ONNX or TensorRT model produces outputs consistent with the original PyTorch model?  
如何比較 Numerical Differences、Output Tolerance、Accuracy、Performance？

4. How would you evaluate the impact of INT8 quantization before deploying a model?  
如何分析 Calibration、Accuracy Loss、Latency、Memory Footprint？

5. ★ What quality gates would you implement before promoting a new computer vision model to production?  
如何設定 Precision、Recall、mAP、IoU、Latency、Memory、Regression Thresholds？

6. How do you version datasets, model weights, training configurations, and inference code?  
如何確保 Model Lineage？

7. How would you use MLflow or a model registry in a production ML workflow?  
如何追蹤 Experiments、Artifacts、Model Versions、Deployment Decisions？

8. How do you ensure reproducible machine learning experiments?  
如何控制 Random Seeds、Environment、Dataset Splits、Dependencies？

9. How would you automate model retraining without accidentally deploying a worse model?  
如何設計 Evaluation Gates、Baseline Comparison、Approval？

10. How do you test non-deterministic GPU inference and training pipelines?  
如何使用 Statistical Tolerance、Reproducibility Controls、Multiple Runs？

### F2. Production CV Reliability & Monitoring（11–20）

11. ★ How would you detect data drift and concept drift in a production computer vision system?  
如何區分 Lighting、Camera、Object Distribution、Label Relationship Changes？

12. How do you monitor model quality when production ground-truth labels are not immediately available?  
如何使用 Proxy Metrics、Sampling、Manual Review、Delayed Labels？

13. How would you design an evaluation strategy for a defect detection system where false negatives are very expensive?  
如何決定 Recall、Precision、Threshold、Cost-sensitive Evaluation？

14. ★ Your model performs well offline but poorly in production. How do you investigate?  
如何檢查 Data Leakage、Preprocessing Mismatch、Domain Shift、Model Version？

15. ★ Design a reliable GPU inference pipeline for multiple high-resolution cameras capturing images concurrently.  
如何處理 Acquisition、Queues、Batching、GPU Scheduling、Synchronization？

16. What happens if one camera fails during a synchronized acquisition cycle?  
如何處理 Timeout、Partial Results、Retry、Recovery？

17. How would you detect and recover from missing or corrupted images?  
如何使用 Integrity Checks、Sequence IDs、Capture Logs、Re-acquisition？

18. How would you automatically reject blurry, overexposed, or low-quality images before inference?  
如何運用 Focus Metrics、Histogram、Exposure Validation？

19. A production camera produces images with a different color distribution from the training dataset. How do you handle it?  
如何使用 Calibration、Normalization、Domain Adaptation？

20. How would you validate camera calibration and geometric registration across multiple devices?  
如何追蹤 Intrinsics、Extrinsics、Calibration Error、Drift？

### F3. GPU Inference & Edge Deployment（21–30）

21. When would you choose batch inference versus real-time inference for a computer vision application?  
如何比較 Latency、Throughput、Memory、Utilization？

22. ★ PyTorch inference takes 1 second per image. How would you reduce it to 100 milliseconds?  
如何使用 Profiling、ROI、Model Architecture、TensorRT、Quantization、Parallelism？

23. ★ How would you prevent an image processing pipeline from running out of memory when input traffic increases?  
如何設計 Bounded Queues、Backpressure、Streaming、GPU Memory Management？

24. How do you manage TensorRT engine compatibility across GPU models, CUDA versions, and deployment environments?  
如何處理 Engine Build、Version Compatibility、Fallback？

25. How would you conduct a canary deployment of a new model across a fleet of edge AI devices?  
如何選擇 Canary Devices、Metrics、Promotion Criteria？

26. ★ How would you roll back a defective AI model deployed across 100 inspection machines?  
如何管理 Model Registry、Artifact Versions、Local Cache、Rollback？

27. How would you process thousands of high-resolution images per minute while keeping GPU costs under control?  
如何設計 Parallel Processing、Batching、Autoscaling、GPU Scheduling？

28. How would you reduce cold-start latency in an AI inference service?  
如何處理 Model Loading、Warm-up、Caching、Container Initialization？

29. What metrics would you monitor for a production computer vision inference system?  
如何追蹤 P95 Latency、Throughput、VRAM、Model Quality、Capture Errors？

30. ★ Design a production-grade visual inspection platform supporting 100 edge devices and continuous model improvement.  
請說明完整 Architecture：Camera → Inference → Storage → Evaluation → CI/CD → Edge Deployment → Monitoring。

這類題目你應該有能力比一般 AI Engineer 回答得更深入。F3、F5、F15、F22、F26、F30 特別值得準備成完整的 System Design 回答。



#### 7  System Design、Reliability & Leadership 
```

```
# G. Senior / Staff System Design、Engineering Leadership & Behavioral

30 題｜Architecture、Scalability、Technical Decision、Production Ownership、Cross-team Leadership

這個類別是 Senior 與 Staff 的重要區別。

在 Microsoft 的官方技術面試說明中，System Design、Coding、Testing、Technical Decision-making 和處理複雜問題的能力都屬於重要評估範圍。

Microsoft Careers

### G1. System Design & Architecture（1–10）

1. ★ Design a scalable distributed computer vision inspection platform from scratch.  
如何進行 Requirements、Architecture、Scalability、Fault Tolerance、Deployment 設計？

2. ★ Design an enterprise AI assistant that can search internal documents, call business tools, and enforce access permissions.  
如何設計 RAG、Agent Orchestration、ACL、Tool Security、Evals、Monitoring？

3. How do you translate ambiguous product requirements into technical requirements and service-level objectives?  
如何確立 Functional / Non-functional Requirements 和 SLO？

4. How would you decide whether to refactor an existing system or rebuild it from scratch?  
如何評估 Technical Debt、Migration Cost、Risk？

5. How would you design a fault-tolerant system that coordinates cameras, sensors, motion control, and cloud services?  
如何管理 Device State、Timeout、Retry、Recovery？

6. How do you guarantee idempotent processing in an asynchronous distributed workflow?  
如何使用 Request IDs、Deduplication、Transactions？

7. How would you scale an AI inference system from one server to hundreds of workers?  
如何處理 Queues、Autoscaling、Model Loading、Distributed Scheduling？

8. How would you design storage for millions of high-resolution images and inspection results?  
如何選擇 Object Storage、Metadata Database、Retention、Indexing？

9. How would you design a system that supports 10,000 simultaneous inference requests?  
如何估算 Load、Concurrency、Latency、Capacity、Cost？

10. How would you reduce the P95 latency of a distributed AI application?  
如何用 Tracing 分析 Critical Path、Network、Database、Model Serving？

### G2. Engineering Leadership（11–20）

11. ★ Tell me about a project where you led the architecture and implementation of a complex system.  
如何介紹自己對 Architecture、Design Decisions 和 Delivery 的貢獻？

12. ★ How do you lead a project when requirements are ambiguous and frequently changing?  
如何定義 Milestones、Prototype、Risk Reduction、Decision Records？

13. How do you resolve technical disagreements between senior engineers?  
如何用 Experiments、Evidence、Tradeoff Analysis 推動決策？

14. How do you manage technical debt while delivering features under tight deadlines?  
如何建立 Prioritization 和 Incremental Refactoring Plan？

15. How do you mentor engineers who are less experienced with AI or production software development?  
如何建立 Coding Standards、Review Culture、Knowledge Sharing？

16. ★ Describe a major production incident you owned. How did you investigate and prevent recurrence?  
如何說明 Incident Detection、Mitigation、Root Cause、Postmortem、Follow-up？

17. What does end-to-end ownership mean for a Senior or Staff Engineer?  
如何界定 Design、Implementation、Quality、Deployment、Monitoring、Maintenance？

18. Tell me about a time you disagreed with your manager's technical decision.  
如何說明你使用數據、實驗與溝通達成共識？

19. How do you coordinate a major release involving software, firmware, hardware, and cloud teams?  
如何處理 Dependencies、Integration Milestones、Release Readiness？

20. How do you balance delivery deadlines, code quality, reliability, and long-term maintainability?  
如何進行 Risk-based Decision-making？

### G3. AI-Native Engineering Leadership（21–30）

21. ★ How would you establish code review standards for a team of 20 engineers?  
如何規範 Review Ownership、Quality Gates、PR Size、SLA、Escalation？

22. ★ Your engineering organization wants to adopt AI coding agents. How would you introduce them safely?  
如何設計 Pilot、Tool Selection、Security Policy、Training、Evaluation？

23. How would you measure the ROI of AI coding tools across an engineering organization?  
如何比較 Productivity、Rework、Review Cost、Defect Rate、Delivery Lead Time？

24. How would you establish organization-wide guidelines for AI-generated code and automated CI/CD?  
如何定義 Human Accountability、Approval、Testing、Audit、Release Policies？

25. Tell me about a production failure caused by a design decision. What would you change?  
如何進行 Blameless Postmortem、Architecture Improvement？

26. How do you make build-versus-buy decisions for AI infrastructure and developer tooling?  
如何比較 Cost、Vendor Lock-in、Maintainability、Security、Time to Market？

27. How would you plan a gradual migration from a legacy C++ application to a Python-based AI service architecture?  
如何確保 Compatibility、Testing、Performance、Incremental Delivery？

28. ★ A new AI feature could improve revenue but creates security and reliability risks. Would you ship it?  
如何使用 Risk Assessment、Feature Flags、Limited Rollout、Approval？

29. ★ How would you lead a multi-team rollout of a new AI model to 100 production devices?  
如何協調 Model Team、Hardware Team、QA、Cloud Infrastructure、Operations？

30. ★ Walk me through the most technically challenging project you have led, including architecture, tradeoffs, failures, and measurable results.  
如何完整回答 Project Ownership、Design Rationale、Execution、Impact？

這個類別非常適合利用你最新履歷中的 Lead Software Engineer，以及先前 Sartorius Production AI 系統開發經驗準備。




#### 最建議你先精通的 15 題
```
最建議你先精通的 15 題
```
### 我最建議你先精通的 15 題

這 15 題能把你已經熟悉的 Production Computer Vision，與新的 AI Coding / Engineering Automation 經驗連接起來。

Interview Preparation Checklist

0/15 prepared

A11. Design an end-to-end AI-assisted development workflow.

A23. Design a coding agent that can test, debug and propose patches.

A25. Evaluate two coding agents using a fair benchmark.

B11. Design a multi-layer automated code review pipeline.

B21. Debug a production failure after deployment.

B30. Debug concurrency failures in multi-camera acquisition.

C1. Design a complete CI/CD pipeline for an AI application.

C21. Design automated rollback for production AI.

C30. Design CI/CD + MLOps for a visual inspection system.

D2. Prevent prompt injection in a coding agent.

E17. Identify whether agent failures originate from the model, tools or orchestration.

E27. Evaluate an AI PR review agent.

F3. Validate PyTorch → ONNX → TensorRT correctness.

F30. Design an AI inspection platform across 100 edge devices.

G22. Lead the adoption of AI coding agents in an engineering team.

Copy checklist

## 四、Senior / Staff 的回答應該達到什麼深度？

面試官通常不會只問你「什麼是 CI/CD？」就結束，而是會逐步追問。

例如：

> Design a CI/CD pipeline for an AI-powered computer vision system deployed across 100 edge devices.

完整回答應該包含：

1. Requirements：設備數量、GPU Hardware、Model Size、Deployment Frequency、Allowed Downtime。
    
2. Architecture：GitHub Actions、Model Registry、Artifact Storage、Deployment Controller、Edge Agent。
    
3. Quality Gates：Automated Tests、Model Accuracy、Numerical Parity、Performance Regression、Security Scan。
    
4. Deployment Strategy：Canary Rollout、Device Groups、Release Versioning、Rollback。
    
5. Failure Handling：Network Disconnection、Corrupted Artifact、Model Incompatibility、Unhealthy Device。
    
6. Observability：Deployment Success Rate、Model Accuracy、P95 Latency、GPU Memory、Device Health。
    
7. Tradeoffs：Release Speed、Reliability、Cloud Cost、Implementation Complexity。
    
8. Experience：你以前實際做過什麼，哪些是設計建議，哪些已在 Production 驗證。
    

這種回答方式會比單純解釋 GitHub Actions 的 YAML 語法更有 Senior / Staff 水準。

另外，2026 年仍然需要準備 LeetCode / Data Structures & Algorithms。Interview Query 的資料指出，2026 年第三季約 30.5% 的技術面試回報仍提到 LeetCode 類題目；明確描述 AI-assisted Coding 面試輪次的比例則約 1.1%。這些是特定平台的面試回報比例，不宜當作所有公司的統一出題率。

Interview Query

## 五、實際面試前的準備標準

對每一道重要題目，我建議你至少能做到：

- 2 分鐘：清楚說明核心概念及何時使用。
    
- 5 分鐘：解釋 Architecture、Key Components 與 Tradeoffs。
    
- 15–20 分鐘：完整設計系統，回答 Failure Modes、Performance、Security、Evaluation。
    
- 實作證據：能引用你自己的 Repository、PR、CI Logs、Benchmarks 或實際工作經驗。
    

針對你目前履歷，我建議準備兩個主要的 Project Deep Dive：

第一個：Production Computer Vision / Automated Inspection Platform。 用來回答 Multi-camera Architecture、C++ / Python Integration、GPU Optimization、Reliability、MLOps、Cross-team Engineering。

第二個：AI-Native Software Engineering / Coding Agent Project。 用來回答 Agentic Development、AI Code Review、GitHub Actions、Automated Tests、DevSecOps、Agent Evaluation。

這兩個專案互相補強，能讓你在 2026 年的面試中同時展示傳統 Software Engineering 深度，以及 AI Coding / AI Agent 帶來的新能力。

最重要的是：不要為了學 AI Coding 而忽略原本的 Software Engineering 基礎。2026 年的 Senior / Staff 面試，更值得準備的是如何利用 AI 提升交付效率，同時證明你能對程式碼品質、架構、部署及 Production 結果負責。