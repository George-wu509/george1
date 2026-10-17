
|                                |     |
| ------------------------------ | --- |
| [[#### Anthropic Code Review]] |     |
|                                |     |
|                                |     |

#### Anthropic Code Review
```
Anthropic Code Review
```
當 AI 寫了 90% 的程式碼，Anthropic 團隊 Code Review 怎麼做？

先說我的觀點：

一、這篇讓我最有共鳴的，是 Anthropic 用 Revert Rate / Incident Rate 作為品質護欄。不追求零錯誤，而是重視快速發現、快速復原，並透過數據驗證流程調整是否真的有效。

二、「You land it, you own it」的文化，搭配 Revert、On-call Rotation、Incident Review 等機制，才讓降低人工 Review 比重這件事變得可控。

以下整理幾個我覺得值得參考的重點。完整的七層審查架構放在圖中，原文連結放在第一則留言。
Anthropic 的 monorepo，每天合併的程式碼變更，從 5 月約 2,500 個，成長到 9 月超過 30,000 個。


1. 重新定義 Review 的目標

與其問「有沒有人看過這段 Code」，不如問「這個變更可能出什麼錯？哪一道檢查最適合抓到它？」​他們追蹤每個 PR 的人工 Approve 數量、等待時間，同時以 [#RevertRate](https://www.facebook.com/hashtag/revertrate?__cft__[0]=AZiS-grtY58IS3mcgN-_0oJ1dRrbAseCE67u7n8fJXXT_H-qRIIuvgY5MKHk6kb3y7nYqgKfx0nBe6KWPfrGvqgqRLOrMVogrt6hAMxRzrnyfVxcggjPNGQlNumIH6B95u-wKWHW9IR3Zb4XgTVXSj_sSGPKJsLTyHyhtReYe7F9YxJM7dDbdD8QcK5qCpo_VqE&__tn__=*NK-R)、[#IncidentRate](https://www.facebook.com/hashtag/incidentrate?__cft__[0]=AZiS-grtY58IS3mcgN-_0oJ1dRrbAseCE67u7n8fJXXT_H-qRIIuvgY5MKHk6kb3y7nYqgKfx0nBe6KWPfrGvqgqRLOrMVogrt6hAMxRzrnyfVxcggjPNGQlNumIH6B95u-wKWHW9IR3Zb4XgTVXSj_sSGPKJsLTyHyhtReYe7F9YxJM7dDbdD8QcK5qCpo_VqE&__tn__=*NK-R) 作為品質護欄。結果從 6 月到 9 月，Claude Code 的 Merged PR 數量增加超過一倍，需要 Revert 的比例反而下降超過 40%。

2. Review 不再只是 PR 的最後關卡

許多檢查在 PR 建立之前就開始了。Claude 會先執行測試、Build、截圖比對，再由另一個使用全新 Context 的 Subagent 審查 Diff。讓問題盡可能在開發階段就被發現，而不是全部留到 PR 才處理。

3. 多層防護，而不是依賴單一關卡

[#CI](https://www.facebook.com/hashtag/ci?__cft__[0]=AZiS-grtY58IS3mcgN-_0oJ1dRrbAseCE67u7n8fJXXT_H-qRIIuvgY5MKHk6kb3y7nYqgKfx0nBe6KWPfrGvqgqRLOrMVogrt6hAMxRzrnyfVxcggjPNGQlNumIH6B95u-wKWHW9IR3Zb4XgTVXSj_sSGPKJsLTyHyhtReYe7F9YxJM7dDbdD8QcK5qCpo_VqE&__tn__=*NK-R)、AI Code Review、Security Review、人工審核、[#MergeQueue](https://www.facebook.com/hashtag/mergequeue?__cft__[0]=AZiS-grtY58IS3mcgN-_0oJ1dRrbAseCE67u7n8fJXXT_H-qRIIuvgY5MKHk6kb3y7nYqgKfx0nBe6KWPfrGvqgqRLOrMVogrt6hAMxRzrnyfVxcggjPNGQlNumIH6B95u-wKWHW9IR3Zb4XgTVXSj_sSGPKJsLTyHyhtReYe7F9YxJM7dDbdD8QcK5qCpo_VqE&__tn__=*NK-R)、[#FeatureFlag](https://www.facebook.com/hashtag/featureflag?__cft__[0]=AZiS-grtY58IS3mcgN-_0oJ1dRrbAseCE67u7n8fJXXT_H-qRIIuvgY5MKHk6kb3y7nYqgKfx0nBe6KWPfrGvqgqRLOrMVogrt6hAMxRzrnyfVxcggjPNGQlNumIH6B95u-wKWHW9IR3Zb4XgTVXSj_sSGPKJsLTyHyhtReYe7F9YxJM7dDbdD8QcK5qCpo_VqE&__tn__=*NK-R)，以及 [#Rollback](https://www.facebook.com/hashtag/rollback?__cft__[0]=AZiS-grtY58IS3mcgN-_0oJ1dRrbAseCE67u7n8fJXXT_H-qRIIuvgY5MKHk6kb3y7nYqgKfx0nBe6KWPfrGvqgqRLOrMVogrt6hAMxRzrnyfVxcggjPNGQlNumIH6B95u-wKWHW9IR3Zb4XgTVXSj_sSGPKJsLTyHyhtReYe7F9YxJM7dDbdD8QcK5qCpo_VqE&__tn__=*NK-R)，共同構成防護機制。每一層都可能漏掉問題，但透過不同機制交叉檢查，可以降低問題一路進入正式環境的機率。

4. AI Review 需要降噪
多個 Agent 平行找 Bug，再由另一組 Agent 交叉驗證，過濾掉不夠可靠的發現與純風格類意見。留下的 Review 意見都需要處理：修正問題，或說明理由後關閉。否則 AI Reviewer 產生大量低價值意見，反而會增加工程師的負擔。

5. 根據風險決定人工審查程度

不是每個 PR 都需要相同數量的 Reviewer。透過分類器依照變更內容判斷風險，低風險變更可以不需要額外人工 Approve，高風險變更則需要更多人審查。同時也有明確限制：AI 的 Approve 不能取代人的 Approve；新 Commit 進來後，原本的核准也會失效。

6. 避免 AI Review 無限循環

曾經有 PR 被 AI 作者與 AI Reviewer 來回修改超過 50 輪。後來透過限制 Review 輪數、只審查新的 Diff、避免重複提出相同問題，讓平均 Review 輪數從 2.08 降到 1.32。

7. 把 Incident 變成 Evaluation Dataset

將過去造成事故或需要 Revert 的 PR 整理成評測集，每天驗證 AI Reviewer 的能力。未來不論是調整 Prompt、工具或模型，都可以透過相同的評測標準，確認是否真的改善。

8. 人應該專注在更需要判斷力的事情

例如需求是否合理、架構方向、高風險變更、證據判讀，以及最終的責任歸屬。這些事情，不會因為 AI 能寫出更多程式碼就自動解決。

![[852661240_10234812014070512_5672273076250322684_n.jpg]]

