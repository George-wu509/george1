![[836843089_122123262537470692_5316901524622327562_n.jpg]]

|                         |     |
| ----------------------- | --- |
| [[#### Harness design]] |     |
|                         |     |
|                         |     |
|                         |     |


#### Harness design
```
Harness design
```
最近在準備 AI Engineering 的課程，正好藉機研究了一下各家的Harness設計的不同

坦白說三家的設計大同小異

Context都會塞滿所有歷史訊息、所有可以使用的tool的簡介、system prompt等等

而執行都是繞著一個agent loop來運作：

送出訊息 -> 模型回話做決策 -> 跑 tool -> 繼續或結束

唯一比較不同的反而比較像設計理念：

Pi 除了是完全開源以外，他主張minimalism

設計哲學比較像是模型已經夠強，harness 應該盡量退場

官方文件也特別建議「用最弱但夠用的那層」

Claude Code 和 Codex 賭的則是模型要在自家客製化的 harness 裡訓練才會表現最好

而且每個model都會有自己專屬的harness

Claude Code 和 Codex 的小差異還有 context 給誰管

Codex 開始漸漸把這件事交給模型自己

包括給模型 new_context、get_context_remaining

model還有自己的 notes 和 history 工具

讓它把東西寫到外面之後再撈回來

Claude Code 的 context 管理大多是 harness + 人類替模型做

先清掉舊的工具輸出再做摘要

事先定義好超過約 3 萬字元的輸出存成檔案，只給model預覽

使用者則可以大量透過 CLAUDE.md 或 /compact 的指示去引導

至於哪種設計比較對呢？

在 NeoSigma 這篇文章

investigating-the-optimal-harness-for-a-model 做的Model-Harness Fit顯示

Codex & Claude code 這種跟 model 升級一起升級的harness在結果上的確是比 Pi 無為而治的理念優秀

我覺得看到這個結果是真的蠻有趣的

所以每次model升級，如果沒有搭配適合它的 harness ，它其實也沒辦法發揮出最大的實力

就跟教養一樣，小孩在不同階段雖然變得更聰明

但還是需要完全不同的調教方式才能讓小孩有最大的發揮

從harness突然寫到媽媽經是正常的嗎？

![[836843089_122123262537470692_5316901524622327562_n 1.jpg]]