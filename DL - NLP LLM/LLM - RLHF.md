
LLM训练一般包括预训练、指令微调和偏好对齐三步。偏好对齐旨在让模型的输出风格更接近人类用户的偏好。ChatGPT使用人类反馈强化学习(RLHF)进行偏好对齐。RLHF包括三步：

- 1）通过监督微调得到基础模型(base model)，
- 2）给定问题，让基础模型多次输出回答，人工基于偏好标准对这些回答排序。最后用排序的回答，通过ranking loss训练奖励模型(reward model)。
- 3）使用奖励模型通过近端策略优化算法(PPO)微调基础模型。

![](https://pic3.zhimg.com/v2-5d7e16c515ec01885561474ab3582aba_1440w.jpg)

Reference:
[llm-from-scratch] 8. DPO原理和代码实现 - 十点雨的文章 - 知乎
https://zhuanlan.zhihu.com/p/1949597631161562522