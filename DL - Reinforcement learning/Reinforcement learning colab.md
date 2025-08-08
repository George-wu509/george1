
Reinforcement learning (RL) 

**environment**: represents the problem to be solved
**agent**: represents the learning algorithm.

![[0_0pt2j6ygDDdcN5dl.webp]]




[1]
Introduction to RL and Deep Q Networks
https://colab.research.google.com/github/tensorflow/agents/blob/master/docs/tutorials/0_intro_rl.ipynb#scrollTo=b5tItHFpLyXG

https://www.kaggle.com/code/dsxavier/dqn-openai-gym-cartpole-with-pytorch

DQN深度强化学习：CartPole倒立摆任务（完整代码） - 林泽毅的文章 - 知乎
https://zhuanlan.zhihu.com/p/21975146686



Use python environment:  **dqn_env1**

使用`gymnasium`​库，启动cartpole环境非常容易，下面是一个简单的示例代码
```python
import gymnasium as gym

env = gym.make("CartPole-v1")
state = env.reset()
done = False

while not done:
    action = 0 if state[2] < 0 else 1  # 根据杆子角度简单决策
    next_state, reward, done, _ = env.step(action)
    state = next_state
    env.render()
```


首先你需要1个Python>=3.8的环境，然后安装下面的库：
swanlab,  gymnasium,  numpy,  torch,  pygame,  moviepy

```python
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install swanlab gymnasium numpy torch pygame moviepy
```

##### 定义QNet
DQN使用神经网络来近似QLearning中的Q表，这个神经网络被称为QNetwork。
QNetwork的输入是状态向量，输出是动作向量，这里用一个非常简单的神经网络：
```python
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.to(device)  # 将网络移到指定设备

    def forward(self, x):
        return self.fc(x)
```




```python
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.q_net = QNetwork(state_dim, action_dim)       # 当前网络
        self.target_net = QNetwork(state_dim, action_dim)  # 目标网络
        self.target_net.load_state_dict(self.q_net.state_dict())  # 将目标网络和当前网络初始化一致，避免网络不一致导致的训练波动
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.replay_buffer = deque(maxlen=10000)           # 经验回放缓冲区
        self.update_target_freq = 100
```


```python
def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 2)  # CartPole有2个动作（左/右）
        else:
            state_tensor = torch.FloatTensor(state).to(device)
            q_values = self.q_net(state_tensor)
            return q_values.cpu().detach().numpy().argmax()
```

![[Pasted image 20250805161844.png]]


完整code
```python
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import swanlab
import os

# 设置随机数种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 定义Q网络
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.q_net = QNetwork(state_dim, action_dim)       # 当前网络
        self.target_net = QNetwork(state_dim, action_dim)  # 目标网络
        self.target_net.load_state_dict(self.q_net.state_dict())  # 将目标网络和当前网络初始化一致，避免网络不一致导致的训练波动
        self.best_net = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.replay_buffer = deque(maxlen=10000)           # 经验回放缓冲区
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 0.1
        self.update_target_freq = 100  # 目标网络更新频率
        self.step_count = 0
        self.best_reward = 0
        self.best_avg_reward = 0
        self.eval_episodes = 5  # 评估时的episode数量

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 2)  # CartPole有2个动作（左/右）
        else:
            state_tensor = torch.FloatTensor(state)
            q_values = self.q_net(state_tensor)
            return q_values.cpu().detach().numpy().argmax()

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # 从缓冲区随机采样
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # 计算当前Q值
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # 计算目标Q值（使用目标网络）
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # 计算损失并更新网络
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 定期更新目标网络
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            # 使用深拷贝更新目标网络参数
            self.target_net.load_state_dict({
                k: v.clone() for k, v in self.q_net.state_dict().items()
            })

    def save_model(self, path="./output/best_model.pth"):
        if not os.path.exists("./output"):
            os.makedirs("./output")
        torch.save(self.q_net.state_dict(), path)
        print(f"Model saved to {path}")

    def evaluate(self, env):
        """评估当前模型的性能"""
        original_epsilon = self.epsilon
        self.epsilon = 0  # 关闭探索
        total_rewards = []

        for _ in range(self.eval_episodes):
            state = env.reset()[0]
            episode_reward = 0
            while True:
                action = self.choose_action(state)
                next_state, reward, done, _, _ = env.step(action)
                episode_reward += reward
                state = next_state
                if done or episode_reward > 2e4:
                    break
            total_rewards.append(episode_reward)

        self.epsilon = original_epsilon  # 恢复探索
        return np.mean(total_rewards)

# 训练过程
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim)


# 初始化SwanLab日志记录器
swanlab.init(
    project="RL-All-In-One",
    experiment_name="DQN-CartPole-v1",
    config={
        "state_dim": state_dim,
        "action_dim": action_dim,
        "batch_size": agent.batch_size,
        "gamma": agent.gamma,
        "epsilon": agent.epsilon,
        "update_target_freq": agent.update_target_freq,
        "replay_buffer_size": agent.replay_buffer.maxlen,
        "learning_rate": agent.optimizer.param_groups[0]['lr'],
        "episode": 600,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
    },
    description="增加了初始化目标网络和当前网络一致，避免网络不一致导致的训练波动"
)

# ========== 训练阶段 ==========

agent.epsilon = swanlab.config["epsilon_start"]

for episode in range(swanlab.config["episode"]):
    state = env.reset()[0]
    total_reward = 0

    while True:
        action = agent.choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.store_experience(state, action, reward, next_state, done)
        agent.train()

        total_reward += reward
        state = next_state
        if done or total_reward > 2e4:
            break

    # epsilon是探索系数，随着每一轮训练，epsilon 逐渐减小
    agent.epsilon = max(swanlab.config["epsilon_end"], agent.epsilon * swanlab.config["epsilon_decay"])  

    # 每10个episode评估一次模型
    if episode % 10 == 0:
        eval_env = gym.make('CartPole-v1')
        avg_reward = agent.evaluate(eval_env)
        eval_env.close()

        if avg_reward > agent.best_avg_reward:
            agent.best_avg_reward = avg_reward
            # 深拷贝当前最优模型的参数
            agent.best_net.load_state_dict({k: v.clone() for k, v in agent.q_net.state_dict().items()})
            agent.save_model(path=f"./output/best_model.pth")
            print(f"New best model saved with average reward: {avg_reward}")

    print(f"Episode: {episode}, Train Reward: {total_reward}, Best Eval Avg Reward: {agent.best_avg_reward}")

    swanlab.log(
        {
            "train/reward": total_reward,
            "eval/best_avg_reward": agent.best_avg_reward,
            "train/epsilon": agent.epsilon
        },
        step=episode,
    )

# 测试并录制视频
agent.epsilon = 0  # 关闭探索策略
test_env = gym.make('CartPole-v1', render_mode='rgb_array')
test_env = RecordVideo(test_env, "./dqn_videos", episode_trigger=lambda x: True)  # 保存所有测试回合
agent.q_net.load_state_dict(agent.best_net.state_dict())  # 使用最佳模型

for episode in range(3):  # 录制3个测试回合
    state = test_env.reset()[0]
    total_reward = 0
    steps = 0

    while True:
        action = agent.choose_action(state)
        next_state, reward, done, _, _ = test_env.step(action)
        total_reward += reward
        state = next_state
        steps += 1

        # 限制每个episode最多1500步,约30秒,防止录制时间过长
        if done or steps >= 1500:
            break

    print(f"Test Episode: {episode}, Reward: {total_reward}")

test_env.close()
```


分三部分來解釋：

1. **核心概念**：DQN如何將深度學習（神經網路）與強化學習結合，來解決像CartPole這樣的問題。
    
2. **網路詳解**：詳細說明 `q_net`, `target_net`, 和 `best_net` 這三個神經網路各自的角色、輸入和輸出。
    
3. **具體範例**：用一個完整的、一步步的例子，來走過整個「選擇動作 -> 學習 -> 更新」的流程。
    
---

### 1. 核心概念：用神經網路來「學會」直覺

在強化學習中，我們的目標是學習一個**策略 (Policy)**，這個策略能告訴智能體 (Agent) 在任何**狀態 (State)** 下，應該採取哪個**動作 (Action)** 才能獲得最大的長期獎勵。

- **傳統方法的困境**：對於簡單問題，我們可以用一個表格（稱為Q-Table）來記錄每一個「狀態-動作」組合的價值（Q-value）。但在CartPole中，狀態是連續的（推車位置、桿子角度等都是浮點數），這意味著狀態有無窮多種，Q-Table會變得無限大，根本無法實現。
    
- **深度學習的解決方案**：DQN (Deep Q-Network) 的天才之處在於，它不用表格去「死記硬背」，而是用一個**神經網路**去「學習和預測」。這個神經網路就像一個函數 Q(s,a)，我們希望它能學會：**只要給定一個狀態 s，它就能預測出在這個狀態下採取所有可能動作的Q值**。
    

在CartPole中，神經網路的任務就是學會一個「直覺」：看到桿子快要往左倒了（這是一個狀態），它就要預測出「向左推」這個動作的價值，會遠高於「向右推」的價值。

---

### 2. 網路詳解：三個網路，各司其職

在您的程式碼中，`DQNAgent` 初始化了三個看起來一樣的神經網路，但它們在訓練過程中的角色截然不同。

#### `q_net` (當前網路 / Online Network)

- **角色**：這是**主角**，是我們積極訓練和更新的網路。它代表了智能體當前對世界（CartPole環境）的理解。它所有的決策和學習都基於這個網路。
    
- **輸入**：一個**狀態 (state)**。在CartPole中，這是一個包含4個數字的一維張量（Tensor），例如 `[推車位置, 推車速度, 桿子角度, 桿子角速度]`。
    
- **輸出**：一個包含**每個動作對應Q值**的一維張量。在CartPole中，動作只有2個（向左推=0，向右推=1），所以輸出會是一個包含2個數字的張量，例如 `[15.2, 18.5]`。這代表 `q_net` 預測：在當前狀態下，向左推的Q值是15.2，向右推的Q值是18.5。
    
- **用途**：
    
    1. **選擇動作 (`choose_action`)**：用來決定要採取哪個動作。它會選擇輸出Q值最高的那個動作（這叫「利用, exploitation」）。
        
    2. **計算損失 (`train`)**：在訓練時，它計算出的Q值 (`current_q`) 會被用來和「目標Q值」比較，從而計算出損失 (loss)，進而透過反向傳播來更新自己的權重。
        

#### `target_net` (目標網路 / Target Network)

- **角色**：這是**穩定器**。它是DQN演算法穩定訓練的關鍵。如果我們用不斷變化的 `q_net` 來計算「目標Q值」，那就像追著一個移動的靶子射擊，訓練會非常不穩定。`target_net` 的存在就是為 `q_net` 提供一個**穩定、延遲更新的學習目標**。
    
- **輸入**：一批**下一個狀態 (next_states)**。
    
- **輸出**：這些 `next_states` 對應的Q值。
    
- **用途**：
    
    1. **計算目標Q值 (`train`)**：它的唯一用途就是在訓練時，計算出`next_states`的最大Q值。這個值被用來建構一個更可靠的學習目標 `target_q`。
        
- **更新方式**：它**不參與反向傳播**。它的權重是**定期**從 `q_net` **完整複製**過來的（每隔 `update_target_freq` 步）。這就像是每隔一段時間，我們給靶子拍一張快照，讓 `q_net` 在接下來的一段時間裡都朝著這張靜態的快照去學習。
    

#### `best_net` (最佳網路 / Best Network)

- **角色**：這是**榮譽榜**。它的作用是儲存整個訓練過程中**表現最好**的那個 `q_net` 的狀態。
    
- **輸入/輸出**：與 `q_net` 完全相同。
    
- **用途**：在您的程式碼片段中，`best_net` 被初始化了，但沒有看到更新和使用它的邏輯。在一個完整的實現中，通常會這樣使用：
    
    1. 訓練過程中，定期呼叫 `evaluate` 函數來評估當前 `q_net` 的性能（例如，玩5局遊戲的平均得分）。
        
    2. 如果這次的平均得分超過了歷史最高分 (`best_avg_reward`)，就把當前 `q_net` 的權重複製到 `best_net`，並更新最高分記錄。
        
    3. 訓練結束時，我們儲存到檔案的不是最後的 `q_net`（它可能因為最後幾次的學習而變差），而是儲存在 `best_net` 中的那個歷史最佳模型。
        

---

### 3. 具體範例：一步步走過DQN的學習循環

假設現在是訓練的某個時刻，智能體處於某個狀態。

**環境設定:**

- `state_dim` = 4
    
- `action_dim` = 2
    
- `gamma` = 0.99
    
- `batch_size` = 64
    

#### 第1步：選擇動作 (`choose_action`)

1. 智能體處於狀態 `s1 = [0.05, 0.2, -0.01, -0.15]`。
    
2. 這個 `s1` 被轉換成 `torch.FloatTensor` 後，輸入到 `q_net`。
    
3. `q_net` 進行前向傳播，輸出預測的Q值，例如 `q_values = [20.1, 22.5]`。
    
4. 假設此時 `epsilon` 很小，智能體不進行隨機探索。它會選擇Q值最大的動作：`argmax([20.1, 22.5])`，所以 `action` = 1 (向右推)。
    

#### 第2步：與環境互動並儲存經驗 (`store_experience`)

1. 智能體在環境中執行動作 `a1 = 1`。
    
2. 環境回饋：
    
    - `reward` = 1.0 (因為桿子還沒倒)
        
    - `next_state` `s2 = [0.07, 0.39, -0.013, -0.44]`
        
    - `done` = `False` (遊戲還沒結束)
        
3. 智能體將這個完整的經驗元組 `(s1, a1, 1.0, s2, False)` 存入 `replay_buffer`。
    

_...這個過程重複很多次，直到 `replay_buffer` 裡的經驗數量超過 `batch_size`..._

#### 第3步：訓練網路 (`train`)

1. **採樣**：從 `replay_buffer` 中隨機抽取 `batch_size` (64) 筆經驗。假設我們剛才存入的那筆經驗 `(s1, a1, 1.0, s2, False)` 被抽中了。
    
2. **計算 `current_q` (我當前的預測是多少？)**
    
    - 將這64個 `states` (包含 `s1`) 一起輸入 `q_net`。
        
    - 對於 `s1`，`q_net` 的輸出仍然是 `[20.1, 22.5]`。
        
    - 程式碼使用 `.gather(1, actions.unsqueeze(1))` 來挑選出我們**實際採取**的那個動作 (`a1=1`) 所對應的Q值。
        
    - 所以，對於這條經驗，`current_q` = **22.5**。
        
3. **計算 `target_q` (我應該學習的目標是多少？)**
    
    - 將這64個 `next_states` (包含 `s2`) 一起輸入**`target_net`** (注意！是目標網路)。
        
    - 假設 `target_net` 對於 `s2` 的輸出是 `[23.0, 22.8]`。
        
    - 我們取其中最大的值，`next_q` = `max(23.0, 22.8)` = **23.0**。這代表目標網路認為，在 `s2` 這個狀態下，能獲得的最大未來價值是23.0。
        
    - 現在使用貝爾曼方程計算目標Q值： `target_q = reward + gamma * next_q * (1 - done)` `target_q = 1.0 + 0.99 * 23.0 * (1 - 0)` `target_q = 1.0 + 22.77` = **23.77**
        
4. **計算損失並更新**
    
    - 我們有了 `q_net` 的預測 (`current_q` = 22.5) 和一個更穩定的學習目標 (`target_q` = 23.77)。
        
    - 計算均方誤差損失 (MSE Loss): `loss = (22.5 - 23.77)²`。
        
    - `optimizer` 根據這個 `loss` 進行反向傳播，**只更新 `q_net` 的權重**，讓它下一次再遇到 `s1` 時，輸出的Q值能更接近 23.77。
        
    - `target_net` 的權重在此步驟中**保持不變**。
        

#### 第4步：更新目標網路

- `step_count` 不斷累加。當 `step_count` 成為 `update_target_freq` (100) 的倍數時，執行 `self.target_net.load_state_dict(self.q_net.state_dict())`。
    
- 這時，`q_net` 在過去100步中學到的「新知識」被完整地複製給了 `target_net`。`target_net` 更新到了一個新的、更準確的狀態，為接下來100步的訓練提供指導。
    

透過以上四個步驟的不斷循環，`q_net` 逐漸學會了準確預測每個狀態-動作的價值，從而能做出最優決策，讓桿子立得更久。