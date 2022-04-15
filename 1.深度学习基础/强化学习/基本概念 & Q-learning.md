# 强化学习从入门到入土(1)

## 1. 基本概念

#### 1.1 术语

首先，我们来定义一些名词：

- agent(智能体)，可自由移动，对应玩家；
- action(动作)，由玩家做出，包括向上移动和出售物品等；
- reward(奖励)，由玩家获得，包括金币和杀死其他玩家等；
- environment(环境)，指玩家所处的地图或房间等；
- state(状态)，指玩家的当前状态，如位于地图中某个特定位置或房间中某个角落；
- goal(目标)，玩家的目标为获得尽可能多的奖励；

玩家(agent)将通过**与环境互动**并获得执行**行动**的**奖励**来学习环境。

> **agent** will learn from the **environment** by **interacting** with it and receiving **rewards** for performing **actions**.

这里“与环境互动”十分重要。其实，我们人类就是通过与环境互动来学习知识的。例如，你是一个襁褓中的孩子，你看到一个壁炉，并接近它。它很温暖，很积极，你感觉很好  *（积极奖励+1）。* 你感觉火是一件好事。

![img](https://static.leiphone.com/uploads/new/sns/article/201812/1545899394132458.png)

但是你试着触摸火焰。哎哟! 它会烧伤你的手  *（负奖励-1）*。你已经明白，当你离足够远时火是积极的，因为它会产生温暖。但是太靠近它会烧伤你。

![img](https://static.leiphone.com/uploads/new/sns/article/201812/1545899394785314.png)

这就是人类通过互动学习的方式。强化学习是一种从行动中学习的计算方法。

#### 1.2 最大化预计累计奖励(expected cumulative reward)  

强化学习循环输出state，action和reward的序列，agent的目的是最大化预计累计奖励(expected cumulative reward)  

每个时间步t的累积奖励可定义为：

![img](https://static.leiphone.com/uploads/new/sns/article/201812/1545899395487400.png)

即为t步之后所有步的收益之和，这相当于：

![img](https://static.leiphone.com/uploads/new/sns/article/201812/1545899395418123.png)

但是实际上，在早期提供的奖励更有用，因为它们比未来长期的奖励**更好预测**（predictable）。所以，我们按照时间对奖励进行discount. 折扣系数$\gamma$是一个位于0~1之间的数。

- $\gamma $越大，折扣越小。这意味着agent更关心long-term reward。
- $\gamma $越小，折扣越大。这意味着 agent  更关心short-term reward。

累积的折扣预期奖励(cumulative discounted reward at t)是：

![img](https://static.leiphone.com/uploads/new/sns/article/201812/1545899396643169.png)

#### 1.3 Episodic or Continuing tasks

- Episodic task: 有一个起点和终点（一个最终状态）。这会创建一个episode(就像电视剧的剧情一样)：一个状态States, 行动Actions, 奖励Rewards, 新状态 New States的列表。例如，Super Mario游戏中，一个剧情开始于游戏开始，结束于当马里奥被杀或者达到了关卡的末尾。
- Continuing tasks:没有终点状态。在这种情况下，agent必须学习如何选择最佳操作并同时与环境交互。例如，agent进行自动股票交易。对于此任务，没有起点和终点状态。agent 会持续执行，直到我们手动把它停止。

#### 1.4 Exploration/Exploitation trade off

- Exploration：寻找有关环境的更多信息。
- Exploitation：利用已知信息来最大化奖励。



## 2. Q-learning

#### 2.1 Q-table

Q-table的列为不同的状态states，行为我们可选的actions，我们将计算每种状态 state 下采取的每种行动 action的**最大的未来预期奖励**(maximum expected feature rewards)。

> Each Q-table score will be the **maximum expected future reward** that I’ll get if I **take that action** at **that state** with the **best policy given**.

Q-table就像一个cheatsheet，对于每个不同的state，我们都去查一查Q-table，然后找到在该state下使用不同的action带来的最大未来预期奖励，然后选择那个最大的action即可。

那么，如何去获得Q-table中每个元素的值呢？

#### 2.2 Q-learning algorithm

得到Q-table中每个元素值的函数叫Q-function, 它 有两个输入：“state”和“action”。它返回该action在该state下的预期未来奖励。

![1*6IqzImIFK1oEiVWmlu1Esw](https://cdn-media-1.freecodecamp.org/images/1*6IqzImIFK1oEiVWmlu1Esw.png)

**step1：  初始化Q-table**  

我们将值初始化为0。

![深度强化学习从入门到大师：通过Q学习进行强化学习(第二部分)](https://img1.3s78.com/codercto/8b1833fbeb6a8e378e745ca2c5b14f13)



之后，重复步骤2到4，直到算法运行次数为 episode 的最大值（由用户指定）或直到我们手动停止训练。

**step2. 选择**

据当前的Q-table，选择当前state下的action。但是......如果每个Q值都是零，那么在该采取什么行动？

这就是我们在中谈到的**exploit/explore-tradeoff**的重要性。

![1*9StLEbor62FUDSoRwxyJrg](https://cdn-media-1.freecodecamp.org/images/1*9StLEbor62FUDSoRwxyJrg.png)

我们的想法是，在开始时，我们将使用**epsilon-greedy策略**：

- 我们指定一个探索率“epsilon”，我们在开始时设置为1，即**随机**执行的step的比例。刚开始学习时，这个速率必须是最高值，因为我们对Q-table的取值一无所知。这意味着我们需要通过随机选择我们的行动进行大量探索。
- 生成一个随机数。如果这个数字> epsilon，那么我们将进行“exploit”（这意味着我们使用已知的Q-table信息来选择每一步的最佳action）。否则，我们会进行explore。
- 在Q函数训练开始时我们必须有一个较大的epsilon。然后，随着Agent对于Q-table的值越来越确定，再减小这个epsilon。

**步骤3-4：评估**

采取action a 并观察结果state s' 和reward r，并更新Q-function Q(s,a)。

我们采取我们在步骤2中选择的action，然后执行此action将返回一个新的state s'和reward r。

然后，使用Bellman方程更新Q(s,a)：

![img](https://pica.zhimg.com/80/v2-6f0c259e9e7a662d93007f5b74d6f21a_1440w.png)

```python
New Q value =   Current Q value +   lr * [Reward + discount_rate * (highest Q value between possible actions from the new state s’ ) — Current Q value ]
```



---

示例：

Q-learning with FrozenLake游戏：

![Frozen Lake](https://github.com/simoninithomas/Deep_reinforcement_learning_Course/raw/c29ac479d33d5f2bc9a60b54b4822e60fc7819d0/Q%20learning/FrozenLake/frozenlake.png)

从start点走到end点，只能走冰面而不能走漩涡；但是，由于冰面很滑，所以并不一定每次都能走到你想走的位置。

```
env = gym.make("FrozenLake-v0")
action_size = env.action_space.n
state_size = env.observation_space.n
qtable = np.zeros((state_size, action_size))
print(qtable)
```

![img](https://pic3.zhimg.com/80/v2-091adcc737b75d399d6359b6dfe3abde_1440w.png)

```python
##parameters
total_episodes = 15000        # Total episodes
learning_rate = 0.8           # Learning rate
max_steps = 99                # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.005             # Exponential decay rate for exploration prob
```

Now we implement the Q learning algorithm:

```python
# List of rewards
rewards = []

# 2 For life or until learning is stopped
for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    
    for step in range(max_steps):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)
        
        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])

        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        
        total_rewards += reward
         # Our new state is state
        state = new_state
        
        # If done (if we're dead) : finish episode
        if done == True: 
            break
        
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
    rewards.append(total_rewards)

print ("Score over time: " +  str(sum(rewards)/total_episodes))
print(qtable)
```

![img](https://pic3.zhimg.com/80/v2-8e1de66fc36861a208a30afe825eec4e_1440w.png)

After 10 000 episodes, our Q-table can be used as a "cheatsheet" to play FrozenLake"!