# 数据通信的基本原理

### 1. 数据通信的理论基础

时域观：对于音乐来说，时域就是一个随着时间变化的震动

频域观：一个永恒的音符（doremi）

“你眼中看似落叶纷飞变化无常的世界，实际只是躺在上帝怀中一份早已谱好的乐章。”

#### 1.1 傅立叶分析

> 傅里叶同学告诉我们：任何周期函数，都可以看作是不同振幅，不同相位正弦波的叠加。

任何一个**周期为T**的有理周期性函数 g(t) 可分解为若干项（可能无限多项）**正弦和余弦函数**之和：
$$
g(t) = \frac{c}{2} + \sum_{n=1}^\infty a_nsin(2\pi nft) + \sum_{i=1}^\infty b_n cos(2\pi nft)
$$
这就将时域上的信号g(t)分解成了若干频率（频域角度）信号的叠加。

其中，

$c = \frac{2}{T} \int_0^T g(t)dt$

$a_n = \frac{2}{T} \int_0^T g(t) sin(2\pi nft)dt$

$b_n = \frac{2}{T} \int_0^T g(t) cos(2\pi nft)dt$



- f = 1/T: 基本频率。这个信号的所有频率成分都是基频的整数倍。信号的周期 = 基频的周期
- an, bn: n次谐波项的正弦和余弦**振幅**值

#### 1.2 有限带宽信号

**频谱**是一个信号所包含的频率范围。信号的**绝对带宽**等于频谱的宽度。许多信号的带宽是无限的，然而信号的主要能量集中在相对窄的频带内，这个频带被称为有效带宽。信号的信息承载能力与带宽有着直接关系，带宽越宽，信息承载能力越强。

信号在信道上传输时的特性：

- 对不同傅立叶分量的衰减不同，因此引起输出失真：信道有截止频率fc, 0 ~ fc频率范围的振幅不衰减， fc频率以上的振幅衰减厉害。 0 ~ fc称为信道的**有效带宽**；
- 通过信道的谐波（正弦/余弦）次数越多，信号越逼真。

波特率和比特率的关系：

- 波特率：信号每秒钟变化的次数

- 比特率：每秒钟传送的二进制位数。

  波特率与比特率的关系取决于信号值与比特位的关系。例：每个信号值可表示３位，则比特率是波特率的３倍；若每个信号值可表示１位，则比特率和波特率相同。

**奈奎斯特：无噪声有限带宽信道的最大数据传输率公式**

最大数据传输率 = $2Hlog_2^V$ (bps)
任意信号通过一个带宽为Ｈ的低通滤波器，则每秒采样Ｈ次就能完整地重现该信号，信号电平分为Ｖ级。

**香农：随机（热）噪声干扰的最大数据传输率**

热噪声出现的大小用信噪比（信号功率S与噪声功率N之比）来衡量。$10log_{10}^{S/N}$, 单位：分贝

带宽为 H 赫兹，信噪比为S/N的任意信道的最大数据传输率为：$Hlog_2^{(1 + S/N)}$ (bps), 与信号电平级数无关。

### 2. 数据通信技术

#### 2.1 数据通信系统的基本结构

![img](https://pic1.zhimg.com/80/v2-e7de10dd26fcf07aa7e7ce6fa017d08b_1440w.jpeg)

从**时域观**来看，电磁信号分为模拟信号和数字信号。模拟信号的信号强度随着时间**平滑**变化；数字信号是0/1突变的。

- 数据
  - 模拟数据 (Analog Data) -- 连续值
  - 数字数据 (Digital Data) -- 离散值
- 数据传输方式
  - 模拟信号 (Analog Signals) - 模拟信道
  - 数字信号 (Digital Signals) - 数字信道


- 数字数据的**数字**传输（**基带**传输）
  - 基带：基本频带，指传输变换前所占用的频带，是原始信号所固有的频带。
  - 基带传输：在传输时直接使用基带信号。
    - 基带传输是一种最简单最基本的传输方式，一般用**低电平**表示“0”，**高电平**表示“1”。

- 数字数据的**模拟**传输（**频带**传输）
  - 频带传输：指在一定频率范围内的线路上，进行**载波传输**。用基带信号对载波进行**调制**，使其变为**适**
    **合于线路传送的信号**。
  - 调制（Modulation）：用基带脉冲对载波信号的某些**参量**（幅度、频率、相位）进行控制，使这些参量随基带脉冲变化。
  - 解调（Demodulation）：**调制的反变换**。
  - **调制解调器MODEM**（modulation-demodulation)

![img](https://pic3.zhimg.com/80/v2-622949fcfde8beef0f7cd34cec711b1b_1440w.png)

​                                                                     (b: 调幅, c: 调频 d: 调相位)

- **模拟**数据**数字**传输
  - 根据Nyquist原理进行采样。(频率为2f)
  - （1）常用的PCM技术. 将模拟信号**振幅**分成多级（2n），每一级用 n 位表示。– 例如：贝尔系统的 T1 载波将模拟信号分成128级，每次采样用7位二进制数表示。
  - （2）差分脉冲代码调制 – 原理：不是将振幅值数字化，而是根据前后两个采样值的**差**进行编码，输出二进制数字。
  - 编码解码器CODEC

![img](https://pic2.zhimg.com/80/v2-e772337cd518a73bc77fda40f8cdb5f8_1440w.jpeg)

#### 2.2 多路复用技术
由于一条传输线路的能力远远超过传输一个用户信号所需的能力，为了提高线路利用率，经常让多个信号同时共用一条物理线路。常用的有三种方法：

- 时分复用 TDM（Time Division Multiplexing）：主要用于**数字**数据传输。如T1载波，分成 24 个信道

![img](https://pic3.zhimg.com/80/v2-091618dd8f83882a02e472b18f57e32f_1440w.png)

- 频分复用 FDM（Frequency Division Multiplexing）

![img](https://pic3.zhimg.com/80/v2-0ba7888d99bf39a2a7f54e66b980b087_1440w.png)

- 波分复用 WDM（Wavelength Division Multiplexing）

#### 2.3 通信线路的通信方式

- 通信方式
  - 单工通信方式：信息只能单向传输，监视信号可回送。
  - 半双工通信方式：信息可以双向传输，但在某一时刻只能单向传输。
  - 全双工通信方式：信息可以同时双向传输，一般采用四线式结构。（传输+回送）

- 同步方式：接收方必须知道每一位信号的开始和持续时间。下面以字符传递为例说明。

  - 异步

    - 信息是以**字符为单位**传送的；每个字符由发送方异步产生，有随机性。
    - 字符一般采用5，6，7或8位二进制编码；
    - 需要辅助位，每个字符可能需要用10位或11位才能传送，例如：
      • 起始位，1位；
      • 字符编码，7位；
      • 奇偶校验位，1位；
      • 终止位，1 ~ 2位。

    ![img](https://pic3.zhimg.com/80/v2-bd6eb2fb18d87bdafa489092f3c63772_1440w.png)

  - 同步

    - 信息是以**报文**为单位传送的；
    - 传输开始时，以同步字符使收发双方同步（传输的信息中不能有同步字符出现）
    - 从传输信息中抽取同步信息，修正同步，保证正确采样。

    ![img](https://pic3.zhimg.com/80/v2-f571457a5328e67bf18ff20fa21e6ec9_1440w.png)

- 通信交换方式

  - 电路交换（circuit switching）：直接利用可切换的**物理通信**线路，连接通信双方。

    - 在发送数据前，必须建立起点到点的物理通路；
    - 建立物理通路时间较长，数据传送延迟较短；

  - 报文交换（message switching）：信息以报文（**逻辑上完整的信息段**）为单位进行存储转发。

    - 线路利用率高；
    - 要求中间结点（网络通信设备）**缓冲大**；
    - 延迟时间长。

  - 分组交换（**packet** switching）：信息以分组为单位进行存储转发。源结点把报文分为分组，在中间结点存储转发，目的结点把分组合成报文。
    分组：**比报文还小的信息段**，可定长，也可变长。

    - 线路利用率高；
    - 结点存储器利用率高；
    - 延迟短；
    - 额外信息增加。

    **数据报分组交换**：每个分组均带有全称网络地址（源、目的），可走不同的路径。

    **虚电路分组交换**：电路交换与分组交换的结合。来自同一个流的分组通过一个预先建立的虚电路传输，来保证分组的顺序。但是不同虚电路的分组可能会交错在一起。

    - 建立：发带有全称网络地址的呼叫分组，**建立虚电路**；
    - 传输：沿建立好的虚电路传输数据；
    - 拆除：拆除虚电路。

    ![img](https://pic1.zhimg.com/80/v2-30a2d973710ff3e773bc57a24eda1ca5_1440w.png)



