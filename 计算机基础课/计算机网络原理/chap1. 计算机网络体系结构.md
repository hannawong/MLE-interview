

### 1. 计算机网络的构成

#### 1.1 资源子网（用户子网）
- 基本组成：
  - 服务器 (server,图中的subnet)
  - 客户计算机，移动终端 (host)

![img](https://pic1.zhimg.com/80/v2-a1d9d188a0e8876c5132555c76846b2e_1440w.png)

#### 1.2 通信子网
- 基本组成：

  - 通信**线路**（或称通道）
  - **网络互连设备**（路由器、交换机、HUB等）

  注：通信子网和资源子网的划分是以功能为依据的，可以理解为OSI模型的不同层次。其中，通信子网对应低三层（物理层、数据链路层、网络层），由各种网络通信设备（如路由器、交换机）和通信线路组成，负责数据传输。资源子网对应高三层（会话层、表示层、应用层），由计算机等终端组成，负责处理数据。

- 基本结构

1. 点到点通道
  - 基本特点：一条线路连接二台网络互连设备；一般情况下，二台计算机的连接要经过多台网络互连设备(多跳)
  - 典型拓扑结构
    • star, ring（loop）, tree, complete, intersecting rings, irregular等
  - 关键技术：路由选择（Routing）

![img](https://pic1.zhimg.com/80/v2-5f943bb869b10266d803d6d9d087225b_1440w.png)

2. 广播通道
   - 基本特点：多台计算机**共享**一条通信线路；任一台计算机发出的信息可以直接被其它计算机接收
   - 典型拓扑结构
     – bus, ring
   - 关键技术：**通道分配**
     • 静态分配：分时间片。特点：控制简单，通道利用率低
     • 动态分配：各站点动态使用通道。特点：控制复杂，通道利用率高
     通道分配方法：
     – 集中式：只有一个仲裁机构
     – **分布式**：各站点均有仲裁机

![img](https://pic1.zhimg.com/80/v2-cdd115940b702c997951b711e8991c00_1440w.png)

- 网络分类（从地域范围角度）：
  • 局域网络（**L**ocal **A**rea **N**etworks）：主要采用**广播**通道技术
  • 城域网络（Metropolitan Area Networks）
  • 广域网络（Wide Area Networks）：主要采用点到点通道技术

### 2.计算机网络的体系结构

计算机网络的体系结构：对计算机网络及其部件所完成功能的比较精确的定义。即从**功能**的角度描述计算机网络的结构，是层次和层间关系的集合。
注意：计算机网络体系结构仅仅定义了网络及其部件通过协议应完成的功能；不定义协议的实现细节和各层协议之间的接口关系。

网络功能的分层-协议的分层-体系结构的分层

协议**分层**易于协议的设计、分析、实现和测试。

#### 2.1 计算机网络功能的分层
计算机网络的基本功能是为地理位置不同的计算机用户之间提供访问通路。
因此，下述功能是必须提供的：
‧ 连接源结点和目的结点的**物理传输**线路，可经过中间结点
‧ 每条线路两端的结点利用波形进行二进制通信
‧ **无差错**的信息传送
‧ 多个用户**共享**一条物理线路
‧ 按照地址信息，进行**路由选择**
‧ 信息缓冲和流量控制
‧ 会话控制
‧ 满足各种用户、各种**应用**的访问要求

上述功能有三个显著特点：
– 上述功能必须同时满足**一对**用户
– 用户之间的通信功能是相互的
– 这些功能分散在各个网络设备和用户设备中。
一般采用“层次结构”的方法来描述计算机网络，即：**计算机网络中提供的功能是分成层次的**。

#### 2.2 协议和协议的分层结构
- 协议的定义和组成
  – 层次结构的计算机网络**功能**中，最重要的功能是**通信功能**
  – 这种通信功能主要涉及**同一层次**中通信**双方**的相互作用
  – 位于不同计算机上进行对话的**第N层**通信各方可分别看成是一种进程，称为**对等进程**。
  – 协议（Protocol）： 计算机网络**同等层次**中，通信双方进行信息交换时必须遵守的规则。
  – 协议的组成
  - 语法（syntax）：以二进制形式表示的命令和响应的**结构**
  - 语义（semantics）：由发出的命令**请求**，**完成**的动作和**回送**的响应组成的集合
  - 定时关系（timing）：有关事件顺序的说明
- 协议的**分层**和**层间**结构
  - 协议的分层
    - 目的计算机第N层收到的报文与源主机第N层发出的报文相同
    - 洋葱结构
    - 协议分层要保证整个计算机网络系统功能完备、高效。
  - 每一**相邻层**之间有一个**接口**（Interface），它定义了**下层向上层**提供的原语操作和服务。
  - 对于第N层协议来说，它有如下特性
    - 不知道上、下层的内部结构；
    - **独立**完成某种功能；
    - 为上层提供服务；
    - 使用下层提供的服务。

#### 2.3 计算机网络的体系结构

- 基本术语与分层结构
  - 接口：定义了**下层向上层**提供的原语操作和服务。
  - 协议：计算机网络**同等层次**中，通信双方进行信息交换时必须遵守的规则。
  - 服务：**层间**交换信息时必须遵守的规则。（注意区分协议与服务，协议是同等的，而服务是层间的）
    ![img](https://pic3.zhimg.com/80/v2-ee0b6fad5e502b2b2f3913bc9156f8ce_1440w.png)

- 服务访问点SAP（Service Access Point）：任何**层间**服务是在**接口的SAP**上进行的。每个SAP有唯一的识别地址；每个层间接口可以有**多个SAP**。
- 接口数据单元IDU（Interface Data Unit）：IDU是通过SAP进行传送的**层间**信息单元，= 上层的服务数据单元**SDU**（Service Data Unit）+ **接口**控制信息ICI（Interface Control Information）。
- 协议数据单元PDU（Protocol Data Unit）：第N层传送给它的**同层**的信息单元。PDU = 服务数据单元SDU或其分段（后面讲分段和重组） + 协议控制信息PCI（Protocol Control Information）。

![img](https://pic1.zhimg.com/80/v2-49b6eb260dbadc32b499956eebbf979a_1440w.jpeg)

![img](https://pica.zhimg.com/80/v2-ca1fc74a5bf457e9218469ca4fbea562_1440w.jpeg)

- 服务的分类：

  - 面向连接的服务：当使用服务传送数据时，**首先建立连接**，然后使用该连接**传送数据**。使用完后，**关闭连接**。
    – 特点：顺序性好。
  - 无连接服务：直接使用服务传送数据，每个**包**独立进行传送。
    – 特点：顺序性差

  注意：连接并不意味可靠，可靠要通过确认、重传等机制来保证

- 服务原语

  请求（Request）: 一个实体想得到某些事情的服务

  指示（Indication）: 另一个实体被通知做这个事情

  响应（Response）: 另一个实体完成事情后想响应

  确认（Confirm）: 返回一个对早期请求的响应

### 2.4 典型计算机网络的参考模型

#### 2.4.1 OSI参考模型

- 物理层（The Physical Layer）：在物理线路上传输原始的**二进制数据**位（基本网络硬件）
- 数据链路层（The Data Link Layer）：在有差错的物理线路上提供无差错的数据传输（Frame）
- 网络层（The Network Layer）：控制**通信子网**提供源点到目的点的数据传送（Packet）
- 运输层（The Transport Layer）：为用户提供端到端的数据传送服务。
- 会话层（The Session Layer）：为用户提供会话控制服务（安全认证）
- 表示层（The Presentation Layer）：为用户提供数据转换和表示服务。
- 应用层（The Application Layer）

![img](https://pic2.zhimg.com/80/v2-ae5468cd4951089907c355da6c42d777_1440w.jpeg)

![img](https://pic3.zhimg.com/80/v2-ef4ee583f3836fcaf3ddf7c07cf01f48_1440w.jpeg)

#### 2.4.2 TCP/IP 参考模型

- 物理层：在物理线路上传输原始的二进制数据位
- 数据链路层：在有差错的物理线路上提供无差错的数据传输
- 网络层（IP层）：控制**通信子网**提供源点到目的点的包传送
- 运输层：提供端到端的数据传送服务。TCP 和 UDP
- 应用层：提供各种 Internet 管理和应用服务功能

![img](https://pic1.zhimg.com/80/v2-171eb3e7e98b06ab03af6680a0f44a15_1440w.png)









