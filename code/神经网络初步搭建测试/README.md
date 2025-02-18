# 处理流程

**目标：使用Tesosrflow搭建神经网络模型进行二分类预测**

****此部分主要为测试，存在很多问题，主要集中在神经层的建立，需要进一步学习。

初步测试：构建3层神经网络

存在问题：

- 神经层数搭建？搭建多少层？

- 神经元个数选择？

- 激活函数选择？隐藏层的激活函数如何选择？输出层的激活函数如何选择?

- 希望得到二分类的概率值，输出层激活函数选择softmax时存在问题，测试中使用sigmoid函数




## 搭建步骤

1.训练数据
2.定义节点准备接收数据
3.定义神经层：隐藏层和预测层
4.定义 loss 表达式
5.选择 optimizer 使 loss 达到最小

对所有变量进行初始化，通过 sess.run optimizer，迭代 500 次进行学习



# 神经层搭建

1. 初步搭建：三层神经网络测试

   - 输入层
   - 3个隐藏层：激活函数：relu
   - 输出层：sigmoid函数

2. 优化函数：AdamOptimizer算法作为优化函数，来保证预测值与实际值之间交叉熵最小

3. 学习速率：0.01

4. 神经元个数

   |        | 激活函数？？ | 神经元个数？？ |
   | ------ | ------------ | -------------- |
   | 第一层 | relu         | 30             |
   | 第二层 | relu         | 20             |
   | 第三层 | relu         | 20             |






# 数据构建



| **特征值：**     | **选取['mom','emotion', 'pricevol', 'minute', 'finance', 'valuation', 'alpha191','growth']  八大类因子组合构成特征值** |
| ---------------- | ------------------------------------------------------------ |
| **分类标签值：** | **按照股票涨跌幅排序，收益前T%标记为1，其余为0**             |
| **数据时间：**   | **2009年10月-2018年8月**                                     |
| **缺失值处理：** | **按照行业均值填充**                                         |
| **数据标准化：** | **基于原始数据的均值（mean）和标准差（standard deviation）** |
| **数据集：**     | **周频：1094178 rows × 161 columns**                         |

|                    |                                                  |
| ------------------ | ------------------------------------------------ |
| **预测滚动周期：** | **周频数据：根据前48周的数据值预测后一周的数据** |





