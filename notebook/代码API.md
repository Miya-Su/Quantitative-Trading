# 代码函数说明

### def read_factor_data(factors_name_list, trade_date_20d_series)

从文件夹读取数据

#### `def risk_factor(alpha_factor_series)`

读取十个风险因子，加入Pandas Series中，完成数据构建



### def nonefill(data,date_list)

数据缺失值填充（按照行业均值填充）

#### ``def fillna_ind_mean(data,date_list)``

对DataFrame格式特征数据按照股票行业均值填充空值

#### `def data_fillna_mean(data,date_list)`

对Series中的每一个DataFrame格式的特征数据进行缺失值填充



### def predict_probo_class(data,model,loop,date_list，T,x,changeindex_model)

分类概率预测

输入DataFrame格式数据、分类预测模型、循环预测周期、时间、分类的标签值百分比T（前后T%的数据）、以及标签值分类数据x（x=0为三分类，x=-1为二分类）、因子数据构造模型（两种，二分类和三分类）

得到预测的因子概率值、预测准确率

#### `def data_input_class(data,date_list,T,x)`

输入DataFrame格式数据、时间、分类的标签值百分比T（前后T%的数据）、以及标签值分类数据x（x=0为三分类，x=-1为二分类）

##### `def concat_data(data,date_list,is_rise_df)`

将特征数据和标签值进行合并

##### `def rise_fall_class(trade_date_m_series，T)`

划分标签值,T为百分数值，T=30表示：将涨跌幅上涨的前30%的股票划分为一类

##### `def data_train_test(data_pct,x)`

划分测试数据和训练数据

| x=-1    | 二分类（前T%为1，其余-1）                         训练数据为（1、-1），测试数据为（1、-1） | def changeindex2(data,M,date_list)     |
| :------ | :----------------------------------------------------------- | -------------------------------------- |
| **x=0** | **三分类（前T%为1，后T%为-1，中间为0)训练数据（1、-1），测试数据（1、-1、0）** | **def changeindex3(data,M,date_list)** |

#### `def splitdata(train,test,i,j,x,y)`

划分每次循环的测试集和训练集，得到的测试集和训练集为numpy.array格式

##### `def countsum(data)`

根据时间划分数据，同一时间的股票数据划分为一组

##### `def standard(X_train,X_test)`

数据标准化

#### `def changeindex2(data,M,date_list)`

因子数据构造，二分类，x=-1时,选择此模型，得到两种分类的因子概率值

##### `def change(data,n,m,M,date_list)`

#### `def changeindex3(data,M,date_list)`

因子数据构造，三分类，x=0时,选择此模型，会得到三种分类的因子概率值

#### `def accu_score(score,name)`

模型准确率评估（每次循环得到的训练集和预测集上的准确率）



### def predict_probo_reg(data,model,loop,date_list)

回归概率预测

输入DataFrame格式数据、回归预测模型、循环预测周期、时间、回归的标签值（涨跌幅数值），因子数据构造模型

得到预测的因子值、预测RMSE

#### `def data_input_reg(data,date_list)`

输入DataFrame格式数据、时间

##### `def concat_data(data,date_list,is_rise_df)`

将特征数据和标签值进行合并

##### `def rise_fall_class(trade_date_m_series)`

获取标签值：股票涨跌幅的具体数值

##### `def data_train_test(data_pct)`

划分测试数据和训练数据

#### `def splitdata(train,test,i,j,x,y)`

##### `def countsum(data):`

##### `def standard(X_train,X_test)`

#### `def changeindex1(data,M,date_list)`

因子数据构造，将回归模型预测得到的值狗造成因子数据

#### `def change(data,n,m,M,date_list)`



### class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin)

简单平均集成学习，通过合成模型进行分类/回归预测

#### `def rmsle_cv(model,data,loop)`

回归预测的交叉验证，通过计算RMSE验证单个模型的准确性，输入模型、数据以及预测周期

#### `def rmsle(model,data,loop,date_list)`

回归预测

#### `def rmse_show(rmse_list,label_list)`

RMSE评估，得到每次循环预测的RMSE，画图显示



### class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin)

Meta-model Stacking：使用前面分类器产生的特征输出作为最后总的meta-classifier的输入数据

#### `def rmsle_cv(model,data,M)`

#### `def rmsle(model,data,loop,date_list)`

#### `def rmse_show(rmse_list,label_list)`



### def factor_test_T(factor_list,factor_name)

因子T检验



###  def factor_analyse(name,factor)

因子测试

#### `def show1(factor_test_obj)`

净值绩效，对冲表现

#### `def show2(factor_test_obj)`

计算因子年化收益，最大回撤，月胜率

#### `def show3(factor_test_obj)`

计算因子绩效

#### `def show4(factor_test_obj)`

计算分组绩效

#### `def show5(factor_test_obj)`

计算分组年化收益

#### `def show6(factor_test_obj)`

计算因子IC值，看IC衰减



### def factor_stock_choose(factor,num)

   投资组合因子检验，直接选取因子值高的前num支股票进行投资组合构建（一般100支或者80支）

   这里为预测后涨跌幅上涨的TOP num支股票

#### `def factor_test_pre(factor)`

因子中性化处理

#### `def stock_choice(data,num)`

投资组合构建：有两种选择

等权选num支股票（每支股票相同权值1/num）

按照基准行业选取num支股票（股票按照基准行业进行权重划分）

##### `def stock_bench_ind(data,num)`

基准行业划分

###### `def get_industry_stock(stock_factor_series, set_date, stock_num=100)`

获得股票行业

###### `def get_bench_ind_weight(set_date, bench_code='ZZ500')`

按照基准行业划分股票权重

##### `def show01(perform_obj)`

净值绩效、对冲表现

##### `def show02(perform_obj)`

计算策略收益

##### `def show03(perform_obj)`

计算换手率

##### `def show04(perform_obj)`

计算分组年化收益