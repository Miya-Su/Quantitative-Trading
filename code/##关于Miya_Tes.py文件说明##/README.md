# Miya_Test.py

Miya_Test.py文件为封装包文件，包含下面34个函数部分，用于分类预测，二分类时直接调用。

可以直接导入使用，完成概率因子的测试。

*code文件夹中158Factor_Test和150Factor_Test均导入Miya_Test.py进行测试



## 使用说明

下面以周频数据为例，一共分为4部分：

1.调用miya.predict_probo_class进行概率预测得到二分类的两种概率因子

2.调用miya.factor_analyse将概率因子做因子处理

3.=调用miya.factor_stock_choose将概率因子用作投资组合检验

4.调用miya.show01，miya.show1进行图像分析

代码例如下：

```
import Miya-Test as miya

#预测概率，使用lgb分类模型，涨跌幅按照前30%划分为1类，其余为0，滚动预测周期为前48期预测后一期
Wfactor0_lgb3,Wfactor1_lgb3,Wtest_accuracy_lgb3,Wtrain_accuracy_lgb3=miya.predict_probo_class(week_data,model_lgb,48,week_list,0.3,-1)
#因子检验
Wfactor1_lgb3_obj=miya.factor_analyse("test1",Wfactor1_lgb3) 
#画图显示
miya.show1(Wfactor1_lgb3_obj)
#投资组合检验
Wsamew_lgb3_pre,Wsamew_lgb3_unpre,Wunw_lgb3_pre,Wunw_lgb3_unpre=miya.factor_stock_choose(Wfactor1_lgb3,100)
#画图显示
miya.show01(Wsamew_lgb3_pre)
```

Wfactor0_lgb3：概率为0的因子

Wfactor1_lgb3：概率为1的因子

Wtest_accuracy_lgb3：滚动预测测试集的准确率

Wtrain_accuracy_lgb3：滚动预测验证集的准确率

Wfactor1_lgb3_obj: 经过因子检验1预处理后的因子

Wsamew_lgb3_pre:等权构建投资组合选取100支后的因子表现，经过中性化处理

Wsamew_lgb3_unpre:等权构建投资组合选取100支后的因子表现，没有经过中性化处理

Wunw_lgb3_pre:不等权构建投资组合选取100支后的因子表现，经过中性化处理

Wunw_lgb3_unpre:不等权构建投资组合选取100支后的因子表现，没有经过中性化处理



## 函数说明

### 分类预测

##### 1.def predict_probo_class(data,model,loop,date_list,T,x)

​	return  factor0_df,factor1_df,test_accuracy,train_accuracy



### 数据构造

#加入十个风险因子

##### 2.def risk_factor(data):

 	return data

##### 3.def rise_fall_class(trade_date_m_series, T):             

​	 return is_rise_df            #标签值-分类


#### 划分测试集和训练集
##### 4.def concat_data(data, date_list, is_rise_df):

​    return data_dict

##### 5.def data_train_test(data_pct, x):

​    return data_pct_train, data_pct_test

##### 6.def data_input_class(data, date_list, T, x):

​    return train1, test1


#### 数据标准化，可以处理空值
##### 7.def standard(X_train, X_test):

​    return X_train_scaled, X_test_scaled

##### 8.def countsum(data):

​    return resultdata, a

#### 两分类划分，划分训练集data，测试集alldata

（训练集的类别只有（0,1），测试集包含所有类别（0,1，-1））

##### 9.def splitdata(data,train,test, i, j, x, y):

​	return X_train_scaled, X_test_scaled, Y_train, Y_test, X_test



### 缺失值填充

#### 按照行业均值进行填充

##### 10.def fillna_ind_mean(data,date_list):   

​	return data                 #只做均值填充

##### 11.def data_fillna_mean(df,date_list):

 	return df               #空值填充，行业均值填充

##### 12.def nonefill(data,date_list):

​	return tempdata



### 预测准确率分析

##### 13.def accu_score(score,name):



### 因子数据合成

##### 14.def change(data,n,m,M,date_list):

​	return factorF_df

##### 15.def changeindex2(data,M,date_list):

​	return  factor0_df,factor1_df

#### 因子T显著度

##### 16.def factor_test_T(factor_list,factor_name):

​	 return  risk_test

#### 因子处理(分成两种方向)

##### 17.def factor_analyse(name,factor): 

​        0-positive , 1-negetive

​	return factor_test_obj

#### 因子测试画图显示

##### 18.def show1(factor_test_obj):

##### 19.def show2(factor_test_obj):

##### 20.def show3(factor_test_obj):

##### 21.def show4(factor_test_obj):

##### 22.def show5(factor_test_obj):

##### 23.def show6(factor_test_obj):



#### 因子中性化预处理

##### 23.def factor_test_pre(factor):  

​    return df

##### 25.def factor_test(stock_weighted_series):

​    return perform_obj

##### 26.def show01(perform_obj):

​    perform_obj.net_value_plot()

##### 27.def show02(perform_obj):

​    perform_obj.get_strategy_perform()
​    return perform_obj.get_strategy_perform()

##### 28.def show03(perform_obj):

​    perform_obj.get_avg_turnover()
​    return perform_obj.get_avg_turnover()

##### 29.def show04(perform_obj):

​    perform_obj.get_annual_perform()
​    return perform_obj.get_annual_perform()



### 投资组合构建

##### 30.def stock_choice(data, num):

​	return stock_choice_obj        #直接挑选概率值前100支股票，等权

##### 31.def stock_bench_ind(data, num):  

​	return stock_choice_obj          #行业中性，基准权重后挑选100支股票

##### 32.def factor_stock_choose(factor, num):

​	return samew_pre, samew_unpre, unw_pre, unw_unpre

#### 行业基准权重

##### 33.def get_bench_ind_weight(set_date, bench_code='ZZ500'):

​	return bench_ind_weight_series

##### 34.def get_industry_stock(stock_factor_series, set_date, stock_num=100):

​	return port_series