import pandas as pd
import numpy as np
import datetime as dt
import daolib.dao as dao
import daolib.dso as dso
import util.sectool as sectool
import util.operatetool as optool
import matplotlib.pyplot as plt
import pickle
import time
from tqdm import tqdm

#分类预测
def predict_probo_class(data,model,loop,date_list,T,x):
    train,test=data_input_class(data,date_list,T,x)
    predict_value=[]
    test_accuracy=[]
    train_accuracy=[]
    start_clock  = time.time()
    for m in range(len(date_list)-loop):
        X_train, X_test, Y_train, Y_test,Xx_test= splitdata(data,train,test,m+loop-1,m,m+loop-1,m+loop)
        model.fit(X_train,Y_train)
        predict=model.predict_proba(X_test)
        predict=pd.DataFrame(predict)
        score_test=model.score(X_test,Y_test)
        score_train=model.score(X_train,Y_train)
        predict['stock']=Xx_test[:,-2]
        predict['date']=Xx_test[:,-1]
        predict_value.append(predict)
        test_accuracy.append(score_test)
        train_accuracy.append(score_train)
    end_clock = time.time()
    print('cpu cost time: ', end_clock - start_clock)
    factor0_df,factor1_df=changeindex2(data,predict_value,loop,date_list)
    return  factor0_df,factor1_df,test_accuracy,train_accuracy




#数据构造
#标签值 - -分类

def rise_fall_class(trade_date_m_series, T):
    zz_df = dao.get_index_component_data('ZZ')
    stock_price_df = dao.get_security_info('stock_price_info')[trade_date_m_series]
    stock_close_df = stock_price_df.xs('close', level=1)[trade_date_m_series]
    trade_status_df = stock_price_df.xs('trade_status', level=1)[trade_date_m_series]
    pause_df = trade_status_df.copy()
    pause_df[pause_df == 1] = np.nan
    pause_df[pause_df == 0] = 1

    stock_chg_df = stock_close_df.pct_change(axis=1)
    stock_return_df = stock_chg_df * pause_df * zz_df
    stock_return_df = stock_return_df.shift(-1, axis=1)

    is_rise_df = stock_return_df.copy()
    rise_quatile_percent = T
    rise_quan30_series = is_rise_df.quantile(rise_quatile_percent)
    rise_quan70_series = is_rise_df.quantile(1.0 - rise_quatile_percent)

    is_rise_df[is_rise_df > rise_quan70_series] = 999
    is_rise_df[is_rise_df < rise_quan30_series] = -999
    is_rise_df[is_rise_df.abs() != 999] = np.nan

    is_rise_df.replace(to_replace=[-999, 999], value=[-1, 1], inplace=True)
    return is_rise_df


# 划分测试集和训练集
def concat_data(data, date_list, is_rise_df):
    factor_class_series = data.map(lambda x: x.loc[:, date_list[0]:date_list[-1]])
    data_df = pd.DataFrame()
    factor_name_list = factor_class_series.index.tolist()
    data_dict = {}
    for trade_date in tqdm(date_list[:]):
        data_section_series = factor_class_series.map(lambda x: x[trade_date] if trade_date in x.columns else None)
        data_section_df = pd.DataFrame(data_section_series.to_dict())
        data_section_df = data_section_df.reindex(columns=factor_name_list)
        data_section_df['rise_fall'] = is_rise_df[trade_date]
        data_dict[trade_date] = data_section_df
        data_section_df['date'] = trade_date
    return data_dict


def data_train_test(data_pct, x):
    data_pct['rise_fall'] = data_pct['rise_fall'].fillna(x)  # 注意fillna填充！！！(x=-1是标志值二分类，分为“-1,1”两类)
    #     data_pct['rise_fall']=data_pct['rise_fall'].fillna(0)   #x=0是标志值三分类，分为“-1,0，1”两类
    data_pct_test = data_pct  # 包含0，-1,1的三种分类的全部数据预测集
    data_pct_test = data_pct[data_pct['trade_status'] == 0]  # 选择正常股票状态的数据
    #     data_pct_test=data_pct_test.dropna()    #删除空值
    data_pct_train = data_pct[~data_pct['rise_fall'].isin([0])]  # 不包含0的训练集
    data_pct_train = data_pct_train[data_pct_train['trade_status'] == 0]  # 选择正常股票状态的数据
    #     data_pct_dropna_train=data_pct_train.dropna()
    return data_pct_train, data_pct_test


def data_input_class(data, date_list, T, x):
    is_rise_df = rise_fall_class(date_list, T)
    data_dict = concat_data(data, date_list, is_rise_df)
    data_df = pd.concat([data_dict[frame] for frame in data_dict.keys()])
    train1, test1 = data_train_test(data_df, x)
    return train1, test1


# 数据标准化，可以处理空值
def standard(X_train, X_test):
    X_train_scaled = 1.0 * (X_train - X_train.mean()) / X_train.std()  # 数据标准化
    X_test_scaled = 1.0 * (X_test - X_test.mean()) / X_test.std()  # 数据标准化
    return X_train_scaled, X_test_scaled


def countsum(data):
    a = data.reset_index()
    a.rename(columns=lambda x: x.replace('index', 'stock'), inplace=True)
    resultdata = (a['stock'].groupby(a['date'])).describe()
    resultdata['sum'] = resultdata['count'].cumsum()
    return resultdata, a


# 两分类划分，划分训练集data，测试集alldata（训练集的类别只有（0,1），测试集包含所有类别（0,1，-1））
def splitdata(data,train,test, i, j, x, y):
    resultdata, a = countsum(train)
    resultalldata, b = countsum(test)
    i = resultdata['sum'][i]
    j = resultdata['sum'][j]
    x = resultalldata['sum'][x]
    y = resultalldata['sum'][y]

    newname = data.index.tolist()
    newname.append('stock')
    newname.append('date')

    X_train = np.array(a[newname][j:i])
    Y_train = np.array(a['rise_fall'][j:i])
    # 第x个月，测试集
    X_test = np.array(b[newname][x:y])
    Y_test = np.array(b['rise_fall'][x:y])
    X_train_scaled, X_test_scaled = X_train[:, :-3], X_test[:, :-3]

    return X_train_scaled, X_test_scaled, Y_train, Y_test, X_test

#缺失值填充
#按照行业均值进行填充
def fillna_ind_mean(data,date_list):
    industry_class_df=dao.get_stock_industry_data('CS')               #股票行业信息----周频信息
    industry_class_m_df=industry_class_df.loc[:,date_list]    #股票行业信息----月频信息
    #按照行进行填充，下一个值
    industry_class_m_df=industry_class_m_df.fillna(method='bfill',axis=1)
    industry_class_m_df=industry_class_m_df.fillna('未知')

    for i in range(1,len(date_list)):
        resl_series = pd.Series()
        industry_series=industry_class_m_df.iloc[:,i]
        group_data = industry_series.index.groupby(industry_series.values)
        industry_list = list(group_data.keys())
        data_series=data.iloc[:,i]
        for industry_name in industry_list:
            industry_temp = data_series.loc[group_data[industry_name]]
            industry_temp = industry_temp.fillna(industry_temp.mean())
            resl_series = resl_series.append(industry_temp)
        stock_list = list(set(data_series.index) - set(industry_series.dropna().index))
        resl_series = resl_series.append(data_series.loc[stock_list])
        data.iloc[:,i]=resl_series
    return data

#只做均值填充
def data_fillna_mean(df,date_list):
    df=fillna_ind_mean(df,date_list)
    return df

#空值填充，行业均值填充
def nonefill(data,date_list):
    tempdata=data
    for i in tempdata:
        i=data_fillna_mean(i,date_list)
    return tempdata

#预测准确率分析
def accu_score(score,name):
    plt.axhline(0.5,color='red')
    plt.plot(score)
    plt.text(score[1],score[1],name,fontdict={'size':'16','color':'black'})
    plt.title("Accuracy on TestData", fontsize=15)

#因子数据合成
import itertools
def change(alldata,data,n,m,M,date_list):
    date=date_list[M:]
    factor_df=pd.DataFrame(columns=date)
    factor_df['stock']=list(alldata[100].index)
    for i,t in itertools.zip_longest(data,date):
        temp=factor_df[['stock']]
        temp[t]=np.nan
        u=i.iloc[:,[n,m]]
        u.columns=[t,'stock']
        factor_Crash=pd.concat([u,temp],join='inner',ignore_index=True)
        factor_Crash.sort_values(t,inplace=True)
        factor_Crash.drop_duplicates(['stock'],inplace=True)
        factor_Crash.sort_values('stock',inplace=True)
        factor_Crash.reset_index(inplace=True)
        factor_df[t]= factor_Crash[t]
    factorF_df=factor_df.set_index(['stock'])
    return factorF_df

def changeindex2(alldata,data,M,date_list):
    factor0_df=change(alldata,data,0,2,M,date_list)
    factor1_df=change(alldata,data,1,2,M,date_list)
    return  factor0_df,factor1_df


#因子T显著度
import util.factortool as ftool
def factor_test_T(factor_list,factor_name):
    risk_test=pd.DataFrame()
    for i ,n in itertools.zip_longest(factor_list,factor_name):
        risk_test[n]=ftool.factor_risk_test_tvalue(i)
    return  risk_test


import alphafactors.factorprepro_class as fp
import alphafactors.factoranalyse as fa

#因子处理(分成两种方向)
def factor_analyse(name,factor):  # 0-positive , 1-negetive
    factor_prepro_obj = fp.FactorPrePro(factor_name=name, factor_data_df=factor, universe='ZZ', neutral_list=None)
    factor_prepro_obj.run_process(start_date=max(factor.columns[0], dt.datetime(2007,1,5)), end_date=factor.columns[-1])
    df = factor_prepro_obj.factor_pro_df
    factor_test_obj = fa.FactorAnalyse(factor_name=name, factor_data_df=df, factor_dr=0)   # 0-positive , 1-negetive
    factor_test_obj.run_analyse_new(start_date=dt.datetime(2009,1,23), universe='ZZ')
    return factor_test_obj

#因子测试画图显示
def show1(factor_test_obj):
    factor_test_obj.net_value_df.iloc[:,-3:].plot(figsize=(20,10))
def show2(factor_test_obj):
    factor_test_obj.factor_perform_df
    return  factor_test_obj.factor_perform_df
def show3(factor_test_obj):
    factor_test_obj.factor_para_df
    return  factor_test_obj.factor_para_df
def show4(factor_test_obj):
    factor_test_obj.port_perform_df
    return     factor_test_obj.port_perform_df
def show5(factor_test_obj):
    factor_test_obj.port_perform_df['annual_return'].plot(kind='bar')
    return factor_test_obj.port_perform_df['annual_return'].plot(kind='bar')   
def show6(factor_test_obj):
    factor_test_obj.factor_index_df['IC值'].plot(kind='bar', figsize=(20,10), color='blue')
    return  factor_test_obj.factor_index_df['IC值'].plot(kind='bar', figsize=(20,10), color='blue')


import util.evalstat as evl


def factor_test_pre(factor):  # 因子中性化预处理
    factor_prepro_obj = fp.FactorPrePro(factor_name='factor_test', factor_data_df=factor, universe='ZZ',
                                        neutral_list=None)
    factor_prepro_obj.run_process(start_date=max(factor.columns[0], dt.datetime(2007, 1, 5)),
                                  end_date=factor.columns[-1])
    df = factor_prepro_obj.factor_pro_df
    return df


def factor_test(stock_weighted_series):
    perform_obj = evl.PortPerform(port_series=stock_weighted_series, ret_type='open', fee=0.0035)
    perform_obj.run()
    return perform_obj


def show01(perform_obj):
    perform_obj.net_value_plot()


def show02(perform_obj):
    perform_obj.get_strategy_perform()
    return perform_obj.get_strategy_perform()


def show03(perform_obj):
    perform_obj.get_avg_turnover()
    return perform_obj.get_avg_turnover()


def show04(perform_obj):
    perform_obj.get_annual_perform()
    return perform_obj.get_annual_perform()

#投资组合构建
def stock_choice(data, num):  # 直接挑选概率值前100支股票，等权
    stock_series = pd.Series()
    for i in data.columns:
        stock_series.loc[i] = pd.Series(index=[data[i].sort_values(ascending=False).head(num).index], data=1 / num)
    stock_choice_obj = factor_test(stock_series)

    return stock_choice_obj


def stock_bench_ind(data, num):  # 行业中性，基准权重后挑选100支股票
    stock_series = pd.Series()
    for i in data.columns:
        set_date = i
        stock_series[i] = get_industry_stock(data[i], set_date, stock_num=num)
    stock_choice_obj = factor_test(stock_series)

    return stock_choice_obj


def factor_stock_choose(factor, num):
    factor_obj = factor_test_pre(factor)  # 做因子预处理
    # 等权选num支
    samew_pre = stock_choice(factor_obj, num)
    samew_unpre = stock_choice(factor, num)  # 不做因子预处理，等权直接选100支
    # 不等权选num支
    unw_pre = stock_bench_ind(factor_obj, num)
    unw_unpre = stock_bench_ind(factor, num)
    return samew_pre, samew_unpre, unw_pre, unw_unpre

#行业基准权重

def get_bench_ind_weight(set_date, bench_code='ZZ500'):
    industry_series = optool.get_series_from_df(data_df=stock_industry_df, set_date=set_date, axis=1).dropna()
    group_data = industry_series.index.groupby(industry_series)
    # 基准行业权重
    bench_component_df = dao.get_index_component_data(bench_code)
    bench_series = pd.Series(bench_component_df[set_date].set_index('code')['weight'])
    bench_series = bench_series / bench_series.sum()
    bench_ind_weight_series = pd.Series(index=stock_industry_list)
    for industry_name in stock_industry_list:
        ind_stock_list = group_data[industry_name]
        temp_series = bench_series.copy()
        bench_ind_weight_series.loc[industry_name] = temp_series.reindex(ind_stock_list).sum()
    bench_ind_weight_series.fillna(0, inplace=True)
    return bench_ind_weight_series


def get_industry_stock(stock_factor_series, set_date, stock_num=100):
    industry_series = optool.get_series_from_df(data_df=stock_industry_df, set_date=set_date, axis=1).dropna()
    stock_series = stock_pool_df[set_date]
    stock_list = industry_series.index.intersection(stock_series.dropna().index).tolist()
    industry_series = industry_series.loc[stock_list]

    group_data = industry_series.index.groupby(industry_series)

    # 基准行业权重
    bench_ind_weight_series = get_bench_ind_weight(set_date=set_date)
    bench_ind_num_series = round(bench_ind_weight_series * stock_num).astype(int)

    port_series = pd.Series()
    # 得到行业中性组合
    for industry_name in stock_industry_list[:]:
        if bench_ind_weight_series[industry_name] <= 0.0:
            continue
        ind_stock_list = group_data[industry_name]
        ind_stock_series = pd.Series(stock_factor_series.loc[ind_stock_list]).reindex(ind_stock_list).sort_values(
            ascending=False)

        ind_stock_num = bench_ind_num_series[industry_name]
        if ind_stock_num < 1:
            ind_stock_num += 1

        if ind_stock_series.shape[0] >= ind_stock_num:
            xx = ind_stock_series.head(ind_stock_num)
            temp_series = xx / xx.sum() * bench_ind_weight_series[industry_name]
        else:
            temp_series = ind_stock_series / ind_stock_series.sum() * bench_ind_weight_series[industry_name]

            temp_series = pd.Series(index=ind_stock_series.index[:ind_stock_num], data=1.0 / ind_stock_num) * \
                          bench_ind_weight_series[industry_name]
        port_series = port_series.append(temp_series)
    return port_series

