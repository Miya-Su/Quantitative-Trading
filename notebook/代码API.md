# 代码函数说明

1. def read_factor_data(factors_name_list, trade_date_20d_series)

   - [ ] `def risk_factor(alpha_factor_series)`

     读取十个风险因子，加入Pandas Series中，完成数据构建

   - [ ] `def fillna_ind_mean(data,date_list)`

     按照行业均值填充空值

   - [ ] `def data_fillna_mean(data,date_list)`

     对Series中的每一个特征数据的DataFrame进行缺失值填充

2. def nonefill(data,date_list)

   - [ ] `def data_input_class(data,date_list,T,x)`

     - [ ] `def concat_data(data,date_list,is_rise_df)`

     - [ ] `def rise_fall_class(trade_date_m_series)`

     - [ ] `def data_train_test(data_pct,x)`

   - [ ] `def changeindex2(data,M,date_list)`
     - [ ] `def change(data,n,m,M,date_list)`
   - [ ] `def changeindex3(data,M,date_list)`
   - [ ] `def accu_score(score,name)`

3. def predict_probo_class(data,model,loop,date_list，T,x,changeindex_model)

   - [ ] 

4. def predict_probo_reg(data,model,loop,date_list)

   - [ ] `def data_input_reg(data,date_list)`
   - [ ] `def splitdata(train,test,i,j,x,y)`
     - [ ] `def countsum(data):`
     - [ ] `def standard(X_train,X_test)`
   - [ ] `def changeindex1(data,M,date_list)`
     - [ ] `def change(data,n,m,M,date_list)`

5. class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin)

   - [ ] `def rmsle_cv(model,data,M)`
   - [ ] `def rmsle(model,data,loop,date_list)`
   - [ ] `def rmse_show(rmse_list,label_list)`

6. class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin)

   - [ ] `def rmsle_cv(model,data,M)`
   - [ ] `def rmsle(model,data,loop,date_list)`
   - [ ] `def rmse_show(rmse_list,label_list)`

7. def factor_test_T(factor_list,factor_name)

   - [ ] 

8. def factor_analyse(name,factor)

   - [ ] `def show1(factor_test_obj)`
   - [ ] `def show2(factor_test_obj)`
   - [ ] `def show3(factor_test_obj)`
   - [ ] `def show4(factor_test_obj)`
   - [ ] `def show5(factor_test_obj)`
   - [ ] `def show6(factor_test_obj)`

9. def factor_stock_choose(factor,num)

   - [ ] `def factor_test_pre(factor)`
   - [ ] `def stock_choice(data,num)`

   - [ ] `def stock_bench_ind(data,num)`
     - [ ] `def get_industry_stock(stock_factor_series, set_date, stock_num=100)`
     - [ ] `def get_bench_ind_weight(set_date, bench_code='ZZ500')`
   - [ ] `def show01(perform_obj)`
   - [ ] `def show02(perform_obj)`
   - [ ] `def show03(perform_obj)`
   - [ ] `def show04(perform_obj)`