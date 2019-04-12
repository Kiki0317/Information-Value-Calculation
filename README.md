# Information-Value-Calculation
Information Value is important in selecting features for binary classification



class IV_calculation():

    def distribution(data,i,N):
        import pandas as pd
        import os
        from pandas import DataFrame as df
        import numpy as np
        import pandas
        pro = df(data.groupby(i)[i].count().values)/data.shape[0]
        pro_bool = pro.apply(lambda x: x> (1-1/N))
        if True in pro_bool.values:
            skew = 1
        else:
            skew = 0
        return skew

    def num_var(data,i): #变量字符型 or 数字型
        import pandas as pd
        import os
        from pandas import DataFrame as df
        import numpy as np
        import pandas
        uni = data[i].unique()
        bool_func = lambda t: isinstance(t,str)
        uni_bool = np.array([bool_func(xi) for xi in uni])
        if True in uni_bool: #是字符型 用0
            num = 0
        else:
            num = 1
        return num

    def create_IV_TABLE(data,default_rule,N,file_name):
        import pandas as pd
        import os
        from pandas import DataFrame as df
        import numpy as np
        import pandas  #default_rule为Bad的变量名，如'rule104'
        
        binned_var = []
        not_binned_var = []
        writer=pd.ExcelWriter(file_name) 
        
        uni = df(data.columns.values)
        uni['rule_unique_value'] = uni[0].apply(lambda x:len(data[x].unique()))
        not_valid = uni[uni['rule_unique_value'] <=1][0].values
        print('feature unique value <=1',' ',not_valid)
        data.drop(not_valid,axis=1,inplace=True)
        
        for j in data.columns.values:  #判断是否数字型，不是则drop
            if  num_var(data,j) == 0:
                print('data',' ',j,' ','is not number or has unique value')
                data.drop([j],axis=1,inplace=True)
        col_num = data.columns.values  #输出所有数字型Var
        print('We have numerical features:',' ',col_num)
        
        y = data[default_rule]
        data.drop([default_rule],axis=1,inplace=True)  #分隔出y
        data = data.fillna(-1)
        
        for i in data.columns.values:
            if len(data[i].unique()) >= N and distribution(data,i,N) == 0:   #变量不同取值>10 且 无占比90%以上变量
                tmp_data = pd.concat([df(data[i]),y],axis=1)
                data_nan = df(tmp_data[tmp_data[i] == -1]) #分离空值
                data_0 = df(tmp_data[tmp_data[i] == 0])
                data_not_nan = df(tmp_data[(tmp_data[i] != -1)&(tmp_data[i] != 0)])
                
                temp0 = df(pd.qcut(data_not_nan[i],N,duplicates = 'drop'))   #非空值变量分箱，标签为区间
                temp0 = pd.concat([temp0,df(data_not_nan[default_rule])],axis = 1)

                temp0['label'] = temp0[i].cat.codes+1 #非空值变量分箱，标签为1,2,3,4......
                temp0.columns = [i,default_rule,'label']

                Good = df(df(temp0[temp0[default_rule] == 0]).groupby('label')[default_rule].count()) #计算命中（Bad）和不命中（Good）样本
                Bad = df(df(temp0[temp0[default_rule] == 1]).groupby('label')[default_rule].count())

                Good.reset_index(inplace=True)
                Bad.reset_index(inplace=True)
                Bad.columns=['label','Bad']
                Good.columns=['label','Good']

                temp = df(temp0.groupby([i,'label'])[i].count()) #计算每种取值样本数量
                temp.columns = ['count']
                temp.reset_index(inplace=True)
                
                temp['Max'] = temp[i].apply(lambda x: x.right if type(x) == pandas._libs.interval.Interval else x) #每个分类区间边界值（默认前开后闭）
                temp['Min'] = temp[i].apply(lambda x: x.left if type(x) == pandas._libs.interval.Interval else x)

                temp = pd.merge(temp,Bad, on = 'label',how = 'left')
                temp = pd.merge(temp,Good, on = 'label',how = 'left')

                temp.drop([i],axis=1,inplace=True) #计算0变量，保存为倒数第二行，标签为0
                new_col = temp.index.values[-1]
                temp.loc[new_col+1] = 0
                temp.set_value(new_col+1,'count',data_0.shape[0])
                temp.set_value(new_col+1,'label',0)
                temp.set_value(new_col+1,'Good',data_0[data_0[default_rule] == 0].shape[0])
                temp.set_value(new_col+1,'Bad',data_0[data_0[default_rule] == 1].shape[0])       


                new_col = temp.index.values[-1]  #计算缺失值变量，保存为最后一行，标签为-1
                temp.loc[new_col+1] = 0
                temp.set_value(new_col+1,'count',data_nan.shape[0])
                temp.set_value(new_col+1,'label',-1)
                temp.set_value(new_col+1,'Good',data_nan[data_nan[default_rule] == 0].shape[0])
                temp.set_value(new_col+1,'Bad',data_nan[data_nan[default_rule] == 1].shape[0])
                
                temp.sort_values(by = ['label'],inplace=True)

                temp = temp.fillna(0)
                temp['Bad_Rate'] = temp['Bad']/(temp['Bad']+temp['Good']) #计算整个表Bad Rate, Log(Br), Woe, IV值
                temp['Log_it'] = temp['Bad_Rate'].apply(lambda x: np.log(x))
                temp['Woe_nolog'] = (temp['Bad']/temp['Good'])/(temp['Bad'].sum()/temp['Good'].sum())
                temp['Woe'] = np.log((temp['Bad']/temp['Good'])/(temp['Bad'].sum()/temp['Good'].sum()))
                temp['IV'] = (temp['Bad']/temp['Bad'].sum() - temp['Good']/temp['Good'].sum())*temp['Woe']
                temp.drop(['Woe_nolog'],axis=1,inplace=True)
                print('IV calculation completed for',': ',i)
                binned_var = np.append(binned_var,i)
                temp = temp.fillna(0)
                
                temp.replace({-np.inf:np.nan, np.inf:np.nan},inplace=True)
                temp.to_excel(writer,sheet_name = i,encoding = 'utf-8',index=False)

            else: #不分箱变量计算
                tmp_data = pd.concat([df(data[i]),y],axis=1)
                data_nan = df(tmp_data[tmp_data[i] == -1])
                data_not_nan = df(tmp_data[tmp_data[i] != -1])
                
                Good = df(df(data_not_nan[data_not_nan[default_rule] == 0]).groupby(i)[default_rule].count())  
                Bad = df(df(data_not_nan[data_not_nan[default_rule] == 1]).groupby(i)[default_rule].count())
                Good.reset_index(inplace=True)
                Bad.reset_index(inplace=True)
                Bad.columns=['label','Bad']
                Good.columns=['label','Good']
                
                temp = df(data_not_nan.groupby([i])[i].count())
                temp.columns = ['count']
                temp.reset_index(inplace=True)
                temp.columns = ['label','count']
                
                
                temp = pd.merge(temp,Bad, on = 'label',how = 'left')
                temp.columns=['label','count','Bad']
                temp = pd.merge(temp,Good, on = 'label',how = 'left')
                temp.columns=['label','count','Bad','Good']
                temp = temp.fillna(0)
                
                temp['label_s'] = temp['label']
                
                
                if distribution(data,i,N) == 1:
                    temp_resi = temp.loc[temp.index.values[1:]]
                    new_col = temp.index.values[-1]
                    temp.loc[new_col+1] = 0
                    temp.set_value(new_col+1,'label',pd.Interval(left = temp_resi.index.values[0], right = temp_resi.index.values[-1], closed = 'both'))
                    temp.set_value(new_col+1,'count',temp_resi['count'].sum())
                    temp.set_value(new_col+1,'Good',temp_resi['Good'].sum())
                    temp.set_value(new_col+1,'Bad',temp_resi['Bad'].sum())
                    temp.set_value(new_col+1,'label_s',(temp_resi.index.values[0]+temp_resi.index.values[-1])/2)
                    temp = temp.loc[[0,temp.index.values[-1]]]
                    del(temp_resi)
                
                new_col = temp.index.values[-1]
                temp.loc[new_col+1] = 0
                temp.set_value(new_col+1,'count',data_nan.shape[0])
                temp.set_value(new_col+1,'label',-1)
                temp.set_value(new_col+1,'label_s',-1)
                temp.set_value(new_col+1,'Good',data_nan[data_nan[default_rule] == 0].shape[0])
                temp.set_value(new_col+1,'Bad',data_nan[data_nan[default_rule] == 1].shape[0])
                
                
                temp.sort_values(by = ['label_s'],inplace=True)
                
                temp['Bad_Rate'] = temp['Bad']/(temp['Bad']+temp['Good'])
                temp['Log_it'] = temp['Bad_Rate'].apply(lambda x: np.log(x))
                temp['Woe_nolog'] = (temp['Bad']/temp['Good'])/(temp['Bad'].sum()/temp['Good'].sum())
                temp['Woe'] = np.log((temp['Bad']/temp['Good'])/(temp['Bad'].sum()/temp['Good'].sum()))
                temp['IV'] = (temp['Bad']/temp['Bad'].sum() - temp['Good']/temp['Good'].sum())*temp['Woe']
                temp.drop(['Woe_nolog'],axis=1,inplace=True)
                print('IV calculation completed with out bin process for',': ',i)
                not_binned_var = np.append(not_binned_var,i)
                
                temp.replace({-np.inf:np.nan, np.inf:np.nan},inplace=True)

                temp.to_excel(writer,sheet_name = i,encoding = 'utf-8',index=False)	

        return not_valid,col_num,binned_var,not_binned_var,temp

    def SUM_IV_byFEATURE(INPUT,OUTPUT): #计算sum(IV)
        import pandas as pd
        import os
        from pandas import DataFrame as df
        import numpy as np
        import pandas 
        table = pd.read_excel(INPUT,sheet_name=None)
        KEY=[]
        Information_Value=[]
        for key in table:
            IV = table[key].IV.sum()
            KEY = np.append(KEY,key)
            Information_Value = np.append(Information_Value,IV)
            print(key,' ',IV)

    KEY = df(KEY)
    KEY.columns=['var']

    Information_Value=df(Information_Value)
    Information_Value.columns=['IV']

    table = pd.concat([KEY,Information_Value],axis=1)
    table.to_csv(OUTPUT,encoding='utf-8',index=False)

# draw bi-var chart of each feature

def plot(file,key):

    from matplotlib import pyplot as plt
    c=[]
    data = pd.read_excel(file,sheet_name = key)
    for i in range(len(data.label.values)):
        if isinstance(data.label[i],str) == True:
            data.label[i] = (int(data.label.values[i][1])+int(data.label.values[i][4]))*0.5
    data.sort_values(by=['label'],inplace=True)
    for i in range(len(data.label.values)):
        c1 = data[data.label == data.label.values[i]]['count'].values[0]
        c = np.append(c,np.linspace(data.label.values[i],data.label.values[i],c1))
    fig,left_axis=plt.subplots()
    right_axis = left_axis.twinx()
    left_axis.hist(c,histtype='bar',facecolor='pink',rwidth=0.8,alpha=0.75,edgecolor='black')
    right_axis.plot(data.label.values,data.Woe.values,color='black',linewidth = 1,marker='o',markersize=5)
    left_axis.set_xlabel(key.split('_')[-1],fontsize = 15)
    left_axis.set_ylabel('Count',fontsize = 15)
    right_axis.set_ylabel('Weight_of_evidence',fontsize = 15)
    plt.show() 
