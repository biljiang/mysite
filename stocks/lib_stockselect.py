# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#import seaborn as sns
import tushare as ts

#import matplotlib.pyplot as plt
from datetime import datetime
from pandas.tseries.offsets import BDay
import pandas as pd
from pandas import DataFrame #, Series
#import numpy as np

#from statsmodels.nonparametric.kde import KDEUnivariate
#from statsmodels.nonparametric.kernel_density import KDEMultivariate

basic_info = ts.get_stock_basics()

name_dict = dict((c,n) for c,n in zip(basic_info.index,basic_info.name))
code_dict = dict((n,c) for c,n in zip(basic_info.index,basic_info.name))


def Last_BD(date = datetime.today().date()):
    if date.isoweekday() in [6,7]:
        date=(date - BDay(1)).date()
    else:
        date=(date - BDay(0)).date()
    return date

def get_Ln_BD(end_date = datetime.today().date(),n_days=5):
    if end_date.isoweekday() in [6,7]:
        end_date=(end_date - BDay(1)).date()
    else:
        end_date=(end_date - BDay(0)).date()
    B_day_range = pd.date_range((end_date-BDay(n_days-1)).date(), end_date,freq='B')
    return B_day_range
    
def get_stock_nday(code = None ,date_range=None):
    df = DataFrame()
    for x in date_range:
        data= ts.get_tick_data(code,date=x)
        data['date']=x
        df = pd.concat([df,data])
    df['date_time']=(df.date + ' ' +df.time).apply(
            lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
    df = df.set_index('date_time')
    return df
'''
def stock_tick_plot(code = None, date = None):
    df = ts.get_tick_data(code,date)
    if len(df)==0:
        return
    df['date']=date
    df['date_time']=(df.date + ' ' +df.time).apply(
            lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
    df=df.sort_values('time')
    df = df.set_index('date_time')    
    df_B = df[df.type == '买盘']
    df_S = df[df.type == '卖盘']
    f1=plt.figure('volume')
    df.volume.plot()
    f2=plt.figure('price')
    df.price.plot()
    f3=plt.figure('dist')
    if len(df_B)!= 0 :
        sns.kdeplot(np.log10(df_B.amount), bw=0.1,color='orange')
    if len(df_S)!= 0 :
        sns.kdeplot(np.log10(df_S.amount), bw=0.1,color='green') 
    plt.close('all')
    return df,f1,f2,f3
'''
def stock_dt(code = None, date = None, pause =0):
    df = ts.get_tick_data(code,date,pause=pause)
    if (len(df)==0) | (df.price.isnull().all()):
        print('None data gotten!')
        return
    df['date']=date
    df['date_time']=(df.date + ' ' +df.time).apply(
            lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
    df['idx_time'] = df['time']
    df['idx_time']=df.time.apply(lambda x: datetime.strptime(x,'%H:%M:%S').time())
    df=df.sort_values('idx_time')
    df = df.set_index('idx_time')
    return df
'''
def kde_plot(x,weights,bw=0.02,color= None,ratio =1.0):
    x_grid = np.arange(x.min()-0.5, x.max()+0.5 , 0.02)
    kde=KDEUnivariate(x)
    kde.fit(bw=bw,fft= False, weights=weights)
    x_value = kde.evaluate(x_grid)*ratio
    plt.plot(x_grid,x_value,color=color)
    return

def stock_dt_plot(code = None, date = None):
    try:
        df = pd.read_hdf('../dtdb/zxs'+code+'.h5',('d'+date.replace('-','_')))
    except:
        try:
            df = stock_dt(code= code, date=date)
        except:
            print('Getting data failed!')
            return
    df_B = df[df.type == '买盘']
    df_B = df_B[df_B.amount != 0]
    df_S = df[df.type == '卖盘']
    df_S = df_S[df_S.amount != 0]
    f1=plt.figure('buying volume')
    try:
        df_B.volume.plot(color='orange')
    except:
        pass
    f2=plt.figure('selling volume')    
    try:
        df_S.volume.plot(color='green')
    except:
        pass
    f3=plt.figure('price')
    df.price.plot()
    f4=plt.figure('dist_price')
    B_total =0; S_total = 0; amount_total=df.amount.sum()
    if len(df_B)!= 0 :
        B_total=df_B.amount.sum()
        wght_B=np.array([x/B_total for x in df_B.amount])
        kde_plot(df_B.price,wght_B,color='orange',ratio = B_total/amount_total)
    if len(df_S)!= 0 :
        S_total = df_S.amount.sum()
        wght_S=np.array([x/S_total for x in df_S.amount])        
        kde_plot(df_S.price,wght_S,color='green',ratio = S_total/amount_total)
    f5=plt.figure('dist_amount')
    B_total =0; S_total = 0; amount_total=df.amount.sum()
    if len(df_B)!= 0 :
        B_total=df_B.amount.sum()
        wght_B=np.array([x/B_total for x in df_B.amount])
        kde_plot(np.log10(df_B.amount),wght_B,bw=0.1,color='orange',ratio = B_total/amount_total)
        #sns.kdeplot(np.log10(df_B.amount), bw=0.1,color='orange')
    if len(df_S)!= 0 :
        S_total = df_S.amount.sum()
        wght_S=np.array([x/S_total for x in df_S.amount])        
        kde_plot(np.log10(df_S.amount),wght_S,bw=0.1,color='green',ratio = S_total/amount_total)
        #sns.kdeplot(np.log10(df_S.amount), bw=0.1,color='green')     
    plt.close('all')
    return df,f1,f2,f3,f4,f5,B_total,S_total
'''
### get today plot---------------------

def stock_today(code = None, pause =0):
    df = ts.get_today_ticks(code,pause=pause)
    if (len(df)==0) | (df.price.isnull().all()):
        print('No data gotten!')
        return
    df['date']=str(datetime.today().date())
    df['date_time']=(df.date + ' ' +df.time).apply(
            lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
    df['idx_time'] = df['time']
    df['idx_time']=df.time.apply(lambda x: datetime.strptime(x,'%H:%M:%S').time())
    df=df.sort_values('idx_time')
    df = df.set_index('idx_time')
    return df

'''
def stock_today_plot(code = None,date = None):
    try:
        df = pd.read_hdf('../dtdb/d'+date.replace('-','_')+'.h5','zxs'+code)
    except:
        try:
            df = stock_today(code)
        except:
            print('Getting data failed!')
            return
    df_B = df[df.type == '买盘']
    df_B = df_B[df_B.amount != 0]
    df_S = df[df.type == '卖盘']
    df_S = df_S[df_S.amount != 0]
    f1=plt.figure('buying volume')
    try:
        df_B.volume.plot(color='orange',title = code)
    except:
        pass
    f2=plt.figure('selling volume')    
    try:
        df_S.volume.plot(color='green',title = code)
    except:
        pass
    f3=plt.figure('price')
    df.price.plot(title =code)
    f4=plt.figure(code+' dist_price')
    B_total =0; S_total = 0; amount_total=df.amount.sum()
    if len(df_B)!= 0 :
        B_total=df_B.amount.sum()
        wght_B=np.array([x/B_total for x in df_B.amount])
        kde_plot(df_B.price,wght_B,color='orange',ratio = B_total/amount_total)
    if len(df_S)!= 0 :
        S_total = df_S.amount.sum()
        wght_S=np.array([x/S_total for x in df_S.amount])        
        kde_plot(df_S.price,wght_S,color='green',ratio = S_total/amount_total)

    f5=plt.figure(code +' dist_amount')
    B_total =0; S_total = 0; amount_total=df.amount.sum()
    if len(df_B)!= 0 :
        B_total=df_B.amount.sum()
        wght_B=np.array([x/B_total for x in df_B.amount])
        kde_plot(np.log10(df_B.amount),wght_B,bw=0.1,color='orange',ratio = B_total/amount_total)
        #sns.kdeplot(np.log10(df_B.amount), bw=0.1,color='orange')
    if len(df_S)!= 0 :
        S_total = df_S.amount.sum()
        wght_S=np.array([x/S_total for x in df_S.amount])        
        kde_plot(np.log10(df_S.amount),wght_S,bw=0.1,color='green',ratio = S_total/amount_total)
        #sns.kdeplot(np.log10(df_S.amount), bw=0.1,color='green')     
    
    plt.close('all')
    return df,f1,f2,f3,f4,f5,B_total,S_total
'''
def get_today_quotes(codelist = None,n=25):
# scanning the whole zxc market quotes asof the time point of running
    df=DataFrame()
    for i in range(len(codelist)//n+1):
        #print(codelist[i*n:(i+1)*n])
        data1 = ts.get_realtime_quotes(codelist[i*n:(i+1)*n])    
        df=pd.concat([df,data1])
        
# data Wrangling: fillna, change txt to numeric and sort  
    df[df=='']='0'
    for ix in ['volume','b1_v','b2_v','b3_v','b4_v','b5_v','a1_v','a2_v','a3_v','a4_v','a5_v']:
        df[ix] = df[ix].apply(int)
    
    for ix in ['open','pre_close','price','high','low','bid','ask','amount',
               'b1_p','b2_p','b3_p','b4_p','b5_p','a1_p','a2_p','a3_p','a4_p','a5_p']:
        df[ix] = df[ix].apply(float)       
    df['pchange']=(df.price/df.pre_close-1)*100    
    df = df.sort_values('pchange',ascending=False)
    df = df.set_index('code')
    return df    

