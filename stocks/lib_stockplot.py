# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import seaborn as sns
import tushare as ts

import matplotlib.pyplot as plt
from datetime import datetime,time
#from pandas.tseries.offsets import BDay
import pandas as pd
#from pandas import DataFrame, Series
import numpy as np
from . import lib_stockselect as l_s

from statsmodels.nonparametric.kde import KDEUnivariate

#from statsmodels.nonparametric.kernel_density import KDEMultivariate

'''
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
    df=df.sort_values('time')
    df = df.set_index('date_time')
    return df
'''
def kde_plot(x,weights,bw=0.02,color= None,ratio =1.0):
    x_grid = np.arange(x.min()-0.5, x.max()+0.5 , 0.02)
    kde=KDEUnivariate(x)
    kde.fit(bw=bw,fft= False, weights=weights)
    x_value = kde.evaluate(x_grid)*ratio
    plt.plot(x_grid,x_value,color=color)
    return

def axes_kde_plot(x,weights,bw=0.02,color= None,ratio =1.0,axes = None):
    x_grid = np.arange(x.min()-0.5, x.max()+0.5 , 0.02)
    kde=KDEUnivariate(x)
    kde.fit(bw=bw,fft= False, weights=weights)
    x_value = kde.evaluate(x_grid)*ratio
    axes.plot(x_grid,x_value,color=color)
    return

def stock_dt_plot(code = None, date = None):
    try:
        df = pd.read_hdf('../dtdb/zxs'+code+'.h5',('d'+date.replace('-','_')))
    except:
        try:
            df = l_s.stock_dt(code= code, date=date)
        except:
            print('Getting data failed!')
            return
    df=df[df.time.duplicated().apply(lambda x: not x)]
    df_B = df[df.type == '买盘']
    df_B = df_B[df_B.amount != 0]
    df_S = df[df.type == '卖盘']
    df_S = df_S[df_S.amount != 0]
    # plot price detail-------------------
    f1, axes = plt.subplots(2, 1,sharex=True,
                            gridspec_kw = {'height_ratios':[2, 1]},
                            num='Stock Daily Details')
    f1.suptitle('Stock '+code+' on '+date)
    f1.subplots_adjust(hspace=0.02)
    try:
        axes[0].plot(df.price)
        #axes[0].set_xticklabels([])
        axes[0].set_xlabel('')
        data_B = pd.concat([df.time,df_B.volume],axis =1)
        data_B = data_B.fillna(0.0)
        axes[1].plot(data_B.volume,color='orange',alpha =0.5)
        #data_B.volume.plot(color='orange',alpha =0.3)
        data_S = pd.concat([df.time,df_S.volume],axis =1)
        data_S = data_S.fillna(0.0)
        axes[1].plot(data_S.volume,color='green',alpha =0.3)
        #data_S.volume.plot(color='green',alpha =0.3)
    except:
        pass
    # plot price distribution--------------------
    f2=plt.figure('Price Distribution')
    ax1=f2.add_subplot(1,1,1)
    ax1.set_title('Price Distribution of '+code+' on '+date)
    B_total =0; S_total = 0; amount_total=df.amount.sum()
    if len(df_B)!= 0 :
        B_total=df_B.amount.sum()
        wght_B=np.array([x/B_total for x in df_B.amount])
        kde_plot(df_B.price,wght_B,color='orange',ratio = B_total/amount_total)
    if len(df_S)!= 0 :
        S_total = df_S.amount.sum()
        wght_S=np.array([x/S_total for x in df_S.amount])        
        kde_plot(df_S.price,wght_S,color='green',ratio = S_total/amount_total)
    # plot amount distribution-----------------------
    f3=plt.figure('dist_amount')
    ax2=f3.add_subplot(1,1,1)
    ax2.set_title('Amount Distribution of '+code+' on '+date)    
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
    return df,df_B,df_S,f1,f2,f3


'''
### get today data--------------------

def stock_today(code = None, pause =0):
    df = ts.get_today_ticks(code,pause=pause)
    if (len(df)==0) | (df.price.isnull().all()):
        print('No data gotten!')
        return
    df['date']=str(datetime.today().date())
    df['date_time']=(df.date + ' ' +df.time).apply(
            lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
    df=df.sort_values('time')
    df = df.set_index('date_time')
    return df
'''


def stock_today_plot(code = None,date = None):
    try:
        df = pd.read_hdf('../dtdb/d'+date.replace('-','_')+'.h5','zxs'+code)
    except:
        try:
            df = l_s.stock_today(code)
        except:
            print('Getting data failed!')
            return
    df=df[df.time.duplicated().apply(lambda x: not x)]    
    df_B = df[df.type == '买盘']
    df_B = df_B[df_B.amount != 0]
    df_S = df[df.type == '卖盘']
    df_S = df_S[df_S.amount != 0]
    f1, axes = plt.subplots(2, 1,sharex=True,
                            gridspec_kw = {'height_ratios':[2, 1]},
                            num = 'Stock Daily Details')
    f1.suptitle('Stock '+code+' on '+date)
    f1.subplots_adjust(hspace=0.02)    
    try:
        axes[0].plot(df.price)
        axes[0].set_xlabel('')
        data_B = pd.concat([df.time,df_B.volume],axis =1)
        data_B = data_B.fillna(0.0)
        axes[1].plot(data_B.volume,color='orange',alpha =0.5)
        data_S = pd.concat([df.time,df_S.volume],axis =1)
        data_S = data_S.fillna(0.0)
        axes[1].plot(data_S.volume,color='green',alpha =0.3)
    except:
        pass

    f2=plt.figure('Price Distribution')
    ax1=f2.add_subplot(1,1,1)
    ax1.set_title('Price Distribution of '+code+' on '+date)
    B_total =0; S_total = 0; amount_total=df.amount.sum()
    if len(df_B)!= 0 :
        B_total=df_B.amount.sum()
        wght_B=np.array([x/B_total for x in df_B.amount])
        kde_plot(df_B.price,wght_B,color='orange',ratio = B_total/amount_total)
    if len(df_S)!= 0 :
        S_total = df_S.amount.sum()
        wght_S=np.array([x/S_total for x in df_S.amount])        
        kde_plot(df_S.price,wght_S,color='green',ratio = S_total/amount_total)
    # plot amount distribution-----------------------
    f3=plt.figure('dist_amount')
    ax2=f3.add_subplot(1,1,1)
    ax2.set_title('Amount Distribution of '+code+' on '+date)  
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
    return df,df_B,df_S,f1,f2,f3

def stockDF_plot(df=None,code=None,date=None):
    df=df[(df.index>=time(9,30,0)) & (df.index <=time(15,0,0))]
    df=df[df.volume != 0]
    df=df[df.time.duplicated().apply(lambda x: not x)]
    df_B = df[df.type == '买盘']
    df_B = df_B[df_B.amount != 0]
    df_S = df[df.type == '卖盘']
    df_S = df_S[df_S.amount != 0]
    f1, axes = plt.subplots(2, 1,sharex=True,
                            gridspec_kw = {'height_ratios':[2, 1]},
                            num = 'Stock Daily Details')
    f1.suptitle('Stock '+code+' on '+date)
    f1.subplots_adjust(hspace=0.02)    
    try:
        axes[0].plot(df.price)
        axes[0].set_xlabel('')
        data_B = pd.concat([df.time,df_B.volume],axis =1)
        data_B = data_B.fillna(0.0)
        axes[1].plot(data_B.volume,color='orange',alpha =0.5)
        data_S = pd.concat([df.time,df_S.volume],axis =1)
        data_S = data_S.fillna(0.0)
        axes[1].plot(data_S.volume,color='green',alpha =0.3)
    except:
        pass

    f2=plt.figure('Price Distribution')
    ax1=f2.add_subplot(1,1,1)
    ax1.set_title('Price Distribution of '+code+' on '+date)
    B_total =0; S_total = 0; amount_total=df.amount.sum()
    if len(df_B)!= 0 :
        B_total=df_B.amount.sum()
        wght_B=np.array([x/B_total for x in df_B.amount])
        kde_plot(df_B.price,wght_B,color='orange',ratio = B_total/amount_total)
    if len(df_S)!= 0 :
        S_total = df_S.amount.sum()
        wght_S=np.array([x/S_total for x in df_S.amount])        
        kde_plot(df_S.price,wght_S,color='green',ratio = S_total/amount_total)
    # plot amount distribution-----------------------
    f3=plt.figure('dist_amount')
    ax2=f3.add_subplot(1,1,1)
    ax2.set_title('Amount Distribution of '+code+' on '+date)  
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
    return df,df_B,df_S,f1,f2,f3  


  
