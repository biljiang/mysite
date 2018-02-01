# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:49:14 2017

@author: Feng
"""

#import lib_stockselect as l_s
from .lib_stockselect import name_dict#, code_dict
from . import lib_stockselect as l_s
from . import lib_stockplot as l_plt
import pandas as pd
from datetime import time,datetime
import matplotlib.pyplot as plt
import numpy as np
#from django.core.files import File

def get_data_fenbi(code=None, date = None):
    s= code; d=date
    filename_csv = '/home/bill/datadisk/shared/mysite/static/csv_files/s'+s+'d'+d.replace('-','')+'.csv'
########## getting data in any DB
    try:
        data =pd.read_csv(filename_csv,encoding='utf-8')
        data.idx_time=data.idx_time.apply(lambda x: datetime.strptime(x,'%H:%M:%S').time())
        data = data.set_index('idx_time')
    except:
        try:
            if d == str(l_s.Last_BD()):
                try:
                    data = l_s.stock_today(s)
                    if datetime.now() > time(15,3,0):
                        data.to_csv(filename_csv)
                except:
                    print("Get today's data failed")
            else:
                try:
                    data = l_s.stock_dt(s,d)
                    data.to_csv(filename_csv,encoding ='utf-8')
                except:
                    print('Get data failed after searching 3 DB')                
        except:
            try:
                data = l_s.stock_dt(s,d)
                data.to_csv(filename_csv, encoding ='utf-8')
            except:
                print('Get data failed after searching web')

    return data

def data_split(df = None):
   # if not df.empty:
    df = df.drop_duplicates()
    df=df[df.volume != 0]
    df_O = df[df.index < time(9,30,0)]
    df_O = df_O[df_O.time.duplicated().apply(lambda x: not x)]
    df_E = df[df.index > time(15,0,0)]
    df_E = df_E[df_E.time.duplicated().apply(lambda x: not x)]
    df=df[(df.index>=time(9,30,0)) & (df.index <=time(15,0,0))]
    df=df[df.time.duplicated().apply(lambda x: not x)]
    df_B = df[df.type == '买盘']
    df_S = df[df.type == '卖盘']
    return df,df_B,df_S,df_O,df_E



def stockDF_graph(df=None,code=None,date=None):
    df,df_B,df_S,df_O,df_E = data_split (df)
    # prepare canvas############
    fig = plt.figure(figsize=(15,15))
    ax1 = plt.axes([0.1,0.6,0.8,0.3])
    ax2 = plt.axes([0.1,0.39,0.8,0.2])
    ax3 = plt.axes([0.1,0.1,0.36,0.24])
    ax4 = plt.axes([0.54,0.1,0.36,0.24])
    ax1.set_title('Stock '+code+' on '+date)
    ax1.set_xticklabels([])
    ax1.plot(df.price)
    data_B = pd.concat([df.time,df_B.volume],axis =1)
    data_B = data_B.fillna(0.0)
    ax2.plot(data_B.volume,color='red',alpha =0.5)
    data_S = pd.concat([df.time,df_S.volume],axis =1)
    data_S = data_S.fillna(0.0)
    ax2.plot(data_S.volume,color='green',alpha =0.3)
    
    ax3.set_title('Price Distribution of '+code+' on '+date)
    ax4.set_title('Amount Distribution of '+code+' on '+date)
    B_total =0; S_total = 0; amount_total=df.amount.sum()
    if len(df_B)!= 0 :
        B_total=df_B.amount.sum()
        wght_B=np.array([x/B_total for x in df_B.amount])
        l_plt.axes_kde_plot(df_B.price,wght_B,
                            color='orange',ratio = B_total/amount_total,axes= ax3)
        l_plt.axes_kde_plot(np.log10(df_B.amount),wght_B,bw=0.05,color='orange',
                            ratio = B_total/amount_total,axes = ax4)
    if len(df_S)!= 0 :
        S_total = df_S.amount.sum()
        wght_S=np.array([x/S_total for x in df_S.amount])        
        l_plt.axes_kde_plot(df_S.price,wght_S,
                            color='green',ratio = S_total/amount_total,axes=ax3)   
        l_plt.axes_kde_plot(np.log10(df_S.amount),wght_S,bw=0.05,color='green',
                            ratio = S_total/amount_total,axes = ax4)
    return fig
   

if __name__=='__main__':
    d='2017-06-22';s='603767'
    data = get_data_fenbi(s,d)
    tu = data_split(data)
    fig = stockDF_graph(data,s,d)

    print('-'*60)
    print('Stock '+s+' on date '+d+': Buying= {0:,}  Selling= {1:,}'.format(
            tu[1].amount.sum(),tu[2].amount.sum()))
    print('Open amount :{:,}   Close amount :{:,}'.format(tu[3].amount.sum(),tu[4].amount.sum()))
    print('Stock name: {}   Code: {}'.format(name_dict[s],s))
    print(data[:5])
    print(data[-5:])
    fig.show()




'''
    df['amount_B'] = [df.amount[i] if (df.type[i] =='买盘') else 0 for i in range(len(df))]
    df['amount_S'] = [df.amount[i] if (df.type[i] =='卖盘') else 0 for i in range(len(df))]
    filename_csv = '../csv_files/s'+s+'d'+d.replace('-','')+'.csv'
    df.to_csv(filename_csv)


fig = plt.figure(figsize=(8,6))
ax1 = plt.subplot2grid((5,1), (0,0), rowspan=2)
ax1.set_xticklabels([])
ax2 = plt.subplot2grid((5,1), (2,0))
#fig.subplots_adjust(hspace=0.02)
ax3 = plt.subplot2grid((5,2), (3,0), rowspan=2)
ax4 = plt.subplot2grid((5,2), (3,1), rowspan=2)

pos4 = ax4.get_position()

ax4.set_position([0.547,0.11,0.352,0.25])
ax4.set_title("my test")

plt.savefig('../www/fig1.png')

fig = plt.figure(figsize=(10,10))

ax1 = plt.axes([0.1,0.6,0.8,0.3])
ax1.set_xticklabels([])
ax2 = plt.axes([0.1,0.39,0.8,0.2])
ax3 = plt.axes([0.1,0.1,0.36,0.24])
ax4 = plt.axes([0.54,0.1,0.36,0.24])
'''
