# -*- coding: utf-8 -*-
"""
Created on Mon May 15 09:39:59 2017

@author: Feng
"""

### Get data from Yahoo

#import pandas_datareader.data as web
from datetime import datetime
import pandas as pd
# BDay is business day, not birthday...
from pandas.tseries.offsets import BDay
from pandas import DataFrame, Series

#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Activation
#from keras.layers import LSTM
#from keras.optimizers import RMSprop
#from keras.utils.data_utils import get_file
import numpy as np
#import random
#import sys

#import json
from keras.models import model_from_json
# load model
json_string = open('../en/tmp/my_model_architecture_en.json', 'r').read()
model = model_from_json(json_string)
model.load_weights('../en/tmp/my_model_weights_en.h5', by_name=False)
#optimizer = RMSprop(lr=0.01)
#model.compile(loss='categorical_crossentropy', optimizer=optimizer)


import pickle
pkl_file = open('../en/tmp/df_parameters1_en.pkl', 'rb')
chars = pickle.load(pkl_file)
r_bins = pickle.load(pkl_file)
df_grouped_mean = pickle.load(pkl_file)
char_indices = pickle.load(pkl_file)
indices_char = pickle.load(pkl_file)
pkl_file.close()

#########---------------------------------

import pytz
tz_US_E = pytz.timezone('US/Eastern')

end_date= (datetime.now(tz=tz_US_E)-BDay(0)).date()
start_date= (end_date - BDay(90)).date()
#Start_date= str(start_date)
#End_date=str(end_date)
import pandas_datareader.data as web
data = web.DataReader("SP500", "fred",start_date,end_date)
data.columns=pd.Index(['Close'])
Index_new_daily = data.dropna()

'''
import tushare as ts

data = ts.get_k_data(code='sh',start =Start_date, end=End_date)

data.columns = pd.Index([x.capitalize() for x in data.columns])
data.Date = data.Date.apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
data.Date = data.Date.apply(datetime.date)


'''



#Index_new_daily=data.set_index('Date')


Index_new_daily['Return'] = Index_new_daily.Close/Index_new_daily.Close.shift(1)-1
Index_new_daily.Return = Index_new_daily.Return.fillna(0.0)

r_cat= pd.cut(Index_new_daily.Return,bins=r_bins)
Index_new_daily['r_cat'] = r_cat

Index_new_daily=Index_new_daily.join(df_grouped_mean,on = 'r_cat')


####----------------------------------------

'''
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
'''

####----------------------
### initailization
text = Index_new_daily.mean_of_group

num_of_days =30
maxlen=30
pred_SHindex=[]
prob_SHindex=[]
for start_index in range((len(text)-maxlen-num_of_days),(len(text)-maxlen)):
    for diversity in [1.0]:
        print()
        print('----- diversity:', diversity)

        ###generated = []
        sentence = text[start_index: start_index + maxlen]

        #generated += sentence
        #print('----- Generating with seed: "' + sentence + '"')
        #sys.stdout.write(generated)
        ###generated.append(sentence)
        print ('---------------------generating with seed------------------------')
        ###print (generated)
        
        for i in range(1):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            
            #next_index = sample(preds, diversity)
            next_index = np.argmax(preds)
            next_char = indices_char[next_index]
            pred_SHindex.append(next_char)
            prob_SHindex.append(preds)
            #generated += next_char
            #sentence = sentence[1:] + next_char

            #sys.stdout.write(next_char)
            #sys.stdout.flush()
            print(next_char)
        print()


##### preparing graph------------------------
line_initial = Index_new_daily.reindex(columns=['Close','Return','r_cat','mean_of_group'])[-30:]
##--------- top1 prob and predict--------------        
max_probs=[prob[prob.argmax()]for prob in prob_SHindex] 
last_close = Index_new_daily.Close[-31:-1]    

#pred_SHindex
#max_probs

line_initial['top_pred']=pred_SHindex
line_initial['top_prob']=max_probs
line_initial['Index_predicted']= np.multiply(last_close.values,(1+line_initial.top_pred))
#line_initial['Index_predicted']= np.multiply(line_initial.Close,(1+line_initial.top_pred))

#duplicate and save line_initial table as is
predicted_table=line_initial.copy()

tracking_file = open('../en/data/tracking_record_en.pkl', 'rb')
tracking_record=pickle.load(tracking_file)
tracking_file.close()

tracking_record=pd.concat([tracking_record,predicted_table])
tracking_record=tracking_record.drop_duplicates()
tracking_record=tracking_record.dropna()
output = open('../en/data/tracking_record_en.pkl', 'wb')
pickle.dump(tracking_record, output)
output.close()
tracking_record.to_csv('../en/data/tracking_record_en.csv')


#making new 5 days forecast
last_Bday= Index_new_daily.index[-1]
f_start_date=(last_Bday + BDay(0)).date()
f_end_date = (last_Bday + BDay(5)).date()
f_day_range = pd.date_range(f_start_date, f_end_date,freq='B')
f_day_range.date

sentence= list(text[-30:])

### making table to show
top1_preds=[]
top1_probs=[]
forecast_probs=[]

for i in range(5):
    x = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.

    preds = model.predict(x, verbose=0)[0]
            
    #next_index = sample(preds, diversity)
    top1_prob=preds.max()
    next_index = np.argmax(preds)
    next_char = indices_char[next_index]
    top1_preds.append(next_char)
    top1_probs.append(top1_prob)
    forecast_probs.append(preds)
    print(next_char,top1_prob)
    sentence = sentence[1:] + [next_char]
print()

top2_preds=[]
top2_probs=[]

for preds in forecast_probs:
    li=list(preds)
    top2_index=li.index(sorted(li)[-2])       
    top2_prob =li[top2_index]
    top2_pred = indices_char[top2_index]
    top2_probs.append(top2_prob)
    top2_preds.append(top2_pred)
    print(top2_pred,top2_prob)
print()

Index_forecast=[]
last_index= Index_new_daily.Close[-1]
Index_forecast.append(last_index)
for i in range(5):
    last_index=last_index*(1+top1_preds[i])
    Index_forecast.append(last_index)

## make the forecast 5 days line

forecast_series= Series(data=Index_forecast, index=f_day_range.date,name='Forecast_index')

line_initial = pd.concat([line_initial,forecast_series],axis=1)

ftable_index=pd.Index(f_day_range.date[1:])
Forecast_table = DataFrame(index=ftable_index,
                           data={'Forecast_S&P500':["%.2f" %x for x in Index_forecast[1:]],
                                 'Probable Return#1':["%.2f%%" %(x*100) for x in top1_preds],
                                 'Probability of R#1':["%.2f%%" %(x*100) for x in top1_probs],
                                 'Probable Return#2':["%.2f%%" %(x*100) for x in top2_preds],
                                 'Probability of R#2':["%.2f%%" %(x*100) for x in top2_probs]})
Forecast_table=Forecast_table.reindex(
        columns=['Forecast_S&P500','Probable Return#1','Probability of R#1','Probable Return#2','Probability of R#2'])

table_css = open('../tmp/table_css.css', 'r').read()
h=table_css + Forecast_table.to_html(classes='table',index=True)
open('../en/www/my_table_en.html', 'w').write(h)


import plotly
#import plotly.plotly as py
import pandas as pd
#from plotly.graph_objs import *
import plotly.graph_objs as go
#from plotly.offline import plot

Actual_Index = go.Scatter (
        x=line_initial.index,
        y=line_initial["Close"],
        mode="lines+markers",
        name="Actual INDEX"
        )   
Predicted_Index = go.Scatter (
        x=line_initial.index,
        y=line_initial["Index_predicted"],
        mode="lines+markers",
        name="Predicted Index",
        line=dict(dash ="dash")
        ) 
Forecast_Index  = go.Scatter (
        x=line_initial.index,
        y=line_initial["Forecast_index"],
        mode="lines+markers",
        name="Forecast Index",
        line=dict(dash ="dot")
        ) 

     
layout = dict(title = 'S&P500 Actual vs Predicted',
              xaxis = dict(title = 'Date'),
              yaxis = dict(title = 'Index'),
              )

data =[Actual_Index,Predicted_Index,Forecast_Index]

fig = dict(data=data, layout=layout)

plot_str = plotly.offline.plot(fig, output_type='div') 
open('../en/www/index_graph1_en.html','w').write(plot_str)
#plotly.offline.plot(fig, filename='../www/index_graph1.html',auto_open=False)

'''
top2_probs=[]
top2_preds=[]
for l in prob_SHindex:
    li=list(l)
    top2_index=li.index(sorted(li)[-2])       
    top2_prob =li[top2_index]
    top2_pred = indices_char[top2_index]
    top2_probs.append(top2_prob)
    top2_preds.append(top2_pred)
'''

########-----------------------
'''
for x in a.columns:
    print(a[x].equals(b[x]))
'''
