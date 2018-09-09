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
json_string = open('../tmp/my_model_architecture.json', 'r').read()
model = model_from_json(json_string)
model.load_weights('../tmp/my_model_weights.h5', by_name=False)
#optimizer = RMSprop(lr=0.01)
#model.compile(loss='categorical_crossentropy', optimizer=optimizer)


import pickle
pkl_file = open('../tmp/df_parameters1.pkl', 'rb')
chars = pickle.load(pkl_file)
r_bins = pickle.load(pkl_file)
df_grouped_mean = pickle.load(pkl_file)
char_indices = pickle.load(pkl_file)
indices_char = pickle.load(pkl_file)
pkl_file.close()

#########---------------------------------

end_date= (datetime.today()-BDay(0)).date()
start_date= (end_date - BDay(90)).date()
Start_date= str(start_date)
End_date=str(end_date)

#aapl = web.DataReader("AAPL", 'yahoo',start_date, end_date)
#Index_new_daily = web.DataReader("000001.SS", 'yahoo',start_date, end_date)
#Index_new_daily = Index_new_daily[-30:]
#tindex=Index_new_daily.index

import tushare as ts

data = ts.get_k_data(code='sh',start =Start_date, end=End_date)

data.columns = pd.Index([x.capitalize() for x in data.columns])
Index_new_daily=data.set_index('Date')


Index_new_daily['Return'] = Index_new_daily.Close/Index_new_daily.Close.shift(1)-1
Index_new_daily.Return = Index_new_daily.Return.fillna(0.0)

r_cat= pd.cut(Index_new_daily.Return,bins=r_bins)
Index_new_daily['r_cat'] = r_cat

Index_new_daily=Index_new_daily.join(df_grouped_mean,on = 'r_cat')


####----------------------------------------


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


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
line_initial = Index_new_daily.reindex(columns=['Close','r_cat','mean_of_group'])[-30:]
##--------- top1 prob and predict--------------        
max_probs=[prob[prob.argmax()]for prob in prob_SHindex] 
    
pred_SHindex
max_probs

line_initial['top_pred']=pred_SHindex
line_initial['top_prob']=max_probs
line_initial['Index_predicted']= np.multiply(line_initial.Close,(1+line_initial.top_pred))



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
        name="Predicted index",
        line=dict(dash ="dash")
        )      
layout = dict(title = 'SHANG HAI INDEX Actual vs predicted',
              xaxis = dict(title = 'Date'),
              yaxis = dict(title = 'Index'),
              )

data =[Actual_Index,Predicted_Index]

fig = dict(data=data, layout=layout)

plotly.offline.plot(fig, filename='../www/index_graph1.html',auto_open=False)


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
