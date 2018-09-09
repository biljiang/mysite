# -*- coding: utf-8 -*-
"""
Created on Mon May 15 10:39:52 2017

@author: Feng
"""

import pandas_datareader.data as web
from datetime import datetime
import pandas as pd
import numpy as np
# BDay is business day, not birthday...
from pandas.tseries.offsets import BDay

### Read df from csv file--------------------

df=pd.read_csv('../data/SH_Index.csv',index_col=0)
df.index=pd.DatetimeIndex(map(lambda x: datetime.strptime(x,'%Y-%m-%d'),df.index))

df['Return'] = df.Close/df.Close.shift(1)-1
df.Return = df.Return.fillna(0.0)

df_clean= df[(np.abs(df.Return) <=0.1)]
r_clean = df_clean.Return

r_cat, r_bins = pd.cut(r_clean,bins=50,retbins=True)
df_clean['r_cat'] = r_cat
df_grouped_mean= df_clean.Return.groupby(df_clean.r_cat).mean()
df_grouped_mean.name='mean_of_group'
df_clean=df_clean.join(df_grouped_mean,on = 'r_cat')

### null check df.mean_of_group[df.mean_of_group.isnull()]

####### RNN practice----------------------------
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys


text = df_clean.mean_of_group

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 30
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 20):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X_train, y_train,
              batch_size=128,
              epochs=1)

    start_index = random.randint(0, len(text) - maxlen - 1)

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
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            #generated += next_char
            #sentence = sentence[1:] + next_char

            #sys.stdout.write(next_char)
            #sys.stdout.flush()
            print(next_char)
        print()
        
### save model
import json
from keras.models import model_from_json
# save model
json_string = model.to_json()
open('../tmp/my_model_architecture.json', 'w').write(json_string)
model.save_weights('../tmp/my_model_weights.h5')

# load model
json_string = open('../tmp/my_model_architecture.json', 'r').read()
model = model_from_json(json_string)
model.load_weights('../tmp/my_model_weights.h5', by_name=False)

## save other parameters method1

df_parameters= {'chars':chars,
                'r_bins':r_bins,
                'df_grouped_mean':df_grouped_mean,
                'char_indices':char_indices,
                'indices_char':indices_char
                }

import pickle
output = open('../tmp/df_parameters.pkl', 'wb')
pickle.dump(df_parameters, output,-1)
output.close()

## reload file from pickle
import pprint, pickle

pkl_file = open('../tmp/df_parameters.pkl', 'rb')
data1 = pickle.load(pkl_file)
pprint.pprint(data1)
pkl_file.close()

### save parameters method2

import pickle
output = open('../tmp/df_parameters1.pkl', 'wb')
pickle.dump(chars, output,-1)
pickle.dump(r_bins, output,-1)
pickle.dump(df_grouped_mean, output,-1)
pickle.dump(char_indices, output,-1)
pickle.dump(indices_char, output,-1)
pickle.dump(X, output,-1)
pickle.dump(y, output,-1)
output.close()
 

import pprint, pickle
pkl_file = open('../tmp/df_parameters1.pkl', 'rb')
chars = pickle.load(pkl_file)
r_bins = pickle.load(pkl_file)
df_grouped_mean = pickle.load(pkl_file)
char_indices = pickle.load(pkl_file)
indices_char = pickle.load(pkl_file)
pkl_file.close()







##### retrain model with Training ,Test set seperated
X_train = X[:5601]
y_train = y[:5601]
X_test =X[5601:]
y_test =y[5601:]

###



### initailization
num_of_days =30

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
            next_index = sample(preds, diversity)
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
line_initial = df_clean.reindex(columns=['Close','r_cat','mean_of_group'])[-30:]
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

plotly.offline.plot(fig, filename='../www/index_graph2.html')


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










