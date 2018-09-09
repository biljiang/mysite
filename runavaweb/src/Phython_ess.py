# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 18:59:14 2016

@author: Feng
"""

import numpy as np
import pylab as pl
import math
from pandas import DataFrame, Series
import pandas as pd
import datetime as dt
from datetime import datetime

t = [x/1048.0 for x in range(1048)]
y = [2.0 + 3.0 * math.cos(2.0 * math.pi * 50 * t0 - math.pi * 30/180) + \
     1.5 * math.cos(2.0 * math.pi * 75 * t0 + math.pi * 90/180) + \
     1.0 * math.cos(2.0 * math.pi * 150 * t0 + math.pi * 120/180) + \
     2.0 * math.cos(2.0 * math.pi * 220 * t0 + math.pi * 30/180) for t0 in t ]

pl.plot(t,y)
pl.show()

N=len(t) # 采样点数
fs=1048.0 # 采样频率
df = fs/(N-1) # 分辨率
f = [df*n for n in range(0,N)] # 构建频率数组

Y = np.fft.fft(y)*2/N #*2/N 反映了FFT变换的结果与实际信号幅值之间的关系
absY = [np.abs(x) for x in Y] #求傅里叶变换结果的模


# print the results that is different than zero;        
i=0
while i < len(absY):
    if absY[i] > 10:
       print('freq: %4d , Y: %5.2f + %5.2f j, A:%3.2f, phi: %6.1f'\
             %(i, Y[i].real, Y[i].imag, absY[i],math.atan2(Y[i].imag,Y[i].real)*180/math.pi))
    i +=1
    
pl.plot(f,absY)
pl.show()   

ss2=pd.read_csv('Data/SS00002.csv')
T=[datetime.strptime(x, '%Y-%m-%d')for x in ss2.Date]
F=[(x>=datetime(2008,10,22)) & (x <=datetime(2014,8,18)) for x in T] # comprehension method to make a filter
ss1 = ss2[F]

# filter method 2
T1=Series(T)
ss1=ss2[(T1>=datetime(2008,10,22)) & (T1 <=datetime(2014,8,18))] 
     
T2=ss2.Date.apply(lambda x: datetime.strptime(x,'%Y-%m-%d')) # Very important method of dataframe operation
ss1=ss2[(T2>=datetime(2008,10,22)) & (T2 <=datetime(2014,8,18))] 


t = [x/6578.0 for x in range(6578)]
y=list(ss2['Adj Close'])
y=list(np.log(ss2['Adj Close']))
N=len(y)
fs=6578.0
df=fs/(N-1)
f = [df*n for n in range(0,N)]
f1= [np.log(x) for x in f]

y=y[::-1]    
pl.plot(t,y)
pl.show()

   
     
Y = np.fft.fft(y)*2/N #*2/N 反映了FFT变换的结果与实际信号幅值之间的关系
absY = [np.abs(x) for x in Y] #求傅里叶变换结果的模    
     
pl.plot(f1[2:3289],absY[2:3289])
pl.show()   
   
pl.plot(f,absY)
pl.show() 



# Pyshon basic

a=[1,2,3]
b=a
a.append(4)
help(list.append)
list.append?

t=[0]*3

a=DataFrame(np.array(range(4)).reshape(2,2))
b=DataFrame(np.array(range(4,8)).reshape(2,2))
c=pd.concat([a,b])
c.mean()
c.groupby(c.index).mean()


############################---------------------------------------------


import sys

def show_sizeof(x, level=0):

    print ("\t" * level, x.__class__, sys.getsizeof(x), x)

    if hasattr(x, '__iter__'):
        if hasattr(x, 'items'):
            for xx in x.items():
                show_sizeof(xx, level + 1)
        else:
            for xx in x:
                show_sizeof(xx, level + 1)




x = T.dmatrix('x')
y = T.dmatrix('y')

z= x+y

f1= function([x+y],z)

x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x, y], z)


from lxml.html import parse
from urllib.request import urlopen

t_url = 'http://finance.yahoo.com/q/op?s=AAPL+Options'
parsed= parse(urlopen(t_url))

doc = parsed.getroot()

links = doc.findall('.//a')

urls = [lnk.get('href') for lnk in doc.findall('.//a')]

txt_content= [lnk.text_content() for lnk in doc.findall('.//a')]

url_dic = dict (zip(txt_content,urls))

tables = doc.findall('.//table')
tables = parsed.findall('.//table')

calls = tables[0]
rows = calls.findall('.//tr')

def _unpack(row, kind='td'):
    elts = row.findall('.//%s' % kind)
    return [val.text_content() for val in elts]

_unpack(rows[0], kind='th')


obj = """
{"name": "Wes",
"places_lived": ["United States", "Spain", "Germany"],
"pet": null,
"siblings": [{"name": "Scott", "age": 25, "pet": "Zuko"},
{"name": "Katie", "age": 33, "pet": "Cisco"}]
}
"""

import json

result = json.loads(obj)

asobj = json.dumps(result)

from pandas import DataFrame

siblings = DataFrame(result['siblings'], columns=['name', 'age'])

jstr= siblings.to_json()

siblings.to_json('../tmp/jstr.json')
jstr= siblings.to_json('../tmp/jstr.json')


# http hello world
def application(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/html')])
    return [b'<h1>Hello, web!</h1>']

def application(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/html')])
    body = '<h1>Hello, %s!</h1>' % (environ['PATH_INFO'][1:] or 'web')
    return [body.encode('utf-8')]



# server.py
# 从wsgiref模块导入:
from wsgiref.simple_server import make_server
# 导入我们自己编写的application函数:
#from hello import application

# 创建一个服务器，IP地址为空，端口是8000，处理函数是application:
httpd = make_server('', 8000, application)
print('Serving HTTP on port 8000...')
# 开始监听HTTP请求:
httpd.serve_forever()



#---------------------------


from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    return '<h1>Home</h1>'

@app.route('/signin', methods=['GET'])
def signin_form():
    return '''<form action="/signin" method="post">
              <p><input name="username"></p>
              <p><input name="password" type="password"></p>
              <p><button type="submit">Sign In</button></p>
              </form>'''

@app.route('/signin', methods=['POST'])
def signin():
    # 需要从request对象读取表单内容：
    if request.form['username']=='admin' and request.form['password']=='password':
        return '<h3>Hello, admin!</h3>'
    return '<h3>Bad username or password.</h3>'

if __name__ == '__main__':
    app.run()


#-----------------------

from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/signin', methods=['GET'])
def signin_form():
    return render_template('form.html')

@app.route('/signin', methods=['POST'])
def signin():
    username = request.form['username']
    password = request.form['password']
    if username=='admin' and password=='password':
        return render_template('signin_ok.html', username=username)
    return render_template('form.html', message='Bad username or password', username=username)

if __name__ == '__main__':
    app.run()


#------------------------------------
import asyncio

@asyncio.coroutine
def hello():
    print("Hello world!")
    # 异步调用asyncio.sleep(1):
    r = yield from asyncio.sleep(1)
    print("Hello again!")

# 获取EventLoop:
loop = asyncio.get_event_loop()
# 执行coroutine
loop.run_until_complete(hello())
loop.close()

#-------------------------------------

def consumer():
    r = ''
    while True:
        n = yield r
        if not n:
            return
        print('[CONSUMER] Consuming %s...' % n)
        r = '200 OK'

def produce(c):
    c.send(None)
    n = 0
    while n < 5:
        n = n + 1
        print('[PRODUCER] Producing %s...' % n)
        r = c.send(n)
        print('[PRODUCER] Consumer return: %s' % r)
    c.close()

c = consumer()
produce(c)

#----------------------------------

import asyncio

@asyncio.coroutine
def wget(host):
    print('wget %s...' % host)
    connect = asyncio.open_connection(host, 80)
    reader, writer = yield from connect
    header = 'GET / HTTP/1.0\r\nHost: %s\r\n\r\n' % host
    writer.write(header.encode('utf-8'))
    yield from writer.drain()
    while True:
        line = yield from reader.readline()
        if line == b'\r\n':
            break
        print('%s header > %s' % (host, line.decode('utf-8').rstrip()))
    # Ignore the body, close the socket
    writer.close()

loop = asyncio.get_event_loop()
tasks = [wget(host) for host in ['www.sina.com.cn', 'www.sohu.com', 'www.163.com']]
for _ in range(5):
    tasks.append(hello())
loop.run_until_complete(asyncio.wait(tasks))


asyncio.set_event_loop(loop)

#loop.close()



#---------------------------------
import threading
import asyncio

@asyncio.coroutine
def hello():
    print('Hello world! (%s)' % threading.currentThread())
    yield from asyncio.sleep(1)
    print('Hello again! (%s)' % threading.currentThread())

loop = asyncio.get_event_loop()
tasks = [hello(), hello()]
loop.run_until_complete(hello())

loop.close()

#-------------------------
import asyncio


async def hello():
    print("Hello world!")
    r = await asyncio.sleep(0.05)
    print("Hello again!")


async def wget(host):
    print('wget %s...' % host)
    connect = asyncio.open_connection(host, 80)
    reader, writer = await connect
    header = 'GET / HTTP/1.0\r\nHost: %s\r\n\r\n' % host
    writer.write(header.encode('utf-8'))
    await writer.drain()
    while True:
        line = await reader.readline()
        if line == b'\r\n':
            break
        print('%s header > %s' % (host, line.decode('utf-8').rstrip()))
    # Ignore the body, close the socket
    writer.close()

loop = asyncio.get_event_loop()
tasks = [wget(host) for host in ['www.sina.com.cn', 'www.sohu.com', 'www.163.com']]
for _ in range(5):
    tasks.append(hello())


loop.run_until_complete(asyncio.wait(tasks))
asyncio.set_event_loop(loop)

#----------------------------------

import asyncio

from aiohttp import web

async def index(request):
    await asyncio.sleep(0.5)
    return web.Response(body=b'<h1>Index</h1>')

async def hello(request):
    await asyncio.sleep(0.5)
    text = '<h1>hello, %s!</h1>' % request.match_info['name']
    return web.Response(body=text.encode('utf-8'))

async def init(loop):
    app = web.Application(loop=loop)
    app.router.add_route('GET', '/', index)
    app.router.add_route('GET', '/hello/{name}', hello)
    srv = await loop.create_server(app.make_handler(), '127.0.0.1', 8000)
    print('Server started at http://127.0.0.1:8000...')
    return srv

loop = asyncio.get_event_loop()
loop.run_until_complete(init(loop))
loop.run_forever()


##---------------------------------------------

import plotly.plotly as py
import pandas as pd
from plotly.graph_objs import *
import plotly.graph_objs as go
from plotly.offline import plot

ss2=pd.read_csv('Data/SS00002.csv')
ss20 = ss2.head(20)


data = Data([
    Bar(
        x=ss20["Date"],
        y=ss20["Close"]
    )
])
    
trace = go.Scatter (
        x=ss20["Date"],
        y=ss20["Close"]       
        )   
    
data =[trace]

import pandas_datareader.data as web
from datetime import datetime
df = web.DataReader("000001.SS", 'yahoo')
df1 = web.DataReader("000001.SS", 'yahoo', datetime(1995,1,1),datetime(2010,1,1))
df3 = pd.concat([df1,df])
df4 = df.sort_index()
df4.equals(df)
df.duplicated()
df.index.duplicated()
df.index.has_duplicates
df3.equals(df3.sort_index())  ### important check method

trace = go.Candlestick(x = ss20.Date,
                       open=ss20.Open,
                       high=ss20.High,
                       low=ss20.Low,
                       close=ss20.Close
                       )
data =[trace]



layout = Layout(
    title='INDEX 2016',
    font=Font(
        family='Raleway, sans-serif'
    ),
    showlegend=False,
    xaxis=XAxis(
        tickangle=-45
    ),
    bargap=0.05
)
fig = Figure(data=data)

plotly.offline.plot(fig,filename='Data/ss20.html')


import numpy as np
df1 = DataFrame(np.full((ss20.shape[0],2),np.nan))
df1 = DataFrame(np.full((ss20.shape[0],2),np.nan),columns =['Close1','Close2'])
ss20.append(df1)
ss22= pd.cocat(ss20,df1,axis=1)
ss22= pd.concat(ss20,df1,axis=1)
ss22= pd.concat([ss20,df1],axis=1)
ss21['Close2']= [None for x in range(len(ss20))]
ss21['Close2']= [np.nan for x in range(len(ss20))]
ss22['Close1'] = [x for x in ss22.Close if ss22.Date < datetime(2016,10,1)]


for i in range(len(ss22)):
    ss22.Close1[i] = ss22.Close[i] if (ss22.Date[i] < datetime (2016,10,1)) else np.nan
    ss22.Close2[i] = ss22.Close[i] if (ss22.Date[i] >= datetime (2016,10,1)) else np.nan
trace1 = go.Scatter (
        x=ss22["Date"],
        y=ss22["Close1"]       
        )   
trace2 = go.Scatter (
        x=ss22["Date"],
        y=ss22["Close2"]       
        )      

data =[trace1,trace2]

#########----------------------

import plotly.plotly as py
import plotly.graph_objs as go

import pandas_datareader.data as web
from datetime import datetime


df = web.DataReader("ss000001", 'yahoo', datetime(2007, 10, 1), datetime(2009, 4, 1))

trace = go.Candlestick(x=df.index,
                       open=df.Open,
                       high=df.High,
                       low=df.Low,
                       close=df.Close)
data = [trace]
plotly.offline.plot(data, filename='Data/simple_candlestick')


###### working on shanghai index
## check data first
import pandas_datareader.data as web
from datetime import datetime
df = web.DataReader("000001.SS", 'yahoo')
df1 = web.DataReader("000001.SS", 'yahoo', datetime(1995,1,1),datetime(2010,1,1))
df3 = pd.concat([df1,df])
df4 = df.sort_index()
df4.equals(df)
df.duplicated()
df.index.duplicated()
df.index.has_duplicates
df3.equals(df3.sort_index())  ### important check method
(df.Close != df.Low).any()
(df.Close == df.Low).all()

df3.to_csv('Data/SH_Index.csv')
df3=pd.read_csv('../Data/SH_Index.csv')


# Method 1 with integer row-index and date as a column
df3['Return']= [np.nan for x in range(len(df3))]# Never use None instead of np.nan!!!
for i in range(len(df3)):
    df3.loc[i,'Return'] = float(df3.loc[i,'Close']/df3.loc[i-1,'Close']-1) if i>0 else float(0)

# Method 2 with date as index
df3=pd.read_csv('Data/SH_Index.csv',index_col=0)
# important method not used here
df3.index=pd.DatetimeIndex(map(lambda x: datetime.strptime(x,'%Y-%m-%d'),df3.index),name='Date')

df3['Return']= [np.nan for x in range(len(df3))]# Never use None instead of np.nan!!!
for i in range(len(df3)):
    df3.iloc[i,6] = (df3.iloc[i,3]/df3.iloc[i-1,3]-1) if i>0 else 0.0

# Or
df3.Return = df3.Close/df3.Close.shift(1)-1
df3.Return = df3.Return.fillna(0.0)

# Try this to find timestamp indexing
df5 = df[:100]
df5.loc[:,'Return']= df5.Close/df5.Close.shift(1)-1



from pandas.tseries.offsets import Hour, Minute,Day
from datetime import timedelta
from dateutil.parser import parse

plot.hist(df3.Return,bins=50,range=(-0.1,0.1))


df3.Close.describe()
df3.Return.describe()
df3.describe()
df3.Return.argmax()
df3.Return.argmin()
df3[(np.abs(df3.Return) >0.1)]
np.percentile(df3.Return,[2.5,97.5])

df_clean= df3[(np.abs(df3.Return) <=0.1)]
r_clean = df_clean.Return
r_cat= pd.cut(r_clean,bins=50)
r_cat.groupby(r_cat).count()


####### RNN practice----------------------------
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys


text =r_cat

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
    model.fit(X, y,
              batch_size=128,
              epochs=1)

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.4,0.6,0.8,1.0,1.2,1.4]:
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
        
        for i in range(10):
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

start_index = len(text) - maxlen - 15



#####---------------------

import pandas as pd
df3=pd.read_csv('Data/SH_Index.csv',index_col=0)
df=df3[:5]
h = df.to_html()
open('../tmp/my_table.html', 'w').write(h)

#####------------------------
import numpy as np
import pandas as pd
from IPython.display import HTML, Javascript
df = pd.DataFrame({'a': np.arange(10), 'b': np.random.randn(10)})
HTML(df.to_html(classes='my_class'))
Javascript('''$('.my_class tbody tr').filter(':last')
                                             .css('background-color', '#FF0000');
                   ''')


###------------------------------

table_css = open('../tmp/table_css.css', 'r').read()
h=table_css + df.to_html(classes='table',index=False)
open('../tmp/my_table.html', 'w').write(h)





table_css='''
<style type="text/css"> 
.table 
{ 
width: 100%; 
padding: 0; 
margin: 0; 
} 
th { 
font: bold 12px "Trebuchet MS", Verdana, Arial, Helvetica, sans-serif; 
color: #4f6b72; 
border-right: 1px solid #C1DAD7; 
border-bottom: 1px solid #C1DAD7; 
border-top: 1px solid #C1DAD7; 
letter-spacing: 2px; 
text-transform: uppercase; 
text-align: left; 
padding: 6px 6px 6px 12px; 
background: #CAE8EA no-repeat; 
} 
td { 
border-right: 1px solid #C1DAD7; 
border-bottom: 1px solid #C1DAD7; 
background: #fff; 
font-size:14px; 
padding: 6px 6px 6px 12px; 
color: #4f6b72; 
} 
td.alt { 
background: #F5FAFA; 
color: #797268; 
} 
th.spec,td.spec { 
border-left: 1px solid #C1DAD7; 
} 
/*---------for IE 5.x bug*/ 
html>body td{ font-size:14px;} 
tr.select th,tr.select td 
{ 
background-color:#CAE8EA; 
color: #797268; 
} 
</style>
'''







