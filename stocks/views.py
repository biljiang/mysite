from django.shortcuts import render
from django.contrib.auth.decorators import login_required
import matplotlib
matplotlib.use('Agg')# use this to turn off the terminal display which will cause errer when running on server
# Create your views here.
from django.http import HttpResponse
#from stocks import lib_stockselect as l_s
#from stocks import lib_stockplot as l_plt
from .lib_stockselect import name_dict,code_dict,Last_BD
from . import draw_st_graph as l_draw
#import matplotlib.pyplot as plt
from io import BytesIO,StringIO
from datetime import datetime,time
import pandas as pd
import tushare as ts
import re
import json


code = None;df = pd.DataFrame() 
date = str(Last_BD())

@login_required
def index(request):
    return render(request, 'stocks/index.html')


def response(request):
    global date,code,df
    s =code;d=date
    data = df
    fig = l_draw.stockDF_graph(df,s,d)
    image_file = BytesIO()
    fig.savefig(image_file,format='png')
    image_data = image_file.getvalue()
    return HttpResponse(image_data, content_type="image/png")

def analysis(request):
    global date,code,df
    date=request.POST['d'];code=request.POST['s']
    if not date:
        date = str(Last_BD())
    if not code:
        return render(request, 'stocks/index.html',{'error':True})
    try:
        df = l_draw.get_data_fenbi(code,date)
    except:
        return render(request, 'stocks/index.html',{'error':True})
    tu = l_draw.data_split(df=df)
    c = {'present_time':str(datetime.now()),'B_amount':tu[1].amount.sum(),'S_amount':tu[2].amount.sum(),'code':code,'date':date,'s_name':name_dict[code]}
    return render(request, 'stocks/stock_dailyANA.html',c)

def b_s_statistics(request):
    global date,code,df
    tu = l_draw.data_split(df)
    txt_file = StringIO()
    print('-'*60,file = txt_file)
    print('Stock '+code+' on date '+date+': Buying= {0:,}  Selling= {1:,}'.format(
            tu[1].amount.sum(),tu[2].amount.sum()),file=txt_file)
    print('Open amount :{:,}   Close amount :{:,}'.format(tu[3].amount.sum(),tu[4].amount.sum()),file=txt_file)
    print('Stock name: {}   Code: {}'.format(name_dict[code],code),file = txt_file)
    print(df[:6],file=txt_file)
    print(df[-6:],file = txt_file)
    return render(request, 'stocks/b_s_statistics.html',{'b_s_stats':txt_file.getvalue()})

def data_details(request):
    global date,code,df
    h=df.to_html(classes='table',index=False)
    return render(request, 'stocks/data_details.html',{'data_frame':h})

def JJJ_input(request):
    return render(request,'stocks/JJJ_input.html')

def JJJ_list(request):
    t_flag = False
    d = str(Last_BD())
    file_name='/home/bill/datadisk/shared/mysite/static/jjj_detail.txt'
    file_name1='/home/bill/datadisk/shared/mysite/static/jjj'+d.replace('-','_')+'.json'
    code_list= request.POST['JJJ_InputList']
    regex=re.compile('\d{6}')
    s_list = regex.findall(code_list)
    s_list = [s for s in s_list if s in name_dict]
    s_dict = [(s,name_dict[s]) for s in s_list]
    with open('/home/bill/datadisk/shared/mysite/static/jjj_list.json','w',encoding = 'utf-8') as file:
        json.dump(s_list,file,ensure_ascii=False)
    with open(file_name1,'w',encoding = 'utf-8') as file:
        json.dump(s_list,file,ensure_ascii=False)
    if datetime.now().time()> time(9,31,0) and datetime.now().time() < time(10,0,0):
        t_flag = True
        file_jjj = open(file_name,'w',encoding = 'utf-8')
        #df_tmp = DataFrame()    
        for s in s_list:
            try:
                data = ts.get_today_ticks(s)
            except:
                try:
                    data = ts.get_tick_data(s,d)
                except:
                    print('Getting data failed!',file = file_jjj)

            print('-'*60,file=file_jjj)
            print('Stock name: {}   Code: {}'.format(name_dict[s],s),file=file_jjj) 
            data = data[-6:]
            data['code'] = s
            print(data,file = file_jjj)
        file_jjj.close()        
    return render(request,'stocks/JJJ_list.html',{'s_dict':s_dict,'s_list':s_list,'t_flag':t_flag})

def display_meta(request):
    values = request.META.items()
    html = []
    for k, v in values:
        html.append('<tr><td>%s</td><td>%s</td></tr>' % (k, v))
    return HttpResponse('<table>%s</table>' % '\n'.join(html))
