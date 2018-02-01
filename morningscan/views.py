from django.shortcuts import render
from datetime import datetime
# Create your views here.

def index(request):
    error = False
    filename = "/static/m925_scan.txt"
    if 'dateofdata' in request.POST:
        dateofdata = request.POST['dateofdata']
        find = request.POST['find']
        if not dateofdata:
            dateofdata = str(datetime.today().date())
        dateofdata = dateofdata.replace('-','_')
        if not find:
            return render(request, 'morningscan/index.html',{'error':True,'filename':"no find input"})
        try:
            if find =='a':
                filename = '/static/scan_925/ob_'+dateofdata+'.txt'
            elif find =='b':
                filename = '/static/scan_930/cb_'+dateofdata+'.txt'
            else :
                filename = '/static/scan_930/qd_'+dateofdata+'.txt'        
            f=open('/home/bill/datadisk/shared/mysite'+filename, 'rb')
            f.close()
        except:
            #return render(request, 'morningscan/index.html',{'error':True})
            return render(request, 'morningscan/index.html',{'error': True,'filename':dateofdata})
        return render(request, 'morningscan/index.html',{'error': False,'filename':filename}) 
    return render(request, 'morningscan/index.html',{'error': error,'filename': filename})
