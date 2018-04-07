from django.shortcuts import render
#import matplotlib
#matplotlib.use('Agg')
# Create your views here.
from django.http import HttpResponse
#from stocks import lib_stockselect as l_s
#from stocks import lib_stockplot as l_plt
#from stocks import draw_st_graph as l_draw
#import matplotlib.pyplot as plt
#from io import BytesIO


def index(request):
    return render(request, 'index.html')


def response(request):
    d='2017-06-22';s='603767'
    data = l_draw.get_data_fenbi(s,d)
    fig = l_draw.stockDF_graph(data,s,d)
    image_file = BytesIO()
    fig.savefig(image_file,format='png')
    image_data = image_file.getvalue()

    return HttpResponse(image_data, content_type="image/png")

def display_meta(request):
    values = request.META.items()
    html = []
    for k, v in values:
        html.append('<tr><td>%s</td><td>%s</td></tr>' % (k, v))
    return HttpResponse('<table>%s</table>' % '\n'.join(html))
