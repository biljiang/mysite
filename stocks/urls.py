from django.conf.urls import url

from . import views

app_name = 'stocks'
urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^display_meta/$', views.display_meta, name='display_meta'),
    url(r'^response/$', views.response, name='response'),
    url(r'^analysis/$', views.analysis, name='analysis'),
    url(r'^b_s_statistics/$', views.b_s_statistics, name='b_s_statistics'),
    url(r'^data_details/$', views.data_details, name='data_details'),
    url(r'^JJJ_input/$', views.JJJ_input, name='JJJ_input'),
    url(r'^JJJ_list/$', views.JJJ_list, name='JJJ_list'),
]
