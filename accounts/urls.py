from django.conf.urls import url
#from django.urls import path
from . import views
#import django.contrib.auth.views as auth_views


app_name = 'accounts'
urlpatterns = [
    url(r'^$', views.index, name='index'),
#    url(r'^login/$', auth_views.login(), name='login'),
    url(r'^login/$', views.login, name='login'),
    url(r'^password_reset/$', views.password_reset, name='password_reset'),
    url(r'^profile/$', views.profile, name='profile'),
]



'''
app_name = 'polls'

urlpatterns = [
    # ex: /polls/
    url(r'^$', views.index, name='index'),
    # ex: /polls/5/
    url(r'^(?P<question_id>[0-9]+)/$', views.detail, name='detail'),
    # ex: /polls/5/results/
    url(r'^(?P<question_id>[0-9]+)/results/$', views.results, name='results'),
    # ex: /polls/5/vote/
    url(r'^(?P<question_id>[0-9]+)/vote/$', views.vote, name='vote'),
]
'''
