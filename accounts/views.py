from django.contrib.auth.views import LoginView
#from django.contrib.auth import login

from django.http import HttpResponse,HttpResponseRedirect
from django.shortcuts import render
from django.contrib.auth import forms

# Create your views here.
def index(request):
    if request.user.is_authenticated:
        return HttpResponseRedirect('profile/')
    else:
        return HttpResponseRedirect('login/')
#    return HttpResponse('welcome to account management page. Under construction')


def login(request):
    
    context =dict(username = request.user.get_username())
    defaults ={
               'extra_context':context
              }
    return LoginView.as_view(**defaults)(request)


def password_reset(request):
    return HttpResponse('welcome to password_reset page. Under construction')

def profile(request):
    return HttpResponse('welcome to account profil page. Under construction')
