"""Monitor URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='Home'),
    path('login/', views.login, name='Login'),
    path('signup/', views.signup, name='Signup'),
    path('registerFace/', views.registerFace, name='RegisterFace'),
    path('account/', views.account, name='Account'),
    path('password/', views.password, name='Password'),
    path('manage/', views.manage, name='Manage'),
    path('adduser/', views.addUser, name='AddUser'),
    path('manage/<int:id>/', views.accountInfo),
    path('video/', views.video, name='Video'),
    path('alert/', views.alert, name='Alert'),
    path('alertPage/', views.alertPage, name='AlertPage')
]
