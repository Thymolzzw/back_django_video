"""demo_django URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
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
from django.contrib import admin
from django.urls import path, re_path
from . import views, back_views
from django.contrib.staticfiles import views as my_view

urlpatterns = [

    path('uploadvideo', back_views.uploadvideo),
    re_path(r'^static/(?P<path>.*)$', my_view.serve),
    path('getAllVideos', views.getAllVideos),
    path('getOneVideos', views.getOneVideos),
    path('getAllPeopleFace', views.getAllPeopleFace),
    path('getEquipment', views.getVideoEquipment),
    path('getBinner', views.getBinner),
    path('getText', views.getText),
    path('doLogin', views.doLogin),
    path('doSignUp', views.doSignUp),
    path('myTest', back_views.myTest),
    path('streamVideo', views.streamVideo),
    path('getSubTitle', views.getSubTitle),
    path('getSubTitleForUpdate', views.getSubTitleForUpdate),
    path('getVideoAdditionData', views.getVideoAdditionData),
    path('getVideoEquipment', views.getVideoEquipment),
    path('getVideoPPT', views.getVideoPPT),
    path('getFace', views.getFace),
    path('deletevideo', back_views.deletevideo),
    path('getUserInfo', views.getUserInfo),
    path('getAllResource', views.getAllResource),
    path('getProduct', views.getProduct),
    path('updateSubTitle', back_views.updateSubTitle),
    path('updateSubTitleItem', back_views.updateSubTitleItem),
    path('updateOCR', back_views.updateOCR),
    path('delOCRItem', back_views.delOCRItem),
    path('updateOCRItem', back_views.updateOCRItem),
    path('updatePPT', back_views.updatePPT),
    path('delPPTImg', back_views.delPPTImg),
    path('searchVideos', back_views.searchVideos),
    path('updateUserInfo', back_views.updateUserInfo),
    path('getVideoVoicePrint', views.getVideoVoicePrint),
    path('updateEquipment', back_views.updateEquipment),
    path('updateFaceItem', views.updateFaceItem),
    path('deleteFaceItem', views.deleteFaceItem),
    path('getCountryList', views.getCountryList),
    path('getResourceList', views.getResourceList),
    path('deleteResource', views.deleteResource),
    path('deleteCountry', views.deleteCountry),
    path('addEditResource', views.addEditResource),
    path('addEditCountry', views.addEditCountry),
    path('changeVideoInfo', back_views.changeVideoInfo),
    path('getPeopleList', views.getPeopleList),
    path('changeVoicePrint', views.changeVoicePrint),
    path('deleteEquipment', back_views.deleteEquipment),

]
