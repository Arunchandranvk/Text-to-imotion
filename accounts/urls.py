from django.urls import path
from .views import *

urlpatterns = [
    path('home/',MainPage.as_view(),name='main'),
    path('registration/',RegView.as_view(),name='reg'),
    path('chatbot/',ChatbotView.as_view(),name='bot'),
    path('emotion/',emotion_view,name='emotion'),
    path('video/',video_feed,name='video_feed'),
   
]