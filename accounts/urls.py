from django.urls import path
from .views import *

urlpatterns = [
    path('home/',MainPage.as_view(),name='main'),
    path('registration/',RegView.as_view(),name='reg'),
    path('chatbot/',ChatbotView.as_view(),name='bot'),
    path('emotion/',emotion_view,name='emotion'),
    path('video/',video_feed,name='video_feed'),
    path('video_feed_object/', video_feed_object, name='video_feed_object'),
    path('object/', ObjectView.as_view(), name='object')
]