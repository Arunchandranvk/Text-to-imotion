from django.shortcuts import render,redirect
from django.views.generic import TemplateView,FormView,CreateView,View
from django.urls import reverse_lazy
from django.contrib.auth import authenticate,login,logout
from .models import *
from .forms import *
from django.http import HttpResponseBadRequest

from django.http import JsonResponse, HttpResponseNotAllowed
from groq import Groq
from django.views.decorators.csrf import csrf_exempt
from .emotion import *
import json


class LoginView(FormView):
    template_name="login.html"
    form_class=LogForm
    def post(self,request,*args,**kwargs):
        log_form=LogForm(data=request.POST)
        if log_form.is_valid():  
            us=log_form.cleaned_data.get('username')
            ps=log_form.cleaned_data.get('password')
            user=authenticate(request,username=us,password=ps)
            if user: 
                login(request,user)
                # if user.is_superuser == 1 :
                #    return redirect('dh')
                # else:
                return redirect('main')
            else:
                return render(request,'login.html',{"form":log_form})
        else:
            return render(request,'login.html',{"form":log_form}) 
        

class RegView(CreateView):
     form_class=Reg
     template_name="register.html"
     model=CustUser
     success_url=reverse_lazy("log")  


class MainPage(TemplateView):
    template_name = 'main.html'
    

from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

@method_decorator(csrf_exempt, name='dispatch')
class CustomLogoutView(View):
    def get(self, request):
        logout(request)  # Log out the user
        response = redirect('log')  # Redirect to home or login page

        return response


    


from django.contrib.auth import logout as auth_logout


def custom_logout(request):
    auth_logout(request)
    return redirect('log')



class ChatbotView(View):
    def get(self, request):
        return render(request, "chatbot.html")
    def post(self, request): 
        try:
            body = json.loads(request.body)
            user_input = body.get('userInput')
        except json.JSONDecodeError as e:
            return JsonResponse({"error": "Invalid JSON format."})
    
        if not user_input:  # If user_input is None or empty
            print("no")
            return JsonResponse({"error": "No user input provided."})  
        
        print("User Input:", user_input)
        
        static_responses = {
            # "hi": "Hello! How can I assist you today?",
            # "hello": "Hi there! How can I help you?",
            # "how are you": "I'm just a chatbot, but I'm doing great! How about you?",
            # "bye": "Goodbye! Take care.",
            # "whats up": "Not much, just here to help you with  queries. How can I help you today?",
        }

        lower_input = user_input.lower().strip()
        if lower_input in static_responses:
            print(static_responses[lower_input])
            return JsonResponse({'response': static_responses[lower_input]})
        
        try:
            print("Processing via GROQ")
            data = get_groq_response(user_input)
            print(data)
            treatment_list = data.split('\n')
            return JsonResponse({'response': treatment_list})
        except Exception as e:
            return JsonResponse({"error": f"Failed to get GROQ response: {str(e)}"})

client = Groq(api_key="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")

import re
def get_groq_response(user_input):
    """
    Communicate with the GROQ chatbot to get a response based on user input.
    
    """
    system_prompt = {
      "role": "system",
      "content":
      "You are a helpful assistant. You reply with very short answers considering emotion and sentiment."
    }

    chat_history = [system_prompt]
    while True:
        print("groq",user_input)

        emotion = predict_emotion(user_input)
        sentiment = predict_sentiment(user_input)
        print("emotion",emotion)
        print("sentiment",sentiment)
        # Append the user input to the chat history
        chat_history.append({"role": "user", "content": user_input+','+ emotion +','+ sentiment})

        chat_completion = client.chat.completions.create(model="llama3-70b-8192",
                                                messages=chat_history,
                                                max_tokens=100,
                                                temperature=1.2)

        chat_history.append({
        "role": "assistant",
        "content": chat_completion.choices[0].message.content
        })
        print("response",chat_completion.choices[0].message.content)
        response = chat_completion.choices[0].message.content
        response = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', response)   
        # response = response.replace('+', ' and ').replace('.', ' there')
        return response

    
import cv2
import numpy as np
from django.shortcuts import render
from django.http import StreamingHttpResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load pre-trained model
best_model = load_model("face_model.h5")

# Emotion class labels
class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_image = cv2.resize(face_roi, (48, 48))
        face_image = image.img_to_array(face_image)
        face_image = np.expand_dims(face_image, axis=0)
        
        predictions = best_model.predict(face_image)
        emotion_label = class_names[np.argmax(predictions)]
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame

def generate_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

# views.py
def emotion_view(request):
    emotions = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    return render(request, 'emotion.html', {'emotions': emotions})

def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
