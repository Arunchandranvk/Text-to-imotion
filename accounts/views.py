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
import re
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from django.http import StreamingHttpResponse, HttpResponse
from ultralytics import YOLO
import threading
import time


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

    



best_model = load_model("face_model.h5")


class_names = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    # Track if we have a detected face and its emotion
    detected_emotion = None
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_image = cv2.resize(face_roi, (48, 48))
        face_image = image.img_to_array(face_image)
        face_image = np.expand_dims(face_image, axis=0)
        
        predictions = best_model.predict(face_image)
        emotion_label = class_names[np.argmax(predictions)]
        detected_emotion = emotion_label
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    if detected_emotion:
        prompt = f"I'm feeling {detected_emotion.lower()}. Give me a short motivational message."
        
        motivational_message = get_groq_response2(prompt)
        y_position = frame.shape[0] - 80  # Position near bottom of frame
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_position - 40), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Split text if needed and display
        words = motivational_message.split()
        line = ""
        for i, word in enumerate(words):
            if len(line + word) < 40:  # Limit chars per line
                line += word + " "
            else:
                cv2.putText(frame, line, (20, y_position), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_position += 30
                line = word + " "

        if line:
            cv2.putText(frame, line, (20, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
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


# views.py


# Global variables
model = YOLO("yolov8n.pt")
camera = None
lock = threading.Lock()
detection_active = True  # Flag to control detection status

# Cache for object descriptions to avoid repeated API calls
object_descriptions = {}

def get_object_description(object_name):
    """
    Get a brief description for the detected object using Groq
    """
    if object_name in object_descriptions:
        return object_descriptions[object_name]
    
    system_prompt = {
        "role": "system",
        "content": "You are a helpful assistant. Provide a brief, informative description (under 30 characters) for the object mentioned."
    }
    
    chat_history = [system_prompt]
    chat_history.append({"role": "user", "content": f"Describe a {object_name} very briefly."})
    
    try:
        chat_completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=chat_history,
            max_tokens=30,  # Limit tokens for shorter responses
            temperature=0.3  # Lower temperature for more factual responses
        )
        
        description = chat_completion.choices[0].message.content
        # Clean up the description and keep it short
        description = description.strip('"\'.,!? ').strip()
        if len(description) > 30:
            description = description[:27] + "..."
        
        # Store in cache for future use
        object_descriptions[object_name] = description
        return description
        
    except Exception as e:
        print(f"Error getting description: {e}")
        return "Common object"

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            raise ValueError("Unable to open video source")

        # Start background frame grabbing thread
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()

        # Store the latest frame
        self.current_frame = None

    def _capture_loop(self):
        while self.is_running:
            success, frame = self.video.read()
            if success:
                with lock:
                    self.current_frame = frame
            time.sleep(0.01)  # Small delay to prevent hogging CPU

    def get_frame(self):
        with lock:
            if self.current_frame is None:
                # Return an error frame if no frame is available
                blank = 255 * np.ones((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                _, jpeg = cv2.imencode('.jpg', blank)
                return jpeg.tobytes()

            # Make a copy of the current frame for annotation
            annotated_frame = self.current_frame.copy()
            
            # Only run object detection if detection is active
            global detection_active
            if detection_active:
                # Process the frame with YOLO
                results = model.predict(source=self.current_frame, conf=0.5)
                
                # Draw bounding boxes with descriptions
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        # Extract coordinates and class
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        cls_name = model.names[cls_id]
                        
                        # Get description for the object
                        description = get_object_description(cls_name)
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Prepare label text with class name, confidence, and description
                        label = f"{cls_name} ({conf:.2f}): {description}"
                        
                        # Calculate text position
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        
                        # Add background for text
                        cv2.rectangle(annotated_frame, 
                                     (x1, y1 - text_size[1] - 5), 
                                     (x1 + text_size[0], y1), 
                                     (0, 255, 0), 
                                     -1)
                        
                        # Add text
                        cv2.putText(annotated_frame, 
                                   label, 
                                   (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.5, 
                                   (0, 0, 0), 
                                   1)
            else:
                # If detection is not active, add a message to the frame
                cv2.putText(annotated_frame, 
                           "Detection Paused", 
                           (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           1, 
                           (0, 0, 255), 
                           2)

            # Convert to JPEG
            _, jpeg = cv2.imencode('.jpg', annotated_frame)
            return jpeg.tobytes()

    def __del__(self):
        self.is_running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if hasattr(self, 'video'):
            self.video.release()

# Camera instance
video_camera = None

def get_camera():
    global video_camera
    if video_camera is None:
        try:
            video_camera = VideoCamera()
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return None
    return video_camera

def generate_frame():
    camera = get_camera()
    if camera is None:
        # Generate error frames if camera fails
        while True:
            blank = 255 * np.ones((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            _, jpeg = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(1)

    # Generate frames from the camera
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)  

def video_feed_object(request):
    return StreamingHttpResponse(
        generate_frame(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


class ObjectView(TemplateView):
    template_name = 'object.html'

def toggle_detection(request):
    """
    Toggle object detection on/off based on the request parameter
    """
    global detection_active
    action = request.GET.get('action', 'stop')
    
    if action == 'stop':
        detection_active = False
        return JsonResponse({'status': 'success', 'detection': 'stopped'})
    elif action == 'resume':
        detection_active = True
        return JsonResponse({'status': 'success', 'detection': 'resumed'})
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid action'}, status=400)
    
    
def get_groq_response2(user_input):
    """
    Get a motivational response from Groq based on detected emotion
    """
    # Cache responses to avoid repeated API calls for the same emotion
    if not hasattr(get_groq_response2, 'cache'):
        get_groq_response2.cache = {}
    
    # Check if we already have a response for this emotion in cache
    emotion_key = user_input.lower()
    if emotion_key in get_groq_response2.cache:
        return get_groq_response2.cache[emotion_key]
    
    system_prompt = {
      "role": "system",
      "content": "You are a motivational assistant. Provide brief,  responses (under 10 characters) for people based on their detected emotion. Be direct and encouraging. and also ask question for angry"
    }

    chat_history = [system_prompt]
    
    emotion = predict_emotion(user_input)
    sentiment = predict_sentiment(user_input)
    chat_history.append({"role": "user", "content": user_input + ',' + emotion + ',' + sentiment})
    chat_completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=chat_history,
        max_tokens=50,  # Limit tokens for shorter responses
        temperature=0.7  # Lower temperature for more consistent responses
    )

    response = chat_completion.choices[0].message.content
    response = re.sub(r'\*\*(.*?)\*\*', r'', response)
    
    # Store in cache for future use
    get_groq_response2.cache[emotion_key] = response
    
    return response