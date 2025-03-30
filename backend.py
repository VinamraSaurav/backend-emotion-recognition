import base64
import cv2
import numpy as np
import socketio
import eventlet
import librosa
import json
from datetime import datetime
from flask import Flask
from tensorflow.keras.models import load_model
import requests
import os
import time
from transformers import pipeline, Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torch
import tensorflow as tf
from dotenv import load_dotenv

# load api_key from dotenv
# Load environment variables from a .env file
load_dotenv()

# Retrieve the Hugging Face API key from the environment
HUGGINGFACE_API_KEY = os.getenv("API_KEY")

if not HUGGINGFACE_API_KEY:
    log("❌ Hugging Face API key not found in environment variables!")

# Configure environment to disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# Initialize Flask app and SocketIO server
app = Flask(__name__)
sio = socketio.Server(cors_allowed_origins="*")
app.wsgi_app = socketio.WSGIApp(sio, app)

# Emotion labels for facial emotion recognition
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def log(msg):
    """Print messages with timestamps for debugging."""
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp} {msg}")

# Load Face Emotion Recognition Model
try:
    log("🔄 Loading facial emotion recognition model...")
    face_model = load_model("model.h5")
    log("✅ Face model loaded successfully!")
except Exception as e:
    log(f"❌ Error loading face model: {e}")
    face_model = None

# Load Haar Cascade for face detection
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        log("❌ Error loading Haar Cascade XML file!")
    else:
        log("✅ Haar Cascade loaded successfully!")
except Exception as e:
    log(f"❌ Error loading Haar Cascade: {e}")
    face_cascade = None

# Initialize speech emotion model - We'll load this on demand if the API fails
speech_model = None
speech_processor = None

def get_local_speech_model():
    """Load the local speech emotion model if not already loaded"""
    global speech_model, speech_processor
    
    if speech_model is None:
        try:
            log("🔄 Loading local speech emotion recognition model (fallback)...")
            model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
            speech_processor = Wav2Vec2Processor.from_pretrained(model_name)
            speech_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
            log("✅ Local speech model loaded successfully!")
        except Exception as e:
            log(f"❌ Error loading local speech model: {e}")
            return None, None
            
    return speech_model, speech_processor

# HuggingFace API configuration for speech emotion recognition
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# Socket Events
@sio.on("connect")
def connect(sid, environ):
    log(f"✅ Client connected: {sid}")
    sio.emit("connection_status", {"status": "connected"}, room=sid)

@sio.on("disconnect")
def disconnect(sid):
    log(f"❌ Client disconnected: {sid}")

def process_audio_locally(audio_path):
    """Process audio using the local model instead of the API"""
    model, processor = get_local_speech_model()
    if model is None or processor is None:
        return None
        
    try:
        # Load audio
        speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
        
        # Process audio with local model
        inputs = processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_class_id = torch.argmax(logits, dim=1).item()
            
        # Map class ID to emotion label
        id2label = model.config.id2label
        predicted_label = id2label[predicted_class_id]
        
        # Calculate confidence
        probs = torch.nn.functional.softmax(logits, dim=1)
        confidence = probs[0][predicted_class_id].item()
        
        return {
            "label": predicted_label,
            "score": confidence
        }
    except Exception as e:
        log(f"❌ Error processing audio locally: {e}")
        return None

@sio.on("send_audio")
def receive_audio(sid, data):
    try:
        log(f"🎤 Received audio from {sid}")

        if "audio" not in data:
            raise ValueError("No audio data received")

        # Process base64 audio data
        audio_data = data["audio"]
        if "," in audio_data:
            audio_data = audio_data.split(",")[1]  # Remove metadata if present
            
        audio_bytes = base64.b64decode(audio_data)
        
        # Save the audio file
        audio_path = f"temp_audio_{sid}.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        log(f"✅ Audio file saved to {audio_path}")

        # Try API approach first
        api_success = False
        max_retries = 2
        retries = 0
        
        while not api_success and retries < max_retries:
            try:
                log("🔄 Sending audio to Hugging Face API...")
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                
                response = requests.post(
                    HUGGINGFACE_API_URL,
                    headers=HEADERS,
                    data=audio_bytes,
                    timeout=5  # Add timeout to prevent long waits
                )

                if response.status_code == 200:
                    # Parse response from Hugging Face API
                    result = response.json()
                    log(f"📊 HuggingFace API response: {result}")
                    
                    # Extract predicted emotion
                    if isinstance(result, list) and len(result) > 0:
                        if "label" in result[0]:
                            emotion = result[0]["label"]
                            confidence = result[0].get("score", 0)
                            log(f"🔮 Predicted Speech Emotion (API): {emotion} (confidence: {confidence:.2f})")
                            
                            # Send result to client
                            sio.emit("emotion_result", {
                                "type": "speech", 
                                "result" : result,
                                "method": "api"
                            }, room=sid)
                            log("📡 Speech Emotion result sent successfully!")
                            api_success = True
                        else:
                            log(f"❌ Unexpected API response format: {result}")
                    else:
                        log(f"❌ Empty or invalid API response: {result}")
                else:
                    log(f"❌ API Error: {response.status_code} - Response too long to display")
                    retries += 1
                    time.sleep(1)  # Wait before retry
                    
            except Exception as e:
                log(f"❌ API request error: {e}")
                retries += 1
                time.sleep(1)  # Wait before retry

        # If API fails, use local model
        if not api_success:
            log("🔄 Falling back to local speech emotion recognition...")
            result = process_audio_locally(audio_path)
            
            if result:
                emotion = result["label"]
                confidence = result["score"]
                log(f"🔮 Predicted Speech Emotion (Local): {emotion} (confidence: {confidence:.2f})")
                
                # Send result to client
                sio.emit("emotion_result", {
                    "type": "speech", 
                    "result": result,
                    "method": "local"
                }, room=sid)
                log("📡 Speech Emotion result (local model) sent successfully!")
            else:
                log("❌ Failed to process audio with local model")
                sio.emit("error", {"message": "Failed to process audio"}, room=sid)
            
        # Clean up
        try:
            os.remove(audio_path)
        except Exception as e:
            log(f"⚠️ Could not remove temporary audio file: {e}")

    except Exception as e:
        log(f"❌ Error processing audio: {e}")
        sio.emit("error", {"message": f"Audio processing error: {str(e)}"}, room=sid)

@sio.on("send_frame")
def receive_frame(sid, data):
    try:
        if face_model is None or face_cascade is None:
            sio.emit("error", {"message": "Face recognition models not loaded properly"}, room=sid)
            return
            
        log(f"📷 Received video frame from {sid}")
        
        if "frame" not in data:
            raise ValueError("No frame data received")
            
        # Decode base64 image
        frame_data = data["frame"]
        if "," in frame_data:
            frame_data = frame_data.split(",")[1]  # Remove metadata if present
            
        frame_bytes = base64.b64decode(frame_data)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Could not decode image")
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        result = {"type": "face", "faces": []}
        
        # Process each face
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            
            # Preprocess for model input
            resized_face = cv2.resize(face_roi, (48, 48))
            normalized_face = resized_face / 255.0
            reshaped_face = normalized_face.reshape(1, 48, 48, 1)
            
            # Predict emotion
            emotion_prediction = face_model.predict(reshaped_face, verbose=1)[0]
            emotion_index = np.argmax(emotion_prediction)
            emotion_label = EMOTION_LABELS[emotion_index]
            confidence = float(emotion_prediction[emotion_index])
            
            # Add to results
            result["faces"].append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "emotion": emotion_label,
                "confidence": confidence
            })
            
            log(f"👤 Face detected: Emotion={emotion_label}, Confidence={confidence:.2f}")
        
        # Send results to client
        sio.emit("emotion_result", result, room=sid)
        log(f"📡 Facial Emotion results sent for {len(faces)} faces")
        
    except Exception as e:
        log(f"❌ Error processing video frame: {e}")
        sio.emit("error", {"message": f"Frame processing error: {str(e)}"}, room=sid)

# Health check endpoint
@app.route('/health')
def health_check():
    return 'Emotion Recognition Backend is running', 200

# Start Flask-SocketIO Server
if __name__ == "__main__":
    log("🚀 Starting Flask-SocketIO Emotion Recognition Server on port 5000...")
    eventlet.wsgi.server(eventlet.listen(("0.0.0.0", 5000)), app)