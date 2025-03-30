import base64
import cv2
import numpy as np
import socketio
import eventlet
import librosa
import json
import wave
import struct
from datetime import datetime
from flask import Flask
from tensorflow.keras.models import load_model
import requests
import os
import time
from dotenv import load_dotenv
from model import get_model

# Load environment variables
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("API_KEY")

if not HUGGINGFACE_API_KEY:
    print("❌ Hugging Face API key not found in environment variables!")

# Configure environment
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Initialize Flask app and SocketIO
app = Flask(__name__)
sio = socketio.Server(cors_allowed_origins="*")
app.wsgi_app = socketio.WSGIApp(sio, app)

# Emotion labels
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def log(msg):
    """Enhanced logging with timestamp"""
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

# Load Haar Cascade
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        log("❌ Error loading Haar Cascade XML file!")
    else:
        log("✅ Haar Cascade loaded successfully!")
except Exception as e:
    log(f"❌ Error loading Haar Cascade: {e}")
    face_cascade = None

# Initialize speech model
speech_model = None

def get_speech_model():
    """Get the speech emotion model with proper error handling"""
    global speech_model
    if speech_model is None:
        try:
            log("🔄 Initializing speech emotion model...")
            speech_model = get_model()
            if speech_model and speech_model.is_loaded:
                log("✅ Speech emotion model loaded successfully!")
            else:
                log("❌ Failed to load speech emotion model")
                return None
        except Exception as e:
            log(f"❌ Error initializing speech model: {str(e)}")
            return None
    return speech_model

def save_audio_file(audio_bytes, file_path):
    """Save raw audio bytes as properly formatted WAV file"""
    try:
        # First try to save as regular WAV
        with wave.open(file_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_bytes)
        return True
    except Exception as e:
        log(f"⚠️ First WAV save attempt failed: {str(e)}")
        try:
            # If that fails, try to load with librosa and resave
            import soundfile as sf
            import io
            with io.BytesIO(audio_bytes) as audio_io:
                data, sr = sf.read(audio_io)
                sf.write(file_path, data, 16000, subtype='PCM_16')
            return True
        except Exception as e2:
            log(f"❌ Error saving audio file: {str(e2)}")
            return False

def validate_audio_file(file_path):
    """More lenient audio validation"""
    try:
        # First try standard WAV validation
        with wave.open(file_path, 'rb') as wav_file:
            duration = wav_file.getnframes() / float(wav_file.getframerate())
            if duration < 0.1:  # Reduced minimum duration
                log(f"⚠️ Audio too short: {duration:.2f}s")
                return False
            if duration > 10:  # Maximum duration
                log(f"⚠️ Audio too long: {duration:.2f}s")
                return False
        return True
    except Exception as e:
        log(f"⚠️ Standard WAV validation failed: {str(e)}")
        try:
            # Fallback to librosa validation
            y, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            if duration < 0.1 or duration > 10:
                return False
            return True
        except Exception as e2:
            log(f"❌ Audio validation failed completely: {str(e2)}")
            return False

def process_audio_bytes_directly(audio_bytes):
    """Try to process audio directly from bytes without saving"""
    try:
        import soundfile as sf
        import io
        
        # Try to read directly from bytes
        with io.BytesIO(audio_bytes) as audio_io:
            audio, sr = sf.read(audio_io)
            
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
                
            # Normalize audio
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
                
            # Predict emotion
            model = get_speech_model()
            if model:
                result = model.predict_emotion(audio, sr)
                if result:
                    return [result]
                    
        return None
    except Exception as e:
        log(f"❌ Direct audio processing failed: {str(e)}")
        return None

def process_audio_with_local_model(audio_path):
    """Process audio with guaranteed valid WAV format"""
    model = get_speech_model()
    if not model:
        return None
        
    try:
        # Try multiple audio loading methods
        audio, sr = None, None
        try:
            import soundfile as sf
            audio, sr = sf.read(audio_path)
        except:
            try:
                audio, sr = librosa.load(audio_path, sr=16000)
            except Exception as e:
                log(f"❌ Failed to load audio: {str(e)}")
                return None

        # Ensure proper sample rate
        if sr != 16000:
            log(f"⚠️ Resampling from {sr}Hz to 16000Hz")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
            
        # Validate audio
        if len(audio) < 1600:
            log("⚠️ Audio too short, padding with zeros")
            audio = np.pad(audio, (0, max(0, 16000 - len(audio))))
            
        # Normalize audio
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
            
        # Predict emotion
        result = model.predict_emotion(audio, sr)
        print(result);
        if result:
            log(f"🔮 Local model prediction: {result['label']} (confidence: {result['score']:.2f})")
            return [result]
            
        return None
        
    except Exception as e:
        log(f"❌ Local model processing error: {str(e)}")
        return None

# HuggingFace API configuration
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

@sio.on("send_audio")
def receive_audio(sid, data):
    try:
        log(f"🎤 Received audio from {sid}")

        if "audio" not in data:
            raise ValueError("No audio data received")

        # Process base64 audio data
        audio_data = data["audio"]
        if "," in audio_data:
            audio_data = audio_data.split(",")[1]
            
        audio_bytes = base64.b64decode(audio_data)
        
        # Save temporary audio file
        audio_path = f"temp_audio_{sid}.wav"
        if not save_audio_file(audio_bytes, audio_path):
            log("⚠️ Trying fallback audio processing...")
            try:
                # Try direct processing without saving
                result = process_audio_bytes_directly(audio_bytes)
                if result:
                    sio.emit("emotion_result", {
                        "type": "speech",
                        "result": result,
                        "method": "direct-processing"
                    }, room=sid)
                    return
            except Exception as e:
                log(f"❌ Fallback processing failed: {str(e)}")
            
            raise ValueError("Failed to process audio file")

        log(f"💾 Saved audio to {audio_path} (size: {len(audio_bytes)} bytes)")

        if not validate_audio_file(audio_path):
            log("⚠️ Audio validation failed but attempting to process anyway")

        # Try HuggingFace API first
        api_success = False
        if HUGGINGFACE_API_KEY:
            max_retries = 2
            retries = 0
            
            while not api_success and retries < max_retries:
                try:
                    log("🌐 Attempting HuggingFace API...")
                    with open(audio_path, "rb") as f:
                        response = requests.post(
                            HUGGINGFACE_API_URL,
                            headers=HEADERS,
                            data=f,
                            timeout=10
                        )

                    if response.status_code == 200:
                        result = response.json()
                        if isinstance(result, list) and len(result) > 0:
                            log(f"📊 API result: {result[0]}")
                            sio.emit("emotion_result", {
                                "type": "speech",
                                "result": result,
                                "method": "api"
                            }, room=sid)
                            api_success = True
                    else:
                        log(f"⚠️ API Error {response.status_code}")
                        retries += 1
                        time.sleep(1)
                        
                except Exception as e:
                    log(f"⚠️ API attempt failed: {e}")
                    retries += 1
                    time.sleep(1)

        # Fallback to local model if API fails
        if not api_success:
            log("🔄 Falling back to local speech model...")
            result = process_audio_with_local_model(audio_path)
            
            if result:
                sio.emit("emotion_result", {
                    "type": "speech",
                    "result": result,
                    "method": "local-model"
                }, room=sid)
                log("📡 Local model results sent")
            else:
                log("❌ Both API and local model failed")
                sio.emit("error", {
                    "message": "Failed to process audio with both API and local model"
                }, room=sid)
            
        # Clean up
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception as e:
                log(f"⚠️ Couldn't remove temp file: {e}")

    except Exception as e:
        log(f"❌ Audio processing error: {e}")
        sio.emit("error", {"message": str(e)}, room=sid)

@sio.on("send_frame")
def receive_frame(sid, data):
    try:
        if face_model is None or face_cascade is None:
            sio.emit("error", {"message": "Face recognition models not loaded"}, room=sid)
            return
            
        log(f"📷 Received frame from {sid}")
        
        if "frame" not in data:
            raise ValueError("No frame data received")
            
        # Decode image
        frame_data = data["frame"]
        if "," in frame_data:
            frame_data = frame_data.split(",")[1]
            
        frame_bytes = base64.b64decode(frame_data)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Couldn't decode image")
            
        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
            resized_face = cv2.resize(face_roi, (48, 48))
            normalized_face = resized_face / 255.0
            reshaped_face = normalized_face.reshape(1, 48, 48, 1)
            
            # Predict emotion
            predictions = face_model.predict(reshaped_face)[0]
            emotion_idx = np.argmax(predictions)
            
            result["faces"].append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "emotion": EMOTION_LABELS[emotion_idx],
                "confidence": float(predictions[emotion_idx])
            })
            
            log(f"😊 Detected {EMOTION_LABELS[emotion_idx]} (confidence: {predictions[emotion_idx]:.2f})")
        
        # Send results
        sio.emit("emotion_result", result, room=sid)
        log(f"📡 Sent {len(faces)} face results")
        
    except Exception as e:
        log(f"❌ Frame processing error: {e}")
        sio.emit("error", {"message": str(e)}, room=sid)

@app.route('/health')
def health_check():
    models_loaded = {
        "face_model": face_model is not None,
        "face_cascade": face_cascade is not None,
        "speech_model": get_speech_model() is not None
    }
    return {
        "status": "running",
        "models_loaded": models_loaded
    }, 200

if __name__ == "__main__":
    log("🚀 Starting server on port 5000...")
    eventlet.wsgi.server(eventlet.listen(("0.0.0.0", 5000)), app)