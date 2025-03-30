import os
import logging
import torch
import torch.nn as nn
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Model configuration
BASE_MODEL_NAME = "facebook/wav2vec2-base-960h"  # For feature extraction
MODEL_DIR = "./saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

class EmotionClassifier(nn.Module):
    """Simple neural network for emotion classification"""
    def __init__(self, input_size, num_emotions):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_emotions)
        )
    
    def forward(self, x):
        return self.layers(x)

class SpeechEmotionModel:
    def __init__(self):
        self.feature_extractor = None
        self.base_model = None
        self.emotion_classifier = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        self.is_loaded = False
        
    def download_and_save_models(self):
        """Download and save models to disk"""
        try:
            logger.info("Downloading base wav2vec2 model...")
            
            # Download feature extractor and base model
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(BASE_MODEL_NAME)
            base_model = Wav2Vec2Model.from_pretrained(BASE_MODEL_NAME)
            
            # Save models
            feature_extractor.save_pretrained(MODEL_DIR)
            base_model.save_pretrained(MODEL_DIR)
            
            # Initialize and save emotion classifier
            classifier = EmotionClassifier(768, len(self.emotions))
            torch.save(classifier.state_dict(), os.path.join(MODEL_DIR, "emotion_classifier.pt"))
            
            logger.info("All models saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading models: {e}")
            return False
    
    def load(self):
        """Load all models from disk or download if not available"""
        try:
            # Check if models exist
            if not all(os.path.exists(os.path.join(MODEL_DIR, f)) 
                      for f in ["config.json", "pytorch_model.bin", "preprocessor_config.json"]):
                logger.info("Models not found locally. Downloading...")
                if not self.download_and_save_models():
                    raise Exception("Failed to download models")
            
            # Load models
            logger.info("Loading feature extractor...")
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR)
            
            logger.info("Loading base model...")
            self.base_model = Wav2Vec2Model.from_pretrained(MODEL_DIR)
            
            logger.info("Loading emotion classifier...")
            self.emotion_classifier = EmotionClassifier(768, len(self.emotions))
            self.emotion_classifier.load_state_dict(
                torch.load(os.path.join(MODEL_DIR, "emotion_classifier.pt"))
            )
            
            # Move to device
            self.base_model.to(self.device)
            self.emotion_classifier.to(self.device)
            
            self.is_loaded = True
            logger.info("All models loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.is_loaded = False
            return False
    
    def extract_features(self, audio_array, sampling_rate=16000):
        """Extract meaningful features from audio"""
        if not self.is_loaded and not self.load():
            return None
        
        try:
            # Preprocess audio
            inputs = self.feature_extractor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Extract features
            with torch.no_grad():
                outputs = self.base_model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                
            # Average over time dimension
            features = torch.mean(last_hidden_state, dim=1)
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None
    
    def predict_emotion(self, audio_array, sampling_rate=16000):
        """Predict emotion from audio with proper feature extraction"""
        if not self.is_loaded and not self.load():
            return None
        
        try:
            # Validate input
            if len(audio_array) < 1600:  # At least 0.1 second at 16kHz
                logger.warning("Audio too short for reliable prediction")
                audio_array = np.pad(audio_array, (0, max(0, 1600 - len(audio_array))))
            
            # Normalize audio
            audio_array = audio_array / np.max(np.abs(audio_array))
            
            # Extract features
            features = self.extract_features(audio_array, sampling_rate)
            if features is None:
                return None
                
            # Predict emotion
            with torch.no_grad():
                logits = self.emotion_classifier(features)
                probs = torch.softmax(logits, dim=-1)
                confidence, pred_idx = torch.max(probs, dim=-1)
                
            return {
                "label": self.emotions[pred_idx.item()],
                "score": confidence.item()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None

# Singleton instance with lazy loading
_speech_model_instance = None

def get_model():
    """Get or initialize the speech emotion model (singleton pattern)"""
    global _speech_model_instance
    
    if _speech_model_instance is None:
        _speech_model_instance = SpeechEmotionModel()
        
        # Try loading existing models first
        if not _speech_model_instance.load():
            # If loading fails, try downloading
            logger.warning("Failed to load existing models. Attempting download...")
            if not _speech_model_instance.download_and_save_models():
                logger.error("Failed to initialize speech emotion model")
                return None
                
            # Try loading again after download
            _speech_model_instance.load()
    
    return _speech_model_instance if _speech_model_instance.is_loaded else None

if __name__ == "__main__":
    # Test the model loading and prediction
    logger.info("Testing speech emotion model...")
    
    model = get_model()
    if model and model.is_loaded:
        logger.info("✅ Model loaded successfully!")
        
        # Test with dummy audio
        dummy_audio = np.random.randn(16000) * 0.1  # 1 second of quiet audio
        result = model.predict_emotion(dummy_audio)
        
        if result:
            logger.info(f"Test prediction: {result['label']} (confidence: {result['score']:.2f})")
        else:
            logger.error("❌ Failed to make test prediction")
    else:
        logger.error("❌ Failed to load model")