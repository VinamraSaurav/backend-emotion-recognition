from transformers import Wav2Vec2Processor

# Set cache directory
cache_dir = "./models/wav2vec2"
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53", cache_dir=cache_dir)
