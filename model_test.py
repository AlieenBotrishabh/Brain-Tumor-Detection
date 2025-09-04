from transformers import pipeline

try:
    classifier = pipeline(
        "image-classification",
        model="Ace6868/brain-tumor-classifier",
        trust_remote_code=True
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
