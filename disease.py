import os
import sys
import logging
import io
from PIL import Image
from google import genai
from google.genai import types
 
# ========== Configuration ==========
 
API_KEY = "AIzaSyD_Y1y7oSbkvH-aKdFpOPbhZuPT_WS9o_U"
MODEL_NAME = "gemini-2.5-flash"
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("AgriImageDiagnosis")
 
# Initialize client: Gemini Developer API via API key
try:
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    logger.error(f"❌ Failed to initialize Gemini Client: {e}")
    sys.exit(1)
 
# ========== Helper Functions ==========
 
def is_agricultural_image(client, image_bytes: bytes) -> bool:
    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        "Is this image related to agriculture, such as a plant, crop, leaf, disease, pest, or insect? Reply ONLY 'Yes' or 'No'."
    ]
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
    )
    text = response.text or ""
    return "yes" in text.strip().lower()
 
def analyze_plant_image(client, image_bytes: bytes) -> str:
    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        (
            "You are an expert agricultural agent specializing in plant disease and pest diagnosis for farmers. "
            "Analyze the uploaded image. If you see any disease, pest, or issue, give an in‑depth, research‑backed analysis including:\n"
            "- The likely disease or pest\n"
            "- Visible symptoms\n"
            "- Likely causes and spread\n"
            "- Recommended actions, treatments, and prevention tips (practical advice for Indian farmers)\n"
            "If you cannot confidently diagnose, explain why and offer general tips for crop health in simple, clear language."
        )
    ]
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
    )
    return response.text.strip() if response and response.text else "No diagnosis was possible for this image."
 
def main(image_path: str):
    if not os.path.exists(image_path):
        logger.error(f"File does not exist: {image_path}")
        sys.exit(1)
 
    try:
        with Image.open(image_path) as image:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            image_bytes = buf.getvalue()
    except Exception as e:
        logger.error(f"Error opening image: {e}")
        sys.exit(1)
 
    if not is_agricultural_image(client, image_bytes):
        print("⚠️ Please upload an image related to farming, crops, plant disease, or pests.")
        return
 
    diagnosis = analyze_plant_image(client, image_bytes)
    print("\nDiagnosis & Advice:\n")
    print(diagnosis)
 
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python agri_diagnosis.py <path_to_image>")
        sys.exit(1)
    image_path = sys.argv[1]
    main(image_path)