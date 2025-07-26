from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import vertexai
from dotenv import load_dotenv
import base64
import io
from PIL import Image
from disease import (client,is_agricultural_image,analyze_plant_image)
from actions import (
    create_session,
    delete_session,
    get_session,
    list_deployments,
    list_sessions,
    send_message,
)

# Load env variables from .env file
load_dotenv()

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "agentic-ai-day-466410")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
STAGING_BUCKET = os.environ.get("GOOGLE_CLOUD_STAGING_BUCKET", "gs://jaikisaan")

# Initialize Vertex AI SDK
vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
    staging_bucket=STAGING_BUCKET,
)

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

def extract_last_text_response(response):
    for event in reversed(response):
        try:
            parts = event.get("content", {}).get("parts", [])
            for part in parts:
                if "text" in part:
                    return part["text"]
        except Exception as e:
            print("Error parsing event:", e)
    return "No text response from agent."


@app.route('/api/ask-agent', methods=['POST'])
def ask_agent():
    data = request.get_json()
    user_input = data.get('message')
    user_id = "frontend-user"

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Step 1: List deployments
        deployments = list_deployments()
        if not deployments:
            return jsonify({"error": "No deployments found"}), 500

        resource_id = deployments[0].resource_name

        # Step 2: Use existing or create session
        sessions = list_sessions(resource_id, user_id)
        session = sessions[0] if sessions else create_session(resource_id, user_id)
        session_id = session["id"]

        # Step 3: Send message
        response = send_message(resource_id, user_id, session_id, message=user_input)

        # if isinstance(response, list) and response:
        #     #reply_text = response[0].get("text", "No response content")
        #     reply_text= response[0].get("text")
        # else:
        #     reply_text = "No valid response from agent."
        reply_text = extract_last_text_response(response)
        return jsonify({"response": reply_text})

        return jsonify({"response": reply_text})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500
@app.route('/api/diagnose-image', methods=['POST'])
def diagnose_image():
    data = request.get_json()
    image_base64 = data.get('image_base64')

    if not image_base64:
        return jsonify({'error': 'Missing image_base64'}), 400

    try:
        image_bytes = base64.b64decode(image_base64)

        # Optional: Check if image is agri-related
        if not is_agricultural_image(client, image_bytes):
            return jsonify({'response': '⚠️ Please upload a plant/crop-related image.'})

        result = analyze_plant_image(client, image_bytes)
        return jsonify({'response': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)