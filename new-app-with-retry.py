from flask import Flask, request, jsonify
from openai import OpenAI
import os
import re
import json
import time
import logging
from dotenv import load_dotenv

# .env file load
load_dotenv()

# Initialize OpenAI client with your API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set. Make sure you have a .env file with this key.")
client = OpenAI(api_key=api_key)

# Create Flask app
app = Flask(__name__)


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_json_from_response(text):
    """Extracts JSON content, potentially removing markdown fences."""
    match = re.search(r"```json\s*({.*?})\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    text = text.strip()
    if text.startswith('{') and text.endswith('}'):
        return text
    return text

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to predict if a name is realistic using OpenAI."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        name = data.get('name', '').strip()
        name = re.sub(r'\s+', ' ', name).strip() # Normalize spaces

        if not name:
            logger.warning("Received request with no name.")
            return jsonify({"error": "No name provided"}), 400


        # --- If Local Validation Passed, Proceed to OpenAI ---
        logger.info(f"'{name}' passed local checks. Proceeding to OpenAI validation.")

        prompt = f"""
        You are an expert in name classification. Determine if the name '{name}' is a realistic human name, used in any culture.
        Consider the name regardless of its capitalization. 

        A name is considered unrealistic if:
        * It contains characters other than letters (a-z, A-Z), spaces, hyphens (-), and dots (.).
        * It is less than three letter in total (excluding spaces, hyphens, and dots).

        Examples of realistic names: 'Mohiuddin Mohi', 'Aisha Khan', 'Sheik Kaykaus', 'Mr. Hanif Uddin', 'John-Doe', 'Mary.Anne Smith', 'm. a. h. hashan', 'p. k. kibria', 'M. k. robi mullah'.
        Examples of unrealistic names: 'Abdullah123'(contains numbers), 'Table Chair'(common phrase), 'Qwert' (Keyboard Pattern), 'asif.azad'(resembles email), 'm.ahmed'(resembles email).

        Respond in JSON format. If the name is 'Realistic', the JSON should only contain:
        {{
            "prediction": "Realistic"
        }}
        If the name is 'Not Realistic', the JSON should contain:
        {{
            "prediction": "Not Realistic",
            "reason": "<brief reason (max 50 character) why the name is not realistic>"
        }}
        """

        max_retries = 3
        prediction_result = None
        openai_error_message = None # Store potential OpenAI error
        ai_reason = None

        for attempt in range(max_retries):
            try:
                logger.info(f"Calling OpenAI API for name: '{name}', attempt {attempt + 1}")
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a precise name classification assistant outputting only JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=60,
                    temperature=0.0
                )

                reply_content = response.choices[0].message.content.strip()
                logger.info(f"Received raw response: {reply_content}")
                json_str_to_parse = extract_json_from_response(reply_content)

                try:
                    result = json.loads(json_str_to_parse)
                    prediction_value = result.get("prediction")

                    if prediction_value not in ["Realistic", "Not Realistic"]:
                        raise ValueError(f"Invalid prediction value from model: {prediction_value}")

                    prediction_result = prediction_value # Store the valid prediction
                    ai_reason = result.get("reason") if prediction_value == "Not Realistic" else None # Extract reason
                    logger.info(f"Successfully parsed prediction for '{name}': {prediction_result}, Reason: {ai_reason}")
                    openai_error_message = None # Clear any previous error message on success
                    break # Success

                except json.JSONDecodeError:
                    logger.error(f"Failed to decode JSON response: '{json_str_to_parse}'")
                    openai_error_message = "Invalid JSON response format from model"
                    # Let's break here, retrying might yield same malformed response
                    break
                except ValueError as ve:
                    logger.error(f"Prediction validation error: {str(ve)}")
                    openai_error_message = str(ve)
                     # Break here, invalid content received
                    break

            except Exception as api_error:
                logger.error(f"OpenAI API call failed (attempt {attempt + 1}/{max_retries}): {str(api_error)}")
                openai_error_message = f"OpenAI API error: {str(api_error)}" # Keep the last error
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"OpenAI API error after {max_retries} attempts.")
                    # Return error immediately after last retry fails
                    return jsonify({"error": openai_error_message}), 500

        # --- Process the final result after loop/retry ---
        if prediction_result == "Realistic":
            return jsonify({
                "name": name,
                "prediction": "Realistic",
                "reason": " "
                # No reason needed for realistic names
            }), 200
        elif prediction_result == "Not Realistic":
            return jsonify({
                "name": name,
                "prediction": "Not Realistic"
                #,"reason": ai_reason if ai_reason else "Reason not provided by AI."
            }), 200
        else:
            # This case handles if prediction_result is still None after retries
            # (e.g., due to JSON errors or API errors not caught by the immediate return)
            logger.error(f"Could not determine prediction for '{name}'. Last known error: {openai_error_message}")
            return jsonify({"error": f"Failed to get prediction. Last error: {openai_error_message or 'Unknown error after retries'}"}), 500


    except Exception as e:
        logger.exception(f"An unexpected server error occurred: {str(e)}")
        return jsonify({"error": f"An unexpected server error occurred"}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False) # Set debug=False for production