from flask import Flask, request
from flask_cors import CORS
import traceback
import logging
import os
from consts import LLM_MODEL_NAME, PROMPT

from gpt4all import GPT4All

logger = logging.getLogger()

HOST = '0.0.0.0'
PORT = 8081

app = Flask(__name__)
CORS(app)

try:
    base_folder = os.path.dirname(__file__)
    base_folder = os.path.dirname(base_folder)
    gpt_models_folder = os.path.join(base_folder, "gpt_models/gpt4all/")
    model_folder = os.environ.get("MODEL_PATH", gpt_models_folder)
    llm_model = GPT4All(model_name=LLM_MODEL_NAME, model_path=model_folder)
except Exception:
    raise ValueError("Error loading LLM model.")

@app.route('/api/completion', methods=['POST'])
def completion():
    try:
        req = request.get_json()
        words = req.get('content')
        if not words:
            raise ValueError("No input word.")
        output = generate_text(words)
        return output, 200
    except Exception:
        logger.error(traceback.format_exc())
        return "Error", 500

def generate_text(content):
    prompt = PROMPT + f"\n{content}"

    with llm_model.chat_session():
        output = llm_model.generate(prompt, temp=0.7, max_tokens=1024)
        output = output.strip()

        return output


if __name__ == '__main__':
    # run web server
    app.run(host=HOST,
            debug=True,  # automatic reloading enabled
            port=PORT)
