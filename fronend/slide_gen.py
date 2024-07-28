import json
import os

from slide_deck import SlideDeck
from llm_call import chat_completion_request

FOLDER = "generated"

if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)

def generate_json_list_of_slides(content):
    try:
        resp = chat_completion_request(content)
        obj = json.loads(resp)
        return obj
    except Exception as e:
        raise e

def generate_presentation(content):
    deck = SlideDeck()
    slides_data = generate_json_list_of_slides(content)
    title_slide_data = slides_data[0]
    slides_data = slides_data[1:]
    return deck.create_presentation(title_slide_data, slides_data)
