import requests

URL = "http://127.0.0.1:8081"

CHAT_API_ENDPOINT = f"{URL}/api/completion"

def chat_completion_request(content):
    headers = {'Content-type': 'application/json'}
    data = {'content': content}

    req = requests.post(url=CHAT_API_ENDPOINT, headers=headers, json=data)
    json_extracted = req.text
    return json_extracted
