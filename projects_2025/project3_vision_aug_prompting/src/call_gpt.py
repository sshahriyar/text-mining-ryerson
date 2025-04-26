import requests
import json
import random
import time
import os
import base64
from datetime import datetime

# Load API keys and proxy configuration from key_list.json
with open('key_list.json', 'r', encoding='utf-8') as f:
    key_list = json.load(f)

base_url = 'api.openai.com'
current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
token_usage_filepath = os.path.join('token_usage', f'{current_time_str}.csv')

with open(token_usage_filepath, 'w', encoding='utf8') as f:
    f.write('time,prompt_tokens,completion_tokens,total_tokens,cost\n')

# Pricing map updated with the latest models and rates
model_pricing_map = {
    'gpt-3.5-turbo': {
        'input': 0.0010 / 1000,
        'output': 0.0020 / 1000,
    },
    'gpt-3.5-turbo-instruct': {
        'input': 0.0015 / 1000,
        'output': 0.0020 / 1000,
    },
    'gpt-4': {
        'input': 0.03 / 1000,
        'output': 0.06 / 1000,
    },
    'gpt-4-32k': {
        'input': 0.06 / 1000,
        'output': 0.12 / 1000,
    },
    'gpt-4-1106-preview': {
        'input': 0.01 / 1000,
        'output': 0.03 / 1000,
    },
    'gpt-4-vision-preview': {
        'input': 0.01 / 1000,
        'output': 0.03 / 1000,
    },
    'gpt-4o': {
        'input': 0.0025 / 1000,
        'output': 0.01 / 1000,
    },
    'gpt-4o-mini': {
        'input': 0.00015 / 1000,
        'output': 0.0006 / 1000,
    }
}

def encode_image(image_path):
    """
    Reads an image file and encodes it in base64.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_gpt(
        question, 
        model, 
        image_path=None, 
        temperature=1,
        max_tokens=None, 
        max_retries=5, 
        retry_delay=30):
    """
    Queries the OpenAI Chat Completions API.
    Supports both text and vision+text prompts.
    Implements exponential backoff in case of rate limiting (HTTP 429).
    """
    current_retry_delay = retry_delay  # initialize the retry delay

    for attempt in range(max_retries):
        # Select an API key randomly from the key_list
        current_index = random.randint(0, len(key_list) - 1)
        API_KYE = key_list[current_index]['API_KEY']
        PROXY = key_list[current_index]['PROXY']

        print('API KEY = ', API_KYE, flush=True)
        print('PROXY = ', PROXY)

        proxies = {
            'http': PROXY,
            'https': PROXY
        }
        url = f'https://{base_url}/v1/chat/completions'
        headers = {
            'Authorization': 'Bearer ' + API_KYE,
            'Content-Type': 'application/json'
        }

        # If an image is provided, encode it to base64 and append to the prompt (as a string)
        if image_path is not None:
            base64_image = encode_image(image_path)
            # Append the image data in Markdown-like format
            question += f"\nImage: data:image/jpeg;base64,{base64_image}"

        data = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ]
        }
        if max_tokens is not None:
            data['max_tokens'] = max_tokens

        # Uncomment this to debug the payload:
        # print("Payload:", json.dumps(data, indent=2))

        try:
            response = requests.post(url, headers=headers, proxies=proxies, json=data)
            response.raise_for_status()  # Raise an exception for HTTP errors
            response = response.json()

            current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            prompt_tokens = response["usage"]["prompt_tokens"]
            completion_tokens = response["usage"]["completion_tokens"]
            total_tokens = response["usage"]["total_tokens"]
            cost = model_pricing_map[model]['input'] * prompt_tokens + model_pricing_map[model]['output'] * completion_tokens
            with open(token_usage_filepath, 'a', encoding='utf8') as f:
                f.write(f'{current_time_str},{prompt_tokens},{completion_tokens},{total_tokens},{cost}\n')

            return response['choices'][0]['message']['content']
        except requests.HTTPError as e:
            # Check if the error status is 429 (rate limit)
            if response.status_code == 429:
                print('Request failed: 429 Too Many Requests. Retrying in', current_retry_delay, 'seconds...')
            else:
                print('Request failed:', e)
            if attempt < max_retries - 1:
                time.sleep(current_retry_delay)
                current_retry_delay *= 2  # exponential backoff
            else:
                raise
        except requests.RequestException as e:
            print('Request failed:', e)
            if attempt < max_retries - 1:
                print(f"Retrying in {current_retry_delay} seconds...")
                time.sleep(current_retry_delay)
                current_retry_delay *= 2
            else:
                raise
