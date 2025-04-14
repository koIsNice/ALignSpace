import requests

def obtain_response(prompt):
    url = "https://api.deepinfra.com/v1/inference/meta-llama/Meta-Llama-3.1-8B-Instruct"
    api_key = 'api-key'
    input_text = f'<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "input": input_text
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

    except requests.exceptions.RequestException as e:
        print("Error:", e)

    return response

if __name__ == '__main__':
    
    prompt = "Can you teach me how to program?"
    text = obtain_response(prompt)
    
    print(text)