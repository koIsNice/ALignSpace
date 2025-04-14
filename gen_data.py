import requests
import pickle 

def gen_data(input_path, output_path):
    prompts = load_prompts(input_path)

    collector = []

    for idx, prompt in enumerate(prompts):
        collector.append(obtain_response(prompt))

        if (idx + 1) % 10 == 0:
            print(f'have collected {idx + 1} responses')

    with open(output_path, 'wb') as f:
        pickle.dump(collector, f)

def load_prompts(file_name):
    with open(file_name, 'rb') as f:
        prompts = pickle.load(f)
    return prompts

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
    gen_data(input_path='data/prompts_similar.pkl', output_path='data/responses_similar.pkl')
    