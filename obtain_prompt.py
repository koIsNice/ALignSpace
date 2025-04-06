from datasets import load_dataset
import random
import pickle
import json
from sentence_transformers import SentenceTransformer, util

def obtain_prompt(num_prompts):
    dataset = load_dataset("nvidia/HelpSteer2", split='train')
    num_rows = dataset.num_rows
    sample_idxs = random.sample(range(0, num_rows, 2), num_prompts)
    samples = dataset[sample_idxs]['prompt']

    with open('prompts.pkl', 'wb') as f:
        pickle.dump(samples, f)

def picking_desired_prompt(max_tokens=10):
    dataset = load_dataset("nvidia/HelpSteer2", split='train')['prompt']

    length_restricted_dataset = []
    for i in range(0, len(dataset), 2):
        if len(dataset[i].split()) <= max_tokens:
            length_restricted_dataset.append(dataset[i])

    device = 'cuda'
    model = SentenceTransformer('sentence-transformers/sentence-t5-base').to(device)
    base_text = model.encode(length_restricted_dataset[87]) # 'can you help me with data analysis ?'

    similar_prompts = []
    threshold = .78
    for prompt in length_restricted_dataset:
        emb = model.encode(prompt)
        if util.pytorch_cos_sim(emb, base_text) >= threshold:
            similar_prompts.append(prompt)

    print(f'number of prompts: {len(similar_prompts)}')

    with open('data/prompts_similar.pkl', 'wb') as f:
        pickle.dump(similar_prompts, f)

def list_to_json_file(l):

    with open("prompts.json", "w") as f:
        json.dump(l, f)

        
if __name__ == '__main__':

    # num_prompts = 1000
    # obtain_prompt(num_prompts)
    picking_desired_prompt()