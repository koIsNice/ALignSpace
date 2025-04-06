import pickle
import torch
import argparse
from sentence_transformers import SentenceTransformer

def obtain_embedding(tokenizer, model, sentence):
    encoding = tokenizer(
        sentence, return_tensors="pt"
    )  # Batch size 1
    output = model.encoder(
        input_ids=encoding["input_ids"], 
        attention_mask=encoding["attention_mask"], 
        return_dict=True
    )
    last_hidden_states = output.last_hidden_state
    sentence_emb = last_hidden_states.mean(dim=1)

    return sentence_emb.clone().detach()

def load_pkl(file_name:str):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def load_pt(file_name:str):
    return torch.load(f'{file_name}.pt')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='sentence-t5-base')
    parser.add_argument('--use_existing_prompts', type=bool, default=True)
    parser.add_argument('--use_existing_responses', type=bool, default=True)
    parser.add_argument('--use_existing_prompts_emb', type=bool, default=False)
    parser.add_argument('--use_existing_responses_emb', type=bool, default=False)
    
    args = parser.parse_args()
    model_name = args.model_name
    use_existing_prompts = args.use_existing_prompts
    use_existing_responses = args.use_existing_responses
    use_existing_prompts_emb = args.use_existing_prompts_emb
    use_existing_responses_emb = args.use_existing_responses_emb

    model = SentenceTransformer(f'sentence-transformers/{model_name}')

    # obtain prompts
    if use_existing_prompts:
        prompts = load_pkl('prompts')
    else:
        pass
    
    # obtain responses
    if use_existing_responses:
        responses = load_pkl('responses')
    else:
        pass
    
    # obtain embeddings for prompts
    if use_existing_prompts_emb:
        prompts_embs = load_pt('prompts_embeddings')
        print(f'shape of batch of embeddings: {prompts_embs.shape}')
    else:
        prompts_embs = model.encode(prompts)

        print('embeddings for prompts are generated.\n')
        print(f'shape of batch of embeddings: {prompts_embs.shape}')

        torch.save(torch.tensor(prompts_embs), 'prompts_embeddings.pt')


    if use_existing_responses_emb:
        prompts_embs = load_pt('responses_embeddings')
        print(f'shape of batch of embeddings: {prompts_embs.shape}')
    else:
        responses_batch = []
        for response in responses:
            responses_batch.append(response.json()['results'][0]['generated_text'])

        responses_embs = model.encode(responses_batch)

        print('embeddings for responses are generated.\n')
        print(f'shape of batch of embeddings: {responses_embs.shape}')

        torch.save(torch.tensor(responses_embs), 'responses_embeddings.pt')