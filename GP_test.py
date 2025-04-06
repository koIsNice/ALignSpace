from sklearn.gaussian_process import GaussianProcessRegressor
from train import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def GP_search(visited_points, current_point, timestep):
   

def beta_t(t=0):
    beta = 2
    return beta

def main(num_timesteps):
    prompt_encoder = SentenceTransformer('sentence-transformers/sentence-t5-base').to(device)
    response_encoder = SentenceTransformer('sentence-transformers/sentence-t5-base').to(device)
    projection_prompt = Projection(input_dim=input_dim, output_dim=output_dim).to(device)
    projection_response = Projection(input_dim=input_dim, output_dim=output_dim).to(device)

    prompt_encoder.load_state_dict(torch.load(f'model2_{output_dim}/prompt_encoder.pt',map_location = torch.device('cuda'), weights_only=True))
    projection_prompt.load_state_dict(torch.load(f'model2_{output_dim}/projection_prompt.pt',map_location = torch.device('cuda'), weights_only=True))
    response_encoder.load_state_dict(torch.load(f'model2_{output_dim}/response_encoder.pt',map_location = torch.device('cuda'), weights_only=True))
    projection_response.load_state_dict(torch.load(f'model2_{output_dim}/projection_response.pt',map_location = torch.device('cuda'), weights_only=True))

    prompts_pre = load_pkl('data/prompts_similar.pkl')
    responses_nontext = load_pkl('data/responses_similar.pkl')

    responses = []
    prompts = []
    for idx, response in enumerate(responses_nontext):
        text = response.json()['results'][0]['generated_text']

        if len(text.split()) <= 100:
         if True:
            responses.append(text)
            prompts.append(prompts_pre[idx])
    print(f'num. of responses: {len(responses)}')

    # randomly select a prompt
    selected_prompt = prompts[11]

    print(f'selected prompt: {selected_prompt}')
    prompt_emb = prompt_encoder.encode(selected_prompt)
    prompt_emb = torch.tensor(prompt_emb, device=device)

    visited_points = []
    # visited_points.append(prompt_emb)

    path = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    model = AutoModelForSequenceClassification.from_pretrained(path, device_map=device, 
                                trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)

    for t in range(num_timesteps):
        GP_search(visited_points=visited_points, current_point=prompt_emb, timestep=t)


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', default=100)
    parser.add_argument('--input_emb_dim', type=bool, default=768)
    parser.add_argument('--output_emb_dim', type=bool, default=8)
    args = parser.parse_args()
    
    input_dim = args.input_emb_dim
    output_dim = args.output_emb_dim
    num_timesteps = args.timesteps

    main(num_timesteps)
