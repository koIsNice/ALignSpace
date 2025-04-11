from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import tqdm
import torch
from DEcap_training import *

def create_support_memory(responses, response_encoder, projection_response):
    features = []
    captions = []
    batch_size = 10

    for i in tqdm(range(len(responses[:])//batch_size if len(responses[:]) % batch_size == 0 else len(responses[:])//batch_size + 1)):
        
        texts = responses[i*batch_size:(i+1)*batch_size]
        with torch.no_grad():
            # texts_token = tokenizer(texts, max_length=128, padding='longest', truncation=True, return_tensors="pt")['input_ids'].to(device)
            feature = torch.tensor(response_encoder.encode(texts), device=device)
            feature = projection_response(feature)
            features.append(feature)
            captions.extend(texts)

    features = torch.cat(features,dim=0)
    return features

def encoder(prompt, prompt_encoder, projection_prompt, support_memory):
    with torch.no_grad():
        prompt_features = torch.tensor(prompt_encoder.encode(prompt), device=device)
        prompt_features = projection_prompt(prompt_features)
        prefix_embedding = emb_aligner(prompt_emb=prompt_features, support_memory=support_memory)
    return prefix_embedding

def emb_aligner(prompt_emb, support_memory):
    with torch.no_grad():
        sim = (prompt_emb / prompt_emb.norm()) @ (support_memory / support_memory.norm(dim=-1, keepdim=True)).T.float()
        sim = (sim*100).softmax(dim=-1)
        prefix_embedding = sim @ support_memory.float()
    return prefix_embedding

def decoder(model, tokenizer, emb):

    with torch.no_grad():
        generated_text_from_Decap = Decoding(model, emb, tokenizer)
        generated_text_from_Decap = generated_text_from_Decap.replace('</s>','')

    return generated_text_from_Decap

def Decoding(model, feature, tokenizer):
    logits = model(feature, '', mode=1).logits
    logits = logits.reshape(-1, logits.shape[-1])
    predicted = tokenizer.decode(logits.argmax(1)).split('</s>')[0]

    return predicted

def GP_search(visited_points:list, 
              corresponding_scores:list, 
              current_point, reward_model, 
              tokenizer_reward_model,
              tokenizer_response,
              response_decoder, 
              prompt,
              timestep):

    # decode response
    response = decoder(response_decoder, tokenizer_response, current_point)

    message = [{"role": "user", "content": prompt},
            {"role": "assistant", "content": response}]
    
    input_ids = tokenizer_reward_model.apply_chat_template(message, return_tensors="pt").to(device)

    with torch.no_grad():
        score = reward_model(input_ids).score.item()

    # print(f'score: {score}')

    visited_points.append(current_point.cpu())
    corresponding_scores.append(score)

    # kernel = 1.0 * RBF(length_scale=1e-2, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-10, 1e1))
    kernel = RationalQuadratic()
    gp = GaussianProcessRegressor(kernel=kernel)
    gp.fit(visited_points, corresponding_scores)

    # create some points around current point
    random_points = torch.normal(mean=current_point, std= 0.02 * torch.ones([16, current_point.shape[0]], device=device)).cpu()
    mean, std = gp.predict(random_points, return_std=True)

    # print(mean, std)

    idx = np.argmax(mean + std)
    return random_points[idx, :].to(device), response, score
    

def main(num_timesteps, input_dim, output_dim):
    prompt_encoder = SentenceTransformer('sentence-transformers/sentence-t5-base').to(device)
    response_encoder = SentenceTransformer('sentence-transformers/sentence-t5-base').to(device)
    projection_prompt = Projection(input_dim=input_dim, output_dim=output_dim).to(device)
    projection_response = Projection(input_dim=input_dim, output_dim=output_dim).to(device)

    prompt_encoder.load_state_dict(torch.load(f'model2_{output_dim}/prompt_encoder.pt',map_location = torch.device('cuda'), weights_only=True))
    projection_prompt.load_state_dict(torch.load(f'model2_{output_dim}/projection_prompt.pt',map_location = torch.device('cuda'), weights_only=True))
    response_encoder.load_state_dict(torch.load(f'model2_{output_dim}/response_encoder.pt',map_location = torch.device('cuda'), weights_only=True))
    projection_response.load_state_dict(torch.load(f'model2_{output_dim}/projection_response.pt',map_location = torch.device('cuda'), weights_only=True))


    # load dataset
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

    # response decoder
    responses_tokens = tokenize_data(response_encoder, responses).to(device)
    response_decoder = DeCap(token_lengths=responses_tokens.shape[-1], prefix_size=output_dim).to(device)
    response_decoder.load_state_dict(torch.load(f'model2_{output_dim}/response_decoder.pt',map_location= torch.device('cuda')))
    response_decoder = response_decoder.eval()
    tokenizer_response = response_encoder.tokenizer

    # prompt decoder
    prompts_tokens = tokenize_data(prompt_encoder, prompts).to(device)
    prompt_decoder = DeCap(token_lengths=prompts_tokens.shape[-1], prefix_size=output_dim).to(device)
    prompt_decoder.load_state_dict(torch.load(f'model2_{output_dim}/prompt_decoder.pt',map_location= torch.device('cuda')))
    prompt_decoder = prompt_decoder.eval()
    # tokenizer_prompt = prompt_encoder.tokenizer

    # support memory
    support_memory = create_support_memory(responses=responses, response_encoder=response_encoder, projection_response=projection_response)

    # reward model
    path = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    reward_model = AutoModelForSequenceClassification.from_pretrained(path, device_map=device, 
                                trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer_reward_model = AutoTokenizer.from_pretrained(path, use_fast=True)


    # randomly select a prompt
    # selected_prompt = prompts[11]
    best_scores = []
    initial_scores = []
    record_file_path = 'log/record2.txt'
    for selected_prompt in prompts:

        print(f'selected prompt: {selected_prompt}')
        prompt_emb = encoder(prompt=selected_prompt, prompt_encoder=prompt_encoder, projection_prompt=projection_prompt, support_memory=support_memory)
        #aligned_emb = emb_aligner(prompt_emb, support_memory)
        aligned_emb = torch.normal(torch.zeros(prompt_emb.shape), std=.01).to(device)
        # initial response
        response = decoder(response_decoder, tokenizer_response, aligned_emb)
        print(f'--------------------\ncurrent response:\n{response}\n--------------------')

        visited_points = []
        corresponding_scores = []
        best_aligned_emb = None
        best_score = -1.

        for t in range(num_timesteps):
            aligned_emb, response, score = GP_search(visited_points=visited_points, 
                                corresponding_scores=corresponding_scores,
                                current_point=aligned_emb, 
                                reward_model=reward_model, 
                                tokenizer_reward_model=tokenizer_reward_model, 
                                tokenizer_response=tokenizer_response,
                                response_decoder=response_decoder, 
                                prompt=selected_prompt,
                                timestep=t)
            
            if score > best_score:
                best_score = score
                best_aligned_emb = aligned_emb

            if t == 0:
                initial_scores.append(score)
                with open(record_file_path, 'a+') as file:
                    file.write(f'Prompt: {selected_prompt}\n')
                    file.write(f'Inital response: {response}\nScore: {score}\n')

            if (t + 1) % 100 == 0:
                print(f'--------------------\ncurrent response:\n{response}\nscore: {score}\n--------------------')
                with open(record_file_path, 'a+') as file:
                    file.write(f'{t+1}th response: {response}\nScore: {score}\n')
        
        best_response = decoder(response_decoder, tokenizer_response, best_aligned_emb)
        print(f'--------------------\nbest response:\n{best_response}\nscore: {best_score} at t={t}\n--------------------')
        with open(record_file_path, 'a+') as file:
            file.write(f'best response: {best_response}\nscore: {best_score}\n\n')

        best_scores.append(best_score)
    print((sum(best_scores) - sum(initial_scores)) / len(best_scores))
    with open(record_file_path, 'a+') as file:
        file.write(f'num of iterations: {len(best_scores)}\n')
        file.write(f'avg. best score: {sum(best_scores) / len(best_scores)}\n')
        file.write(f'avg. initial score: {sum(initial_scores) / len(best_scores)}\n')
        file.write(f'avg. scores difference: {(sum(best_scores) - sum(initial_scores)) / len(best_scores)}\n')

if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', default=300)
    parser.add_argument('--input_emb_dim', type=bool, default=768)
    parser.add_argument('--output_emb_dim', type=bool, default=8)
    args = parser.parse_args()
    
    input_dim = args.input_emb_dim
    output_dim = args.output_emb_dim
    num_timesteps = args.timesteps

    main(num_timesteps, input_dim, output_dim)
