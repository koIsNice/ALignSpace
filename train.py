import torch
import torch.nn as nn
import copy
import numpy as np
import gc
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from obtain_embedding import *

class Projection(nn.Module):
    def __init__(self, input_dim=768, output_dim=8):
        super(Projection, self).__init__()
        self.input_dim = input_dim
        self.linear1 = nn.Linear(input_dim, output_dim)

    def forward(self, emb):
        emb = self.linear1(emb)
        return emb
    

def contrastive_loss(prompt_batch, response_batch, temperature, batch_size):
    prompt_batch = prompt_batch / torch.norm(prompt_batch, dim=1, keepdim=True)
    response_batch = response_batch / torch.norm(response_batch, dim=1, keepdim=True)

    similarity = prompt_batch @ response_batch.T / torch.exp(temperature)
    labels = torch.arange(batch_size, device=device)

    loss1 = nn.CrossEntropyLoss()(similarity, labels)
    loss2 = nn.CrossEntropyLoss()(similarity.T, labels)
    return (loss1 + loss2) / 2


def loss_of_alignment(prompts, responses, prompt_encoder, response_encoder, projection_prompt, projection_response):
    prompts_embs = prompt_encoder.encode(prompts)
    prompts_embs = torch.tensor(prompts_embs, device=device)
    prompts_embs = projection_prompt(prompts_embs)
    responses_embs = response_encoder.encode(responses)
    responses_embs = torch.tensor(responses_embs, device=device)
    responses_embs = projection_response(responses_embs)

    prompts_embs_normalized = prompts_embs / torch.norm(prompts_embs, dim=1, keepdim=True)
    responses_embs_normalized = responses_embs / torch.norm(responses_embs, dim=1, keepdim=True)

    similarity = prompts_embs_normalized @ responses_embs_normalized.T

    diff = torch.abs(prompts_embs - responses_embs).mean().item() / (torch.abs(prompts_embs).mean() + torch.abs(responses_embs).mean()).item()/2.
    avg_similarity = torch.trace(similarity) / similarity.shape[0]    
    return diff, avg_similarity

def train_one_epoch(prompts, responses, prompt_encoder, response_encoder, projection_prompt, projection_response, temperature, batch_size, optimizer):
    num_data = prompts.shape[0]
    num_batch = num_data // batch_size if num_data % batch_size == 0 else (num_data // batch_size) + 1
    random_batches = np.random.permutation(num_data)
    
    # Compute contrastive loss
    # total_loss is only for recording
    total_loss = 0
    for n in range(num_batch):
        
        if n != num_batch - 1:
            prompt_batch = prompts[random_batches[n*batch_size:(n+1)*batch_size]]
            response_batch = responses[random_batches[n*batch_size:(n+1)*batch_size]]
        else:
            prompt_batch = prompts[random_batches[n*batch_size:]]
            response_batch = responses[random_batches[n*batch_size:]]

        prompt_batch = prompt_encoder.encode(prompt_batch)
        prompt_batch = torch.tensor(prompt_batch, device=device)
        prompt_batch = projection_prompt(prompt_batch)
        response_batch = response_encoder.encode(response_batch)
        response_batch = torch.tensor(response_batch, device=device)
        response_batch = projection_response(response_batch)

        if n != num_batch - 1:
            loss = contrastive_loss(prompt_batch, response_batch, temperature, batch_size)
        else:
            loss = contrastive_loss(prompt_batch, response_batch, temperature, num_data % batch_size)
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del prompt_batch
        del response_batch
        torch.cuda.empty_cache()

        with torch.no_grad():
            total_loss += loss

    return total_loss / num_batch

def train(prompts, responses, prompt_encoder, response_encoder, projection_prompt, projection_response, num_epoch, batch_size):
    temperature = nn.Parameter(torch.tensor(0.1))
    optimizer = optim.Adam(params=list(prompt_encoder.parameters()) + list(response_encoder.parameters()) + list(projection_prompt.parameters())+list(projection_response.parameters()) + [temperature])

    projection_prompt_best = None
    projection_response_best = None
    prompt_encoder_best = None
    response_encoder_best = None
    best_similarity = -1.

    num_data = prompts.shape[0]
    random_order = np.random.permutation(num_data)
    prompts_train = prompts[random_order][:int(0.9 * num_data)]
    prompts_val = prompts[random_order][int(0.9 * num_data):]
    responses_train = responses[random_order][:int(0.9 * num_data)]
    responses_val = responses[random_order][int(0.9 * num_data):]

    # pre-record the diff and similarity for validation set
    alignment_dff, alignment_sim = loss_of_alignment(prompts_val, responses_val, prompt_encoder, response_encoder, projection_prompt, projection_response)
    writer.add_scalar('alignment diff', alignment_dff, -1)
    writer.add_scalar('alignment average similarity', alignment_sim, -1)
    print(f'before training-> alignment diff: {alignment_dff}, alignment average similarity: {alignment_sim}')

    for t_epoch in range(num_epoch):
        avg_loss = train_one_epoch(prompts_train, responses_train, prompt_encoder, response_encoder, projection_prompt, projection_response, temperature, batch_size, optimizer)
        alignment_dff, alignment_sim = loss_of_alignment(prompts_val, responses_val, prompt_encoder, response_encoder, projection_prompt, projection_response)

        if best_similarity < alignment_sim.item():
            best_similarity = alignment_sim.item()
            del prompt_encoder_best
            del response_encoder_best
            del projection_prompt_best
            del projection_response_best
            prompt_encoder_best = copy.deepcopy(prompt_encoder)
            response_encoder_best = copy.deepcopy(response_encoder)
            projection_prompt_best = copy.deepcopy(projection_prompt)
            projection_response_best = copy.deepcopy(projection_response)

        writer.add_scalar('loss', avg_loss, t_epoch)    
        writer.add_scalar('alignment diff', alignment_dff, t_epoch)
        writer.add_scalar('alignment average similarity', alignment_sim, t_epoch)
        writer.add_scalar('temperature', temperature, t_epoch)
        print(f'epoch {t_epoch}, total loss:{avg_loss}, alignment diff: {alignment_dff}, alignment average similarity: {alignment_sim}')

    print('The training process is done.\n')

    # observe the similarity after projections
    alignment_dff, alignment_sim = loss_of_alignment(prompts_val, responses_val, prompt_encoder, response_encoder, projection_prompt_best, projection_response_best)
    print(f'after training-> alignment diff: {alignment_dff}, alignment average similarity: {alignment_sim}')

    return prompt_encoder_best, response_encoder_best, projection_prompt_best, projection_response_best

if __name__ == '__main__':

    writer = SummaryWriter("./training_tb_record")

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_emb_dim', type=bool, default=768)
    parser.add_argument('--output_emb_dim', type=bool, default=8)
    parser.add_argument('--num_epoch', default=30)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--gpu_device', default='cuda')

    args = parser.parse_args()
    input_dim = args.input_emb_dim
    output_dim = args.output_emb_dim
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    device = args.gpu_device

    prompts = load_pkl('data/prompts.pkl') + load_pkl('data/prompts_similar.pkl')
    responses_nontext = load_pkl('data/responses.pkl') + load_pkl('data/responses_similar.pkl')
    # prompts = load_pkl('data/prompts.pkl')
    # responses_nontext = load_pkl('data/responses.pkl')

    print(f'number of prompts/responses: {len(prompts)}')

    responses = []
    for response in responses_nontext:
        text = response.json()['results'][0]['generated_text']

        responses.append(text)

    # list to np
    prompts = np.array(prompts)
    responses = np.array(responses)

    prompt_encoder = SentenceTransformer('sentence-transformers/sentence-t5-base').to(device)
    response_encoder = SentenceTransformer('sentence-transformers/sentence-t5-base').to(device)
    projection_prompt = Projection(input_dim=input_dim, output_dim=output_dim).to(device)
    projection_response = Projection(input_dim=input_dim, output_dim=output_dim).to(device)

    # prompt_encoder.load_state_dict(torch.load(f'model2_{output_dim}/prompt_encoder.pt',map_location = torch.device('cuda')))
    # projection_prompt.load_state_dict(torch.load(f'model2_{output_dim}/projection_prompt.pt',map_location = torch.device('cuda')))
    # response_encoder.load_state_dict(torch.load(f'model2_{output_dim}/response_encoder.pt',map_location = torch.device('cuda')))
    # projection_response.load_state_dict(torch.load(f'model2_{output_dim}/projection_response.pt',map_location = torch.device('cuda')))

    prompt_encoder, response_encoder, projection_prompt, projection_response = train(prompts=prompts, responses=responses, prompt_encoder=prompt_encoder, response_encoder=response_encoder, projection_prompt=projection_prompt, 
                                                   projection_response=projection_response, num_epoch=num_epoch, batch_size= batch_size)
    

    torch.save(prompt_encoder.state_dict(), f'model2_{output_dim}/prompt_encoder.pt')
    torch.save(response_encoder.state_dict(), f'model2_{output_dim}/response_encoder.pt')
    torch.save(projection_prompt.state_dict(), f'model2_{output_dim}/projection_prompt.pt')
    torch.save(projection_response.state_dict(), f'model2_{output_dim}/projection_response.pt')