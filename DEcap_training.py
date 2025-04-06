import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple
import sys
import json
import os
from train import *


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

class DeCap(nn.Module):

    def __init__(self, token_lengths ,prefix_size: int = 64):
        super(DeCap, self).__init__()

        # decoder: 4 layers transformer with 4 attention heads
        # the decoder is not pretrained
        with open('./decoder_config.pkl','rb') as f:
            config = pickle.load(f)
        self.decoder = GPT2LMHeadModel(config)
        self.embedding_size = self.decoder.transformer.wte.weight.shape[1]
        self.token_lengths = token_lengths
        # self.project = MLP((prefix_size, self.embedding_size,self.embedding_size * self.token_lengths))
        self.project = MLP((prefix_size, self.embedding_size * self.token_lengths))
        
    def forward(self, clip_features, gpt_tokens, mode=0):

        if mode == 0:
            embedding_text = self.decoder.transformer.wte(gpt_tokens)
            input_embeds = embedding_text
        else:
            embedding_clip = self.project(clip_features)
            embedding_clip = embedding_clip.reshape(-1, self.token_lengths,self.embedding_size)
            input_embeds = embedding_clip
        # embedding_cat = torch.cat([embedding_clip,embedding_text],dim=1)
        out = self.decoder(inputs_embeds = input_embeds)
        return out




def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


def train_decoder(responses, responses_tokens, response_encoder, projection_response, args, lr: float = 1e-5, warmup_steps: int = 1000, output_dir: str = ".", output_prefix: str = ""):

    batch_size = args.batch_size
    num_data = responses.shape[0]
    num_iteration = num_data // batch_size if num_data % batch_size == 0 else num_data // batch_size + 1
    num_epochs = args.num_epoch

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    args.is_master = args.local_rank == 0

    # set the device
    device = 'cuda'
    SEED=42
    tokens_per_response = responses_tokens.shape[-1]
    
    model = DeCap(token_lengths=tokens_per_response, prefix_size=output_dim).to(device)
    # model.load_state_dict(torch.load(f'model_{output_dim}/response_decoder.pt',map_location = torch.device('cuda')))
    best_model = None
    best_acc = .0
    
    loss_ce = torch.nn.CrossEntropyLoss(ignore_index=0,label_smoothing=0.1)
    optimizer = AdamW(model.parameters(),lr=lr)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_epochs 
    )
    
    
    for epoch in range(num_epochs):
        
        sys.stdout.flush()
        if args.is_master:
            print(f">>> Training epoch {epoch}")
            progress = tqdm(total=num_iteration, desc=output_prefix)

        total_ac = total_loss = 0
        for n in range(num_iteration):

            random_batches = np.random.permutation(num_data)

            if n != num_iteration - 1:
                # prompt_batch = prompts[random_batches[n*batch_size:(n+1)*batch_size]]
                response_batch = responses[random_batches[n*batch_size:(n+1)*batch_size]]
                responses_tokens_batch = responses_tokens[random_batches[n*batch_size:(n+1)*batch_size]]
            else:
                # prompt_batch = prompts[random_batches[n*batch_size:]]
                response_batch = responses[random_batches[n*batch_size:]]
                responses_tokens_batch = responses_tokens[random_batches[n*batch_size:]]

            with torch.no_grad():
                response_batch = response_encoder.encode(response_batch)
                response_batch = torch.tensor(response_batch, device=device)
                response_batch = projection_response(response_batch)

            # if epoch <= 250:
            #     outputs = model(response_batch.float(), responses_tokens_batch)
            # else:
            #     outputs = model(response_batch.float(), responses_tokens_batch, mode=1)
            outputs = model(response_batch.float(), responses_tokens_batch, mode=1)
            logits = outputs
            
            logits = logits.logits

            # logits = logits[:,: -1]
            responses_tokens_batch = responses_tokens_batch.flatten()
            logits = logits.reshape(-1, logits.shape[-1])
            
            loss_token = loss_ce(logits, responses_tokens_batch)
            ac=((logits.argmax(1)==responses_tokens_batch)*(responses_tokens_batch>0)).sum()/(responses_tokens_batch>0).sum()
            optimizer.zero_grad()
            loss_all = loss_token
            loss_all.backward()
            optimizer.step()
            scheduler.step()
                
            total_ac += ac.item()
            total_loss += loss_token.item()

            progress.set_postfix({"loss_token": total_loss/(n+1) , "acc_token":total_ac/(n+1)})
            progress.update()

        log_dir = './log/'+args.dataset+'.txt'
        with open(log_dir,'a+') as f:
            f.writelines('epoch ' +str(epoch) +': '+ progress.postfix+'\r\n')
        progress.close()

        if total_ac > best_acc:
            best_acc = total_ac
            best_model = copy.deepcopy(model)

        if total_ac/(n+1) >= .97:
            break

    torch.save(
        best_model.state_dict(),
        f'model2_{output_dim}/prompt_decoder.pt'
    )

    return model

def tokenize_data(model: SentenceTransformer, data):

    tokens_list = model.tokenizer(data, max_length=128, padding='longest', truncation=True, return_tensors="pt")['input_ids']
    return tokens_list

def main():
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

    responses_tokens = tokenize_data(response_encoder, responses).to(device)
    prompts_tokens = tokenize_data(prompt_encoder, prompts).to(device)

    prompts = np.array(prompts)
    responses = np.array(responses)

    train_decoder(prompts, prompts_tokens, prompt_encoder, projection_prompt, args, output_dir=args.out_dir, output_prefix=args.prefix)
    # train_decoder(responses, responses_tokens, response_encoder, projection_response, args, output_dir=args.out_dir, output_prefix=args.prefix)


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='./coco_model')
    parser.add_argument('--prefix', default='./coco_prefix', help='prefix for saved filenames')
    parser.add_argument('--dataset', default='test', help='coco or cc3m or bookcorpus')
    parser.add_argument('--num_epoch', default=10000)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--local_rank', type=int, default=0, metavar='N', help='Local process rank.') 
    parser.add_argument('--input_emb_dim', type=bool, default=768)
    parser.add_argument('--output_emb_dim', type=bool, default=8)
    args = parser.parse_args()
    
    input_dim = args.input_emb_dim
    output_dim = args.output_emb_dim

    main()

