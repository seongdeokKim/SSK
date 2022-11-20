import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data

from transformers import AutoTokenizer
from transformers import BertForSequenceClassification



def define_argparser():
    '''
    Define argument parser to take inference using fine-tuned model.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--max_length', type=int, default=320)
    p.add_argument('--top_k', type=int, default=1)

    config = p.parse_args()

    return config

def read_text():
    '''
    Read text from standard input for inference.
    '''
    lines = []

    for line in sys.stdin:
        if line.strip() != '':
            lines += [line.strip()]

    return lines

def main(config):
    saved_data = torch.load(
        config.model_fn,
        map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
    )

    train_config = saved_data['config']
    bert_best = saved_data['bert']
    index_to_label = saved_data['classes']

    lines = read_text()
    #lines = load_text(config.input_file)

    with torch.no_grad():
        # Declare model and load pre-trained weights.
        tokenizer = AutoTokenizer.from_pretrained(train_config.pretrained_model_name)
        model = BertForSequenceClassification.from_pretrained(train_config.pretrained_model_name,
                                                              num_labels=len(index_to_label))
        model.load_state_dict(bert_best)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
        device = next(model.parameters()).device

        # Don't forget turn-on evaluation mode.
        model.eval()

        y_hats = []
        for idx in range(0, len(lines), config.batch_size):
            mini_batch = tokenizer(lines[idx:idx + config.batch_size],
                                   max_length=config.max_length,
                                   padding=True,
                                   truncation=True,
                                   return_tensors="pt")

            input_ids = mini_batch['input_ids']
            input_ids = input_ids.to(device)
            attention_mask = mini_batch['attention_mask']
            attention_mask = attention_mask.to(device)

            # Take feed-forward
            logits = model(input_ids,
                           attention_mask=attention_mask)[0]
            current_y_hats = F.softmax(logits, dim=-1)

            y_hats += [current_y_hats]

        # Concatenate the mini-batch wise result
        y_hats = torch.cat(y_hats, dim=0)
        probs, indices = y_hats.cpu().topk(k=len(index_to_label))

        for i in range(len(lines)):
            sys.stdout.write('{}\t{}\t{}\n'.format(
                ",".join([index_to_label.get(int(j)) for j in indices[i][:config.top_k]]),
                ",".join([str(float(j))[:6] for j in probs[i][:config.top_k]]),
                lines[i],
            ))


if __name__ == '__main__':

    import time
    start = time.time()

    config = define_argparser()
    main(config)

    sys.stdout.write('\n{:.4f} seconds\n'.format(time.time()-start))
