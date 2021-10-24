import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from transformers import BertForSequenceClassification
from transformers import AlbertForSequenceClassification
from transformers import RobertaForSequenceClassification
from transformers import ElectraForSequenceClassification
from transformers import XLNetForSequenceClassification

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torch_optimizer as custom_optim

from utils.trainer import Trainer
from utils.data_loader import BertDataset, TokenizerWrapper


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', required=True)
    p.add_argument('--test_fn', required=True)

    p.add_argument('--pretrained_model_name', type=str, default='bert-base-uncased')
    
    p.add_argument('--gpu_id', type=int, default=-1)

    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--n_epochs', type=int, default=10)

    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--warmup_ratio', type=float, default=.1)
    p.add_argument('--adam_epsilon', type=float, default=1e-8)
    p.add_argument('--use_radam', action='store_true')

    p.add_argument('--max_length', type=int, default=100)

    config = p.parse_args()

    return config


def read_text(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        labels, texts = [], []
        for line in f:
            if line.strip() != '' and len(line.split('\t')) == 2:
                # The file should have tab delimited two columns.
                # First column indicates label field,
                # and second column indicates text field.

                label, text = line.split('\t')
                labels += [label.strip()]
                texts += [text.strip()]

    return labels, texts


def get_loaders(config, tokenizer):

    # Get list of labels and list of texts for training phase
    labels_for_train, texts_for_train = read_text(config.train_fn)

    # Generate label to index map.
    unique_labels = list(set(labels_for_train))
    label_to_index = {}
    index_to_label = {}
    for i, label in enumerate(unique_labels):
        label_to_index[label] = i
        index_to_label[i] = label

    # Convert label text to integer value.
    _labels_for_train = list(map(label_to_index.get, labels_for_train))

    # Shuffle before split into train and validation set.
    shuffled = list(zip(texts_for_train, _labels_for_train))
    random.shuffle(shuffled)
    texts_for_train = [e[0] for e in shuffled]
    _labels_for_train = [e[1] for e in shuffled]
    idx = int(len(texts_for_train) * .8)

    # Get dataloaders using given tokenizer as collate_fn.
    train_loader = DataLoader(
        BertDataset(texts_for_train[:idx], _labels_for_train[:idx]),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=TokenizerWrapper(tokenizer, config.max_length).collate,
    )
    valid_loader = DataLoader(
        BertDataset(texts_for_train[idx:], _labels_for_train[idx:]),
        batch_size=config.batch_size,
        collate_fn=TokenizerWrapper(tokenizer, config.max_length).collate,
    )


    # Get list of labels and list of texts for test phase
    labels_for_test, texts_for_test = read_text(config.test_fn)

    # Convert label text to integer value.
    _labels_for_test = list(map(label_to_index.get, labels_for_test))

    test_loader = DataLoader(
        BertDataset(texts_for_test, _labels_for_test),
        batch_size=config.batch_size,
        collate_fn=TokenizerWrapper(tokenizer, config.max_length).collate,
    )

    return index_to_label, train_loader, valid_loader, test_loader

def get_pretrained_language_model(model_name, index_to_label):

    if 'bert' in model_name:
        model = BertForSequenceClassification.from_pretrained(
            config.pretrained_model_name,
            num_labels=len(index_to_label)
        )
    elif 'albert' in model_name:
        model = AlbertForSequenceClassification.from_pretrained(
            config.pretrained_model_name,
            num_labels=len(index_to_label),
        )
    elif 'electra' in model_name:
        model = ElectraForSequenceClassification.from_pretrained(
            config.pretrained_model_name,
            num_labels=len(index_to_label)
        )
    elif 'roberta' in model_name:
        model = RobertaForSequenceClassification.from_pretrained(
            config.pretrained_model_name,
            num_labels=len(index_to_label)
        )
    elif 'xlnet' in model_name:
        model = XLNetForSequenceClassification.from_pretrained(
            config.pretrained_model_name,
            num_labels=len(index_to_label)
        )

    return model

def main(config):

    # Get pretrained tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)

    # Get dataloaders using tokenizer from untokenized corpus.
    index_to_label, train_loader, valid_loader, test_loader = get_loaders(config, tokenizer)

    # Get pretrained model with specified softmax layer.
    model = get_pretrained_language_model(config.pretrained_model_name, index_to_label)


    print(
        '|train| =', len(train_loader) * config.batch_size,
        '|valid| =', len(valid_loader) * config.batch_size,
        '|test| =', len(test_loader) * config.batch_size,
    )


    if config.use_radam:
        optimizer = custom_optim.RAdam(model.parameters(), lr=config.lr)
    else:
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.lr,
            eps=config.adam_epsilon
        )

    n_total_iterations = len(train_loader) * config.n_epochs
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        n_warmup_steps,
        n_total_iterations
    )


    if torch.cuda.is_available() and config.gpu_id >= 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        print(f'"""""" {n_gpu} of GPU ON """""" ')
    else:
        device = torch.device("cpu")
        print('"""""" CPU ON """""" ')

    model.to(device)

    # Start train.
    trainer = Trainer(config)
    model = trainer.train(
        model,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
        device,
    )

    trainer.test(
        model,
        test_loader,
        device,
    )

    torch.save({
        'model_name': config.pretrained_model_name,
        'finetuned_model': model.state_dict(),
        'config': config,
        'vocab': None,
        'classes': index_to_label,
        'tokenizer': tokenizer,
    }, config.model_fn)


if __name__ == '__main__':

    config = define_argparser()
    main(config)
