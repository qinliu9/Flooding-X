"""
Script for running finetuning on glue tasks under Freelb.

Largely copied from:
    https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AdamW, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
)
from tqdm import tqdm
import utils as utils
from info_regularizer import (CLUB, InfoNCE)

logging.basicConfig(
    format="%(asctime)s-%(levelname)s-%(name)s-%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.

    From:
        https://github.com/uds-lsv/bert-stable-fine-tuning/blob/master/src/transformers/optimization.py
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_loss(logits, labels):
    loss = F.cross_entropy(logits, labels.squeeze(-1))
    return loss


def feature_ranking(grad, cl=0.5, ch=0.9):
    n = len(grad)
    import math
    lower = math.ceil(n * cl)
    upper = math.ceil(n * ch)
    norm = torch.norm(grad, dim=1)  # [seq_len]
    _, ind = torch.sort(norm)
    res = []
    for i in range(lower, upper):
        res += ind[i].item(),
    return res


def get_seq_len(batch):
    lengths = torch.sum(batch['attention_mask'], dim=-1)
    return lengths.detach().cpu().numpy()


def train_mi_upper_estimator(mi_upper_estimator, outputs, batch=None):
    hidden_states = outputs.hidden_states  # need to set config.output_hidden = True
    last_hidden, embedding_layer = hidden_states[-1], hidden_states[0]  # embedding layer: batch x seq_len x 768
    embeddings = []
    lengths = get_seq_len(batch)
    for i, length in enumerate(lengths):
        embeddings.append(embedding_layer[i, :length])
    embeddings = torch.cat(embeddings)  # [-1, 768]   embeddings without masks
    return mi_upper_estimator.update(embedding_layer, embeddings)


def get_local_robust_feature_regularizer(mi_estimator, args, outputs, local_robust_features):
    hidden_states = outputs.hidden_states  # need to set config.output_hidden = True
    last_hidden, embedding_layer = hidden_states[-1], hidden_states[0]  # embedding layer: batch x seq_len x 768
    sentence_embeddings = last_hidden[:, 0]  # batch x 768  # CLS
    local_embeddings = []
    global_embeddings = []
    for i, local_robust_feature in enumerate(local_robust_features):
        for local in local_robust_feature:
            local_embeddings.append(embedding_layer[i, local])
            global_embeddings.append(sentence_embeddings[i])
    lower_bounds = []
    from sklearn.utils import shuffle
    local_embeddings, global_embeddings = shuffle(local_embeddings, global_embeddings, random_state=args.info_seed)
    for i in range(0, len(local_embeddings), args.bsz):
        local_batch = torch.stack(local_embeddings[i: i + args.bsz])
        global_batch = torch.stack(global_embeddings[i: i + args.bsz])
        lower_bounds += mi_estimator(local_batch, global_batch),
    return -torch.stack(lower_bounds).mean()


def local_robust_feature_selection(args, batch, grad):
    """
    :param input_ids: for visualization, print out the local robust features
    :return: list of list of local robust feature posid, non robust feature posid
    """
    grads = []
    lengths = get_seq_len(batch)
    for i, length in enumerate(lengths):
        grads.append(grad[i, :length])
    indices = []
    nonrobust_indices = []
    for i, grad in enumerate(grads):
        indices.append(feature_ranking(grad, args.cl, args.ch))
        nonrobust_indices.append([x for x in range(lengths[i]) if x not in indices])
    return indices, nonrobust_indices


def main(args):
    logger.info(args)
    set_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = AutoConfig.from_pretrained(args.model_name, num_labels=args.num_labels, output_hidden_states=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=args.do_lower_case)
    model.to(device)
    hidden_size = model.config.hidden_size
    mi_upper_estimator = CLUB(hidden_size, hidden_size, beta=args.beta).to(model.device)
    mi_estimator = InfoNCE(hidden_size, hidden_size).to(model.device)
    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    train_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                              subset=args.task_name)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
    dev_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                            subset=args.task_name, split=args.valid)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)
    if args.bias_correction:
        betas = (0.9, 0.999)
    else:
        betas = (0.0, 0.000)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)] +
                      list(mi_estimator.parameters()),
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        betas=betas,
        eps=args.adam_epsilon,
        correct_bias=args.bias_correction
    )

    # Use suggested learning rate scheduler
    # todo 要不要加入schedule?
    # Use suggested learning rate scheduler
    num_training_steps = len(train_dataset) * args.epochs // args.bsz
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, num_training_steps)
    if args.dataset_name == 'imdb' or args.dataset_name == 'ag_news':
        output_dir = Path(os.path.join(args.ckpt_dir, 'infobert_{}_mag{}_adv-steps{}_adv-lr{}_epochs{}'
                                       .format(args.dataset_name, args.adv_init_mag,
                                               args.adv_steps, args.adv_lr, args.epochs)))
    else:
        output_dir = Path(os.path.join(args.ckpt_dir, 'infobert_{}-{}_mag{}_adv-steps{}_adv-lr{}_epochs{}'
                                       .format(args.dataset_name, args.task_name, args.adv_init_mag,
                                               args.adv_steps, args.adv_lr, args.epochs)))
    if not output_dir.exists():
        logger.info(f'Making checkpoint directory: {output_dir}')
        output_dir.mkdir(parents=True)
    elif not args.force_overwrite:
        raise RuntimeError('Checkpoint directory already exists.')
    log_file = os.path.join(output_dir, 'INFO.log')
    logger.addHandler(logging.FileHandler(log_file))
    best_accuracy = 0
    best_dev_epoch = 0
    for epoch in range(args.epochs):
        logger.info('Training...')
        model.train()
        model.zero_grad()
        avg_loss = utils.ExponentialMovingAverage()
        for model_inputs, labels in tqdm(train_loader):
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            word_embedding_layer = model.get_input_embeddings()

            total_loss, upperbound_loss, lowerbound_loss = 0.0, 0.0, 0.0
            input_ids = model_inputs['input_ids']
            attention_mask = model_inputs['attention_mask']
            embeds_init = word_embedding_layer(input_ids)

            # initialize delta
            delta = torch.zeros_like(embeds_init)
            if args.adv_init_mag > 0:
                input_mask = attention_mask.to(embeds_init)
                input_lengths = torch.sum(input_mask, 1)
                if args.adv_norm_type == "l2":
                    delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                    dims = input_lengths * embeds_init.size(-1)
                    mag = args.adv_init_mag / torch.sqrt(dims)
                    delta = (delta * mag.view(-1, 1, 1)).detach()
                elif args.adv_norm_type == "linf":
                    delta = torch.zeros_like(embeds_init).uniform_(-args.adv_init_mag,
                                                                   args.adv_init_mag) * input_mask.unsqueeze(2)

            for astep in range(args.adv_steps):
                # (0) forward
                delta.requires_grad_()
                batch = {'inputs_embeds': delta + embeds_init, 'attention_mask': attention_mask}
                outputs = model(**batch)
                logits = outputs.logits

                # (1) backward
                loss = F.cross_entropy(logits, labels.view(-1))
                loss = loss / args.adv_steps
                total_loss += loss.item()
                if mi_upper_estimator:
                    upper_bound = train_mi_upper_estimator(mi_upper_estimator, outputs, batch) / args.adv_steps
                    loss += upper_bound
                    upperbound_loss += upper_bound.item()
                loss.backward(retain_graph=True)
                delta_grad = delta.grad.clone().detach()
                if mi_estimator:
                    local_robust_features, _ = local_robust_feature_selection(args, batch, delta_grad)
                    lower_bound = get_local_robust_feature_regularizer(mi_estimator, args, outputs,
                                                                       local_robust_features) * args.alpha / args.adv_steps
                    lower_bound.backward()
                    lowerbound_loss += lower_bound.item()

                if astep == args.adv_steps - 1:  ## if no freelb, set astep = 1, adv_init=0
                    # further updates on delta
                    break

                # (3) update and clip
                if args.adv_norm_type == "l2":
                    denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                    denorm = torch.clamp(denorm, min=1e-8)
                    delta = (delta + args.adv_lr * delta_grad / denorm).detach()
                    if args.adv_max_norm > 0:
                        delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                        exceed_mask = (delta_norm > args.adv_max_norm).to(embeds_init)
                        reweights = (args.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                        delta = (delta * reweights).detach()
                elif args.adv_norm_type == "linf":
                    denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
                    denorm = torch.clamp(denorm, min=1e-8)
                    delta = (delta + args.adv_lr * delta_grad / denorm).detach()
                    if args.adv_max_norm > 0:
                        delta = torch.clamp(delta, -args.adv_max_norm, args.adv_max_norm).detach()
                embeds_init = word_embedding_layer(input_ids)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            avg_loss.update(total_loss)
        s = Path(str(output_dir) + '/epoch' + str(epoch))
        if not s.exists():
            s.mkdir(parents=True)
        model.save_pretrained(s)
        tokenizer.save_pretrained(s)
        logger.info('Evaluating...')
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for model_inputs, labels in dev_loader:
                model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                labels = labels.to(device)
                logits = model(**model_inputs).logits
                _, preds = logits.max(dim=-1)
                correct += (preds == labels.squeeze(-1)).sum().item()
                total += labels.size(0)
            accuracy = correct / (total + 1e-13)
        logger.info(f'Epoch: {epoch}, '
              f'Loss: {avg_loss.get_metric(): 0.4f}, '
              f'Lr: {optimizer.param_groups[0]["lr"]: .3e}, '
              f'Accuracy: {accuracy}')
        if accuracy > best_accuracy:
            logger.info('Best performance so far.')
            best_accuracy = accuracy
            best_dev_epoch = epoch
    logger.info(f'Best dev metric: {best_accuracy} in Epoch: {best_dev_epoch}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument("--dataset_name", default='glue', type=str)
    parser.add_argument("--task_name", default=None, type=str)
    parser.add_argument('--valid', type=str, default='validation')
    parser.add_argument('--trigger_len', default=0, type=int, help='trigger_len')
    parser.add_argument('--ckpt_dir', type=Path, default=Path('/root/outputs/'))
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_lower_case', type=bool, default=True)
    parser.add_argument('--do_attack', action="store_true")
    parser.add_argument('--bsz', type=int, default=16)
    parser.add_argument('--eval_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)  # different with normal finetune
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=500, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bias_correction', default=True)
    parser.add_argument('--weight_decay', type=float, default=1e-2)  # small value has little effect
    parser.add_argument('-f', '--force-overwrite', default=True)
    parser.add_argument('--debug', action='store_true')

    # Adversarial training specific
    parser.add_argument('--adv_steps', default=3, type=int,
                        help='Number of gradient ascent steps for the adversary')
    parser.add_argument('--adv_lr', default=0.04, type=float,
                        help='Step size of gradient ascent')
    parser.add_argument('--adv_init_mag', default=0.08, type=float,
                        help='Magnitude of initial (adversarial?) perturbation')
    parser.add_argument('--adv_max_norm', default=0, type=float,
                        help='adv_max_norm = 0 means unlimited')
    parser.add_argument('--adv_norm_type', default='l2', type=str,
                        help='norm type of the adversary')
    parser.add_argument('--adv_change_rate', default=0.2, type=float,
                        help='change rate of a sentence')
    parser.add_argument('--max_grad_norm', default=1, type=float, help='max gradient norm')
    parser.add_argument('--alpha', default=5e-3, type=float,
                        help='hyperparam of InfoNCE')
    parser.add_argument('--beta', default=5e-3, type=float,
                        help='hyperparam of Info upper bound')
    parser.add_argument('--cl', default=0.5, type=float,
                        help='lower bound of Local Anchored Feature Extraction')
    parser.add_argument('--ch', default=0.9, type=float,
                        help='lower bound of Local Anchored Feature Extraction')
    parser.add_argument('--info_seed', default=42, type=float,
                        help='seed for InfoBERT')
    # for attack
    parser.add_argument('--attack_method', type=str, default='textfooler')
    parser.add_argument("--pattern_id", default=3, type=int)
    parser.add_argument("--neighbour_vocab_size", default=10, type=int)
    parser.add_argument("--modify_ratio", default=0.0, type=float)
    parser.add_argument("--sentence_similarity", default=0.85, type=float)
    parser.add_argument("--num_examples", default=200, type=int)
    args = parser.parse_args()
    main(args)
