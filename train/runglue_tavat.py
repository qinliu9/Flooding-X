"""
Script for running finetuning on glue tasks under Freelb.

Largely copied from:
    https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py
"""
import argparse
import logging
import os
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
import adversarial_utils as utils
from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import HuggingFaceDataset
from textattack.attack_results import SuccessfulAttackResult, MaximizedAttackResult, FailedAttackResult
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack import Attacker
from textattack import AttackArgs
logging.basicConfig(
    format="%(asctime)s-%(levelname)s-%(name)s-%(message)s",
    datefmt='%Y-%m-%d,%H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


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


def delta_embedding_init(args, model):
    vocab_size = model.config.vocab_size
    hidden_size = model.config.hidden_size
    delta_global_embedding = torch.zeros([vocab_size, hidden_size]).uniform_(-1, 1)
    # 30522 bert # 50265 roberta# 21128 bert-chinese
    dims = torch.tensor([hidden_size]).float()
    mag = args.adv_init_mag / torch.sqrt(dims)
    delta_global_embedding = (delta_global_embedding * mag.view(1, 1))
    delta_global_embedding = delta_global_embedding.to(model.device)
    return delta_global_embedding


def delta_lb_token(args, input_lengths, embeds_init, input_mask, delta_global_embedding, input_ids_flat, bs, seq_len):
    dims = input_lengths * embeds_init.size(-1)  # B x(768^(1/2))
    mag = args.adv_init_mag / torch.sqrt(dims)  # B
    delta_lb = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
    delta_lb = (delta_lb * mag.view(-1, 1, 1)).detach()
    gathered = torch.index_select(delta_global_embedding, 0, input_ids_flat)  # B*seq-len D
    delta_tok = gathered.view(bs, seq_len, -1).detach()  # B seq-len D
    denorm = torch.norm(delta_tok.view(-1, delta_tok.size(-1))).view(-1, 1, 1)  # norm in total degree?
    delta_tok = delta_tok / denorm  # B seq-len D  normalize delta obtained from global embedding
    return delta_lb, delta_tok


def main(args):
    print(args)
    set_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config = AutoConfig.from_pretrained(args.model_name, num_labels=args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)
    tokenizer = utils.add_task_specific_tokens_stabilizer_continuous(tokenizer, args.trigger_len)
    model.to(device)

    collator = utils.Stabilizer_Collator(pad_token_id=tokenizer.pad_token_id)
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
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        betas=betas,
        eps=args.adam_epsilon
    )

    # Use suggested learning rate scheduler
    # todo 要不要加入schedule?
    # Use suggested learning rate scheduler
    num_training_steps = len(train_dataset) * args.epochs // args.bsz
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, num_training_steps)
    output_dir = Path(os.path.join(args.ckpt_dir, 'TAVAT_{}_{}_{}_adv{}_epochs{}_test'
                                   .format(args.model_name, args.dataset_name, args.task_name, args.adv_steps, args.epochs)))
    if not output_dir.exists():
        print(f'Making checkpoint directory: {output_dir}')
        output_dir.mkdir(parents=True)
    elif not args.force_overwrite:
        raise RuntimeError('Checkpoint directory already exists.')

    best_accuracy = 0
    best_dev_epoch = 0
    delta_global_embedding = delta_embedding_init(args, model)
    for epoch in range(args.epochs):
        print('Training...')
        model.train()
        model.zero_grad()
        avg_loss = utils.ExponentialMovingAverage()
        for model_inputs, labels in tqdm(train_loader):
            del model_inputs['trigger_mask']
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            word_embedding_layer = model.get_input_embeddings()
            input_ids = model_inputs['input_ids']
            token_type_ids = model_inputs['token_type_ids']
            attention_mask = model_inputs['attention_mask']
            embeds_init = word_embedding_layer(input_ids)
            input_ids_flat = input_ids.contiguous().view(-1)
            input_mask = attention_mask.float()
            input_lengths = torch.sum(input_mask, 1)

            total_loss = 0
            bs, seq_len = embeds_init.size(0), embeds_init.size(1)
            delta_lb, delta_tok = delta_lb_token(args, input_lengths, embeds_init, input_mask, delta_global_embedding,
                                                 input_ids_flat, bs, seq_len)
            for astep in range(args.adv_steps):
                delta_lb.requires_grad_()
                delta_tok.requires_grad_()
                inputs_embeds = embeds_init + delta_lb + delta_tok
                batch = {'inputs_embeds': inputs_embeds,
                         'attention_mask': attention_mask,
                         'token_type_ids': token_type_ids}
                logits = model(**batch).logits
                loss = get_loss(logits, labels)
                loss = loss / args.adv_steps
                total_loss += loss.item()
                loss.backward(retain_graph=True)
                if astep == args.adv_steps - 1:
                    delta_tok = delta_tok.detach()
                    delta_global_embedding = delta_global_embedding.index_put_((input_ids_flat,), delta_tok, True)
                    break

                delta_lb_grad = delta_lb.grad.clone().detach()
                delta_tok_grad = delta_tok.grad.clone().detach()
                denorm_lb = torch.norm(delta_lb_grad.view(bs, -1), dim=1).view(-1, 1, 1)
                denorm_lb = torch.clamp(denorm_lb, min=1e-8)
                denorm_lb = denorm_lb.view(bs, 1, 1)
                denorm_tok = torch.norm(delta_tok_grad, dim=-1)
                denorm_tok = torch.clamp(denorm_tok, min=1e-8)
                denorm_tok = denorm_tok.view(bs, seq_len, 1)
                delta_lb = (delta_lb + args.adv_lr * delta_lb_grad / denorm_lb).detach()
                delta_tok = (delta_tok + args.adv_lr * delta_tok_grad / denorm_tok).detach()

                # calculate clip
                delta_norm_tok = torch.norm(delta_tok, p=2, dim=-1).detach()
                mean_norm_tok, _ = torch.max(delta_norm_tok, dim=-1, keepdim=True)
                reweights_tok = (delta_norm_tok / mean_norm_tok).view(bs, seq_len, 1)
                delta_tok = delta_tok * reweights_tok
                total_delta = delta_tok + delta_lb

                if args.adv_max_norm > 0:
                    delta_norm = torch.norm(total_delta.view(bs, -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > args.adv_max_norm).to(embeds_init)
                    reweights = (args.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                    delta_lb = (delta_lb * reweights).detach()
                    delta_tok = (delta_tok * reweights).detach()
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
        print('Evaluating...')
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for model_inputs, labels in dev_loader:
                model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                labels = labels.to(device)
                del model_inputs['trigger_mask']
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
            print('Best performance so far.')
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            best_accuracy = accuracy
            best_dev_epoch = epoch
    s = str(output_dir) + 'last'
    model.save_pretrained(s)
    tokenizer.save_pretrained(s)
    logger.info(f'Best dev metric: {best_accuracy} in Epoch: {best_dev_epoch}')
    if args.do_attack:
        model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
        attack = TextFoolerJin2019.build(model_wrapper)
        dataset = HuggingFaceDataset(args.dataset_name, args.task_name, split=args.valid)
        attack_args = AttackArgs(num_examples=args.num_examples,
                                 disable_stdout=True, random_seed=args.seed)
        attacker = Attacker(attack, dataset, attack_args)
        num_results = 0
        num_successes = 0
        num_failures = 0
        for result in attacker.attack_dataset():
            num_results += 1
            if (
                    type(result) == SuccessfulAttackResult
                    or type(result) == MaximizedAttackResult
            ):
                num_successes += 1
            if type(result) == FailedAttackResult:
                num_failures += 1
        print("[Succeeded / Failed / Total] {} / {} / {}".format(num_successes, num_failures, num_results))
        print(f'Original Accuracy: {(num_successes + num_failures) / num_results: 0.4f}, '
              f'Accuracy Under Attack: {num_failures / num_results: 0.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument("--dataset_name", default='glue', type=str)
    parser.add_argument("--task_name", default=None, type=str)
    parser.add_argument('--valid', type=str, default='validation')
    parser.add_argument('--trigger_len', default=0, type=int, help='trigger_len')
    parser.add_argument('--ckpt_dir', type=Path, default=Path('/root/outputs/freelb'))
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_attack', action="store_true")
    parser.add_argument('--bsz', type=int, default=16)
    parser.add_argument('--eval_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)  # different with normal finetune
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=500, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bias-correction', default=True)
    parser.add_argument('--weight-decay', type=float, default=1e-6)  # small value has little effect
    parser.add_argument('-f', '--force-overwrite', default=True)
    parser.add_argument('--debug', action='store_true')

    # Adversarial training specific
    parser.add_argument('--adv_steps', default=3, type=int,
                        help='Number of gradient ascent steps for the adversary')
    parser.add_argument('--adv_lr', default=0.03, type=float,
                        help='Step size of gradient ascent')
    parser.add_argument('--adv_init_mag', default=0.2, type=float,
                        help='Magnitude of initial (adversarial?) perturbation')
    parser.add_argument('--adv_max_norm', default=0.5, type=float,
                        help='adv_max_norm = 0 means unlimited')
    parser.add_argument('--adv_norm_type', default='l2', type=str,
                        help='norm type of the adversary')
    parser.add_argument('--adv_change_rate', default=0.2, type=float,
                        help='change rate of a sentence')
    parser.add_argument('--max_grad_norm', default=1, type=float, help='max gradient norm')
    parser.add_argument('--use_global_embedding', default=True, type=bool,
                        help='global embedding')
    # for attack
    # for attack
    parser.add_argument('--attack_method', type=str, default='textfooler')
    parser.add_argument("--pattern_id", default=3, type=int)
    parser.add_argument("--neighbour_vocab_size", default=10, type=int)
    parser.add_argument("--modify_ratio", default=0.0, type=float)
    parser.add_argument("--sentence_similarity", default=0.85, type=float)
    parser.add_argument("--num_examples", default=200, type=int)
    args = parser.parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    main(args)
