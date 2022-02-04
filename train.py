from __future__ import absolute_import, division, print_function
import sys, os, random, argparse, time, logging, json
from tqdm import tqdm
from tqdm import trange
import numpy as np
from torch.optim import AdamW, Adam
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import torch
from tensorboardX import SummaryWriter

from utils import ConfigBiaffine, DependencyEvaluator
from model import SyntacticTreeParser
from data import load_and_cache_examples, load_dictionaries
from infos import *

try:
    from transformers import WarmupLinearSchedule, get_constant_schedule_with_warmup
except:
    from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def comput_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return torch.tensor(total_norm)


def save_model(save_directory, model):
    output_model_file = os.path.join(save_directory, 'syntactic_tree_parser.pt')
    torch.save(model.state_dict(), output_model_file)
    logger.info("Model weights saved in {}".format(output_model_file))


def save_args_config(args, config):
    torch.save(args, os.path.join(args.saving, 'args.bin'))
    torch.save(config.__dict__, os.path.join(args.saving, 'config.json'))


def load_args_config(saving_path):
    args = torch.load(os.path.join(saving_path, 'args.bin'))
    config = ConfigBiaffine(**torch.load(os.path.join(saving_path, 'config.json')))
    return args, config


def train(args, model, train_dataset, val_dataset):
    """ Train the model """
    best_loss = 1000000000
    tb_writer = SummaryWriter(args.logs)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  drop_last=True)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    #AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    #if not args.no_sceduler:
    #   scheduler = get_constant_schedule_with_warmup(optimizer, args.warmup_steps, last_epoch=-1)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", 1000)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    model.do_train_generator = False

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            inputs = {'input_ids': batch[0].to(device).long(),
                      'attention_mask': batch[1].to(device).float(),
                      'postag_ids': batch[3].to(device).long(),
                      'heads': batch[-2].to(device).long(),
                      'labels': batch[-1].to(device).long()
                      }
            loss = model(**inputs)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tb_writer.add_scalar('training_loss', loss.item(), global_step)
            #for key, value in dict_metric.items():
            #    tb_writer.add_scalar(key, value, global_step)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                #if not args.no_sceduler:
                #    scheduler.step()  # Update learning rate schedule

                model.zero_grad()
                global_step += 1
                #if not args.no_sceduler:
                #    tb_writer.add_scalar('lr', optimizer.get_lr()[0], global_step)
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.save_steps:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, val_dataset)
                        for key, value in results.items():
                            tb_writer.add_scalar(f'eval_{key}', value, global_step)

                    # TODO : need to be tested save model with lower loss
                    if results['loss'] < loss:
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Take care of distributed/parallel training
                        save_model(args.saving, model_to_save)
                        #torch.save(args, os.path.join(args.output_dir, 'best_model_args.bin'))

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.saving, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    save_model(args.saving, model_to_save)
                    #torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

    tb_writer.close()



def evaluate(args, model, eval_dataset):
    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.train_batch_size, drop_last=True)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.train_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    all_head_prev = []
    all_lbl_prev = []
    all_head_golden = []
    all_lbl_golden = []
    all_attention = []
    step_eval = 0
    evaluator = DependencyEvaluator()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        step_eval += 1
        model.eval()
        with torch.no_grad():
            inputs = {'input_ids': batch[0].to(device).long(),
                      'attention_mask': batch[1].to(device).float(),
                      'postag_ids': batch[3].to(device).long(),
                      'heads': batch[-2].to(device).long(),
                      'labels': batch[-1].to(device).long()
                      }

            loss, head_preds, lbl_preds = model(**inputs)
            eval_loss += loss

            all_head_prev += [head_preds]
            all_head_golden += [inputs['heads'].cpu()]
            all_attention += [inputs['attention_mask'].cpu()]
            all_lbl_prev += [lbl_preds]
            all_lbl_golden += [inputs['labels'].cpu()]
            nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    results = {
        'loss': eval_loss.item()
    }
    logger.info(f"***** Eval Loss {eval_loss} *****")

    all_head_prev = torch.cat(all_head_prev)[:, 1:]
    all_head_golden = torch.cat(all_head_golden)[:, 1:]
    all_attention = torch.cat(all_attention)[:, 1:]
    all_lbl_prev = torch.cat(all_lbl_prev)[:, 1:]
    all_lbl_golden = torch.cat(all_lbl_golden).squeeze()[:, 1:]

    #assert torch.logical_and(all_attention.sum(-1) == (all_head_prev != 0).sum(-1))

    summary = evaluator.eval(all_attention, all_head_prev, all_lbl_prev, all_head_golden, all_lbl_golden)
    results.update(summary)

    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--no_sceduler", action='store_true')
    parser.add_argument("--overwrite_cache", action='store_true')
    parser.add_argument("--data_dir", default='./data')
    parser.add_argument("--logs", default='logs')
    parser.add_argument('--word_embedding_path', default='')
    parser.add_argument("--lang", default='en', type=str, required=False)
    parser.add_argument("--corpus", default='ewt', type=str, required=False)
    parser.add_argument("--output_dir", default='output_dir', type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--config_name", default="config_bigger_bigger_bigger.json", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--max_length", default=160, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--load_embedding", action='store_true')
    parser.add_argument("--load_from_checkpoint", action='store_true')
    parser.add_argument("--model_weight_path", type=str)
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--save_steps", default=1000, type=int)
    parser.add_argument("--num_train_epochs", default=20, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--seed", default=42, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")

    args = parser.parse_args()
    exp_dir = f'{time.strftime("%m%d-%H%M%S")}'
    args.logs = os.path.join(args.output_dir, 'logs', exp_dir)
    args.saving = os.path.join(args.output_dir, 'saving', exp_dir)
    os.makedirs(args.saving, exist_ok=True)

    if args.word_embedding_path:
        args.word_embedding_path = os.path.join(args.word_embedding_path, f'wiki.{args.lang}_{args.corpus}.align.pt')

    dictionary_words, dictionary_postags, dictionary_labels = load_dictionaries(args.lang, args.corpus)
    args.voc_size = len(dictionary_words)
    args.nb_pos_tag = len(dictionary_postags)
    args.nb_lbl = len(dictionary_labels)
    # Setup logging
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger.info("Process device: %s", device)

    # Set seed
    set_seed(args)
    #tokenizer = BertTokenizer.from_pretrained(args.bert_name, do_lower_case=False)
    #logger.info('Number of token {}'.format(tokenizer.vocab_size))
    #args.tokenizer = tokenizer
    """
    num_labels – integer, default 2. Number of classes to use when the model is a classification model (sequences/tokens)
    output_attentions – boolean, default False. Should the model returns attentions weights.
    output_hidden_states – string, default False. Should the model returns all hidden-states.
    """

    if not args.test:
        logger.info("START TRAIN")
        config = ConfigBiaffine()
        save_args_config(args, config)
        model = SyntacticTreeParser(args, config)
        model.to(device)

        train_dataset = load_and_cache_examples(args, 'train')
        dev_dataset = load_and_cache_examples(args, 'dev')
        global_step, tr_loss = train(args, model, train_dataset, dev_dataset)
    else:
        logger.info("START TEST")
        exp_dir = args.model_weight_path.split('/')[-2]
        saving_path = os.path.join(args.output_dir, 'saving', exp_dir)

        results_dir = os.path.join(args.output_dir, 'results', exp_dir)
        os.makedirs(results_dir, exist_ok=True)

        # Load model with right word embedding
        config = ConfigBiaffine(**torch.load(os.path.join(saving_path, 'config.json')))
        args.word_embedding_path = os.path.join(OUT_DIR, f'wiki.{args.lang}_{args.corpus}.align.pt')
        model = SyntacticTreeParser(args, config)
        # Warm up and exclude english word emnbedding
        param = torch.load(args.model_weight_path, map_location=device)
        param = {key: value for key, value in param.items() if not 'word_embdd' in key}
        model.load_state_dict(param, strict=False)
        model.to(device)

        test_dataset = load_and_cache_examples(args, 'test')
        results = evaluate(args, model, test_dataset)
        print(results)
        save_file_path = os.path.join(results_dir, f'{args.lang}_{args.corpus}.json')
        with open(save_file_path, 'w') as f:
            json.dump(results, f, indent=6)

    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # Saving
    logger.info("Saving model checkpoint to %s", args.output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    #model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    # model_to_save.save_pretrained(args.output_dir)
    #tokenizer.save_pretrained(args.output_dir)
    # Good practice: save your training arguments together with the trained model
    #torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))