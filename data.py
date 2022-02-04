import os
import io
import json
from tqdm import tqdm
import torch
from gensim.corpora import Dictionary
from torch.utils.data import TensorDataset
import logging
from infos import *

logger = logging.getLogger(__name__)


def read_file(path):
    with open(path, 'r', encoding='utf-8') as f :
        data = f.read().splitlines()
    return data


def get_data(path):
    data = read_file(path)
    examples = {
        'words': [],
        'lengths': [],
        'postags': [],
        'heads': [],
        'labels': []
    }
    words, postags, heads, labels = ['<ROOT>'], ['<ROOT>'], [-1], ['<ROOT>']
    for raw in tqdm(data):
        if raw and raw[0] == '#':
            continue
        line = raw.strip()
        if not line:
            examples['words'].append(words)
            examples['postags'].append(postags)
            examples['lengths'].append(len(postags))
            examples['heads'].append(heads)
            examples['labels'].append(labels)
            words, postags, heads, labels = ['<ROOT>'], ['<ROOT>'], [-1], ['<ROOT>']
        else:
            infos = line.split('\t')
            # Skip dummy tokens used in ellipsis constructions, and multiword tokens.
            if '.' in infos[0] or '-' in infos[0]:
                continue
            words.append(infos[1])
            postags.append(infos[3])
            heads.append(int(infos[6]))
            labels.append(infos[7].split(':')[0])
    return examples


def load_dictionaries(lang='en', corpus='ewt'):
    dictionary_words = Dictionary.load(os.path.join(DIC_DIR, f'{lang}_{corpus}-words.dictionary'))
    dictionary_postags = Dictionary.load(os.path.join(DIC_DIR, f'postags.dictionary'))
    dictionary_labels = Dictionary.load(os.path.join(DIC_DIR, f'labels.dictionary'))
    return dictionary_words, dictionary_postags, dictionary_labels


def encode_to_id(sequence, dictionary, max_length, pad_token='<PAD>', mask=False):
    n = len(sequence)
    postags_id = [dictionary.token2id[s] for s in sequence]
    padding = [dictionary.token2id[pad_token]] * (max_length - n)
    res = (postags_id + padding, )
    if mask:
        mask = [1] * n + [0] * (max_length - n)
        res += (mask, )
    return res


def process_examples(examples, lang='en', corpus='ewt', max_length=160):
    dictionary_words, dictionary_postags, dictionary_labels = load_dictionaries(lang, corpus)
    encode_to_id_words = lambda w: encode_to_id(w, dictionary_words, max_length, mask=True)
    encode_to_id_postags = lambda w: encode_to_id(w, dictionary_postags, max_length)
    encode_to_id_labels = lambda w: encode_to_id(w, dictionary_labels, max_length)

    N = len(examples['words'])
    all_input_ids = []
    all_attention_mask = []
    all_lengths = []
    all_postags = []
    all_labels = []
    all_heads = []
    for i in tqdm(range(N)):
        n = len(examples['words'][i])
        words_id, mask = encode_to_id_words(examples['words'][i])
        all_input_ids.append(words_id)
        all_attention_mask.append(mask)
        all_lengths.append(n)
        all_postags.append(encode_to_id_postags(examples['postags'][i]))
        all_labels.append(encode_to_id_labels(examples['labels'][i]))
        all_heads.append(examples['heads'][i]+[-1] * (max_length - n))

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_lengths = torch.tensor(all_lengths, dtype=torch.long)
    all_postags = torch.tensor(all_postags, dtype=torch.long)
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    all_heads = torch.tensor(all_heads, dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_lengths, all_postags, all_heads, all_labels)
    return dataset


def load_and_cache_examples(args, split='train'):
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, f'cached_{args.lang}_{args.corpus}_{split}_{args.max_length}')
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        dataset = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        dir = list(filter(lambda x: args.corpus in x.lower(), LANGUAGE_LIST[args.lang][0]))[0]
        path = os.path.join(UD2_DIR, dir, f'{args.lang}_{args.corpus}-ud-{split}.conllu')
        examples = get_data(path)
        dataset = process_examples(examples, args.lang, args.corpus)

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(dataset, cached_features_file)
    return dataset


if __name__ == '__main__':
    print('MAIN')
    #dictionary_words = Dictionary.load(f'../cache/en_ewt_words.dictionary')
    #set_up_word_embedding(dictionary_words)
    lang = 'fr'
    corpus = 'gsd'
    split = 'test'
    dir = list(filter(lambda x: corpus in x.lower(), LANGUAGE_LIST[lang][0]))[0]
    path = os.path.join(UD2_DIR, dir, f'{lang}_{corpus}-ud-{split}.conllu')
    examples = get_data(path)

    #examples = get_data(PATHS['en_ewt']['train'])
    #dataset = load_and_cache_examples(examples)