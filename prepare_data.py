import os, io, json, torch, logging
from tqdm import tqdm
from gensim.corpora import Dictionary
from torch.utils.data import TensorDataset
from data import get_data
from multiprocessing import Pool, cpu_count
from glob import glob
from infos import *


def set_dictionaries(dir, lang='en', corpus='ewt'):
    splits = ['train', 'dev', 'test'] if lang == 'en' else ['test']
    splits = [f'{lang}_{corpus}-ud-{split}.conllu' for split in splits]
    examples = ()
    for split in splits:
        path = os.path.join(UD2_DIR, dir, split)
        examples += (get_data(path),)

    words = sum(map(lambda x: x['words'], examples), [])
    postags = sum(map(lambda x: x['postags'], examples), [])
    labels = sum(map(lambda x: x['labels'], examples), [])

    dictionary_words = Dictionary(words)
    dictionary_words.add_documents([['<SOS>'], ['<PAD>'], ['<EOS>']])
    dictionary_postags = Dictionary(postags)
    dictionary_labels = Dictionary(labels)

    dictionary_words.save(os.path.join(DIC_DIR, f'{lang}_{corpus}-words.dictionary'))
    dictionary_postags.save(os.path.join(DIC_DIR, f'{lang}_{corpus}-postags.dictionary'))
    dictionary_labels.save(os.path.join(DIC_DIR, f'{lang}_{corpus}-labels.dictionary'))


def load_dictionaries(lang='en', corpus='ewt'):
    dictionary_words = Dictionary.load(os.path.join(DIC_DIR, f'{lang}_{corpus}-words.dictionary'))
    dictionary_postags = Dictionary.load(os.path.join(DIC_DIR, f'postags.dictionary'))
    dictionary_labels = Dictionary.load(os.path.join(DIC_DIR, f'labels.dictionary'))
    return dictionary_words, dictionary_postags, dictionary_labels


def set_up_word_embedding(dictionary, lang='eng', corpus='ewt'):
    path = os.path.join(LIB_DIR, f'wiki.{lang}.align.vec')
    fin = io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    N = len(dictionary)
    embeddings = torch.zeros(N, d)
    voc = list(dictionary.values())
    cache = []
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        token = tokens[0]
        if token in voc:
            cache.append(token)
            embeddings[dictionary.token2id[token]] = torch.tensor(list(map(float, tokens[1:])))
            if len(cache) == N:
                break

    torch.save(embeddings, os.path.join(OUT_DIR, f'wiki.{lang}_{corpus}.align.pt'))


def merge_dictionaries(type):
    dict_paths = glob(os.path.join(DIC_DIR, f'*{type}*'))
    merge_dict = set()
    for path in dict_paths:
        d = Dictionary.load(path)
        merge_dict = merge_dict.union(set(d.values()))
    merge_dict = Dictionary([merge_dict])
    merge_dict.add_documents([['<PAD>']])
    merge_dict.save(os.path.join(DIC_DIR, f'{type}.dictionary'))


if __name__ == '__main__':

    def process_dictionary(lang):
        dirs, name = LANGUAGE_LIST[lang]
        for dir in dirs:
            corpus = dir.split('-')[-1].lower()
            set_dictionaries(dir, lang, corpus)

    def process_word_embedding(lang):
        dirs, name = LANGUAGE_LIST[lang]
        for dir in dirs:
            corpus = dir.split('-')[-1].lower()
            dictionary_words, _, _ = load_dictionaries(lang, corpus)
            set_up_word_embedding(dictionary_words, lang, corpus)

    print(f'count cpu {cpu_count()}')
    print('START PROCESSING DICTIONARY')
    with Pool(cpu_count()) as p:
        p.map(process_dictionary, LANGUAGE_LIST.keys())

    print('START MERGING DICTIONARY')
    merge_dictionaries('postags')
    merge_dictionaries('labels')

    #print('START PROCESSING WORD EMBEDDING')
    #with Pool(cpu_count()) as p:
    #    p.map(process_word_embedding, LANGUAGE_LIST.keys())
