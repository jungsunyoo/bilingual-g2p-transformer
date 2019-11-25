# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 12:14:29 2019

@author: Affinity
"""

import re
import Levenshtein as Lev
from text.korean import ALL_SYMBOLS, jamo_to_korean
from jamo import hangul_to_jamo, h2j, j2hcj
import pdb

_sos = '<s>'
_eos = '</s>'
_pad = '<pad>'

_punc = '!\'(),-.:;? '



_arpabet_characters = [
    'AA', 'AE', 'AH',
    'AO', 'AW', 'AY',
    'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY',
    'F', 'G', 'HH', 'IH', 'IY',
    'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY',
    'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW',
    'V', 'W', 'Y', 'Z', 'ZH'
]
arpabet_symbols = [_pad, _sos, _eos] + list(_punc) + _arpabet_characters

#arpabet_to_id = {s: i for i, s in enumerate(arpabet_symbols)}
#id_to_arpabet = {i: s for i, s in enumerate(arpabet_symbols)}

_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')
_whitespace_re = re.compile(r'\s+')


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)

def load_vocab():
    '''Loads vocabulary file and returns idx<->token maps
    vocab_fpath: string. vocabulary file path.
    Note that these are reserved
    0: <pad>, 1: <unk>, 2: <s>, 3: </s>

    Returns
    two dictionaries.
    '''

    vocab = ["<pad>", "<unk>", "<s>", "</s>", " ", "◎"] + list("abcdefghijklmnopqrstuvwxyz") + ['AA', 'AE', 'AH', 'AO',
                                                                                                'AW', 'AY', 'B', 'CH',
                                                                                                'D', 'DH', 'EH', 'ER',
                                                                                                'EY', 'F', 'G', 'HH',
                                                                                                'IH', 'IY', 'JH', 'K',
                                                                                                'L', 'M', 'N', 'NG',
                                                                                                'OW', 'OY', 'P', 'R',
                                                                                                'S', 'SH', 'T', 'TH',
                                                                                                'UH', 'UW', 'V', 'W',
                                                                                                'Y', 'Z', 'ZH']
    # yjs added <space> token for delineating words ; space = ◎
    kor = ALL_SYMBOLS
    kor = list(kor)
    vocab += kor

    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}
    return token2idx, idx2token


def calculate_ler(prediction, target, dict):
    """
    Computes the Character Error Rate, defined as the edit distance.
    Arguments:
        prediction (string): space-separated sentence
        target (string): space-separated sentence
    """
    prediction = arpabet_cleaners(prediction, dict)
    if target.isupper() is False: # if korean
        target = h2j(target)
#        target = list(target)
    target = arpabet_cleaners_target(target, dict)

    prediction = ''.join([chr(idx) for idx in prediction])

    target = ''.join([chr(idx) for idx in target])
    return Lev.distance(prediction, target), len(target)


def arpabet_cleaners(text, dict):
#    pdb.set_trace()
#    text = text.upper()
    text = clean_str(text)
    text = collapse_whitespace(text)

          
    text = text.split()
    sequence = []
    # if korean
    
    sequence = [dict.get(t, dict["<unk>"]) for t in text]
    return sequence

def arpabet_cleaners_target(text, dict):
#    pdb.set_trace()
#    text = text.upper()
    text = clean_str(text)
    text = collapse_whitespace(text)

    if text.isupper() is False: # if korean
        text = list(text)
    else:
        text = text.split()
    sequence = []
    # if korean
    
    sequence = [dict.get(t, dict["<unk>"]) for t in text]
    return sequence

def clean_str(text):
    pattern = '[{}◎]'  # 특수기호제거
    text = re.sub(pattern=pattern, repl='', string=text)

    return text

if __name__ == "__main__":
    fpath1 = 'sentences/test.p.reduced.bpe' # target
    fpath2 = 'sentences/iwslt2016_E34L1.80-128146' # predict
    letter_err_cnt = 0
    letter_tot_cnt = 0
    line_count = 0
    token2idx, idx2token = load_vocab()
    with open(fpath1, 'rt', encoding='UTF-8') as f1, open(fpath2, 'rt', encoding='UTF-8') as f2: 
        for target, pred in zip(f1, f2): 
#            pdb.set_trace()
            target = target.replace('\n', '')
            pred = pred.replace('\n', '')
#            pred=re.sub('')
            letter_err_cnt_i, letter_tot_cnt_i = calculate_ler(pred, target, token2idx)
#    with open('test/3/iwslt2016_E20L1.49-36760', 'rt', encoding='utf-8') as rf:
#        for line_idx, pred in enumerate(rf.readlines()):
#            idx2token, token2idx = load_vocab()
#            pdb.set_trace()
#            line = line.replace('\n', '')
#            data = line.split('\t')
#            target = data[0]
            
#    with open(fpath1, 'rt', encoding='UTF-8') as f1, open(fpath2, 'rt', encoding='UTF-8') as f2:
#        for sent1, sent2 in zip(f1, f2):
#            if len(sent1.split()) + 1 > maxlen1: continue # 1: </s>
#            if len(sent2.split()) + 1 > maxlen2: continue  # 1: </s>
            
            
#            pred = data[1]

#            letter_err_cnt_i, letter_tot_cnt_i = calculate_ler(pred, target)
            line_count += 1
            letter_err_cnt += letter_err_cnt_i
            letter_tot_cnt += letter_tot_cnt_i
            print('idx: {}, per: {}%'.format(line_count, letter_err_cnt_i / letter_tot_cnt_i * 100))

    print('total per: {}%'.format(letter_err_cnt / letter_tot_cnt * 100))
