# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Note.
if safe, entities on the source side have the prefix 1, and the target side 2, for convenience.
For example, fpath1, fpath2 means source file path and target file path, respectively.
'''
import tensorflow as tf
from utils import calc_num_batches
import pdb
import re
from text.korean import ALL_SYMBOLS, jamo_to_korean
from jamo import hangul_to_jamo, h2j, j2hcj
import numpy as np


# from g2p_en import G2p
def load_vocab(vocab_fpath):
    '''Loads vocabulary file and returns idx<->token maps
    vocab_fpath: string. vocabulary file path.
    Note that these are reserved
    0: <pad>, 1: <unk>, 2: <s>, 3: </s>

    Returns
    two dictionaries.
    '''
    #    vocab = [line.split()[0] for line in open(vocab_fpath, 'r').read().splitlines()]
    #    pdb.set_trace()

    #            self.graphemes = ["<pad>", "<unk>", "</s>"] + list("abcdefghijklmnopqrstuvwxyz")
    #        self.phonemes = ["<pad>", "<unk>", "<s>", "</s>"] + ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
    #                                                             'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH',
    #                                                             'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
    #                                                             'EY2', 'F', 'G', 'HH',
    #                                                             'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L',
    #                                                             'M', 'N', 'NG', 'OW0', 'OW1',
    #                                                             'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH',
    #                                                             'UH0', 'UH1', 'UH2', 'UW',
    #                                                             'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
    #        self.g2idx = {g: idx for idx, g in enumerate(self.graphemes)}
    #        self.idx2g = {idx: g for idx, g in enumerate(self.graphemes)}

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


def load_data(fpath1, fpath2, maxlen1, maxlen2):
    '''Loads source and target data and filters out too lengthy samples.
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.

    Returns
    sents1: list of source sents
    sents2: list of target sents
    '''
    sents1, sents2 = [], []
    #    with open(fpath1, 'r') as f1, open(fpath2, 'r') as f2:

    with open(fpath1, 'rt', encoding='UTF-8') as f1, open(fpath2, 'rt', encoding='UTF-8') as f2:
        for sent1, sent2 in zip(f1, f2):
            if len(sent1.split()) + 1 > maxlen1: continue  # 1: </s>
            if len(sent2.split()) + 1 > maxlen2: continue  # 1: </s>
            #            pdb.set_trace()
            sent1 = re.sub("\n", "", sent1)
            sent2 = re.sub("\n", "", sent2)

            #            YJS added on 2019-11-18 -> dealing with curly brackets and space between words
            sent11 = h2j(sent1)
            sent22 = h2j(sent2)

            if sent11 == sent1:  # if english
                sent1 = re.sub(' ', '◎', sent1)
                sent2 = re.sub("} {", " ◎ ", sent2)
                sent2 = re.sub("{", "", sent2)
                sent2 = re.sub("}", "", sent2)
            else:  # if korean
                sent1 = re.sub(" ", "◎", sent11)
                sent2 = re.sub(" ", "◎", sent22)

            #            pattern2 = '[}{!?."]'
            #            sent1 = re.sub(pattern=pattern2, repl='', string=sent1)
            #            sent2 = re.sub(pattern=pattern2, repl='', string=sent2)

            sents1.append(sent1.strip())
            sents2.append(sent2.strip())
    return sents1, sents2


def clean_str(text):
    #     cleaned_text = re.sub('[a-zA-Z]', '',text)
    #     cleaned_text = re.sub('[0-9]', '',cleaned_text)
    #     cleaned_text = re.sub('[→…★\\{\\}\\[\\]新\\/?.,;:⑦|\\♥)*~`·!^\\-_+<>@\\#$%&\\\\\\=\\(\\[]', '', cleaned_text)
    pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'  # E-mail제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'  # URL제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'  # 한글 자음, 모음 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '<[^>]*>'  # HTML 태그 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '[^\w\s]'  # 특수기호제거
    text = re.sub(pattern=pattern, repl='', string=text)
    return text


def encode(inp, type, dict):
    # def encode(inp, type, dict):
    '''Converts string to number. Used for `generator_fn`.
    inp: 1d byte array.
    type: "x" (source side) or "y" (target side)
    dict: token2idx dictionary

    Returns
    list of numbers
    '''
    #pdb.set_trace()
    #    inp_str=inp

    # inp_str = re.sub("b'", "", inp_str)
    # inp_str = re.sub("'", "", inp_str)
    #    inp_str = inp

    #inp_str = inp
    inp_str = inp.decode("UTF-8")
    #
    # YJS changed: according to paper, <s> and </s> in both source and target
    #    tokens = ["<s>"] + inp_str.split() + ["</s>"]
    # =============================================================================
    if type == "x":
        tokens = list(inp_str) + ["</s>"]
    else:
        if inp_str.isupper() is True:
            tokens = ["<s>"] + inp_str.split() + ["</s>"] # If English PHONEME
        else:
            tokens = ["<s>"] + list(inp_str) + ["</s>"] # If Korean PHONEME


        # YJS comment: Eng: inp_str.split; Kor: list(inp_str)

    #    if type == "x":
#        tokens = list(inp_str) + ["</s>"]
#    else:
#        tokens = ["<s>"] + inp_str.split() + ["</s>"]
#        if np.shape(tokens) == (3,):  # If inp_str.split is no effect -> equals Korean!!!
        #tokens = ["<s>"] + list(inp_str) + ["</s>"]

    #        tokens = ["<s>"] + list(inp_str) + ["</s>"]
    # tokens = ["<s>"] + list(inp_str) + ["</s>"]

    #     if type=="x": tokens = inp_str.split() + ["</s>"]
    #     else: tokens = ["<s>"] + inp_str.split() + ["</s>"]
    #
    # =============================================================================
    x = [dict.get(t, dict["<unk>"]) for t in tokens]
    return x


def generator_fn(sents1, sents2, vocab_fpath):
    '''Generates training / evaluation data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.

    yields
    xs: tuple of
        x: list of source token ids in a sent
        x_seqlen: int. sequence length of x
        sent1: str. raw source (=input) sentence
    labels: tuple of
        decoder_input: decoder_input: list of encoded decoder inputs
        y: list of target token ids in a sent
        y_seqlen: int. sequence length of y
        sent2: str. target sentence
    '''
    #    pdb.set_trace()
    token2idx, _ = load_vocab(vocab_fpath)
    for sent1, sent2 in zip(sents1, sents2):
        #        pdb.set_trace()
        #        if language=='english':
        x = encode(sent1, "x", token2idx)
        y = encode(sent2, "y", token2idx)

        decoder_input, y = y[:-1], y[1:]

        x_seqlen, y_seqlen = len(x), len(y)
        yield (x, x_seqlen, sent1), (decoder_input, y, y_seqlen, sent2)


def input_fn(sents1, sents2, vocab_fpath, batch_size, shuffle=False):
    '''Batchify data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean

    Returns
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    '''
    shapes = (([None], (), ()),
              ([None], [None], (), ()))
    types = ((tf.int32, tf.int32, tf.string),
             (tf.int32, tf.int32, tf.int32, tf.string))
    paddings = ((0, 0, ''),
                (0, 0, 0, ''))

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(sents1, sents2, vocab_fpath))  # <- arguments for generator_fn. converted to np string arrays

    if shuffle:  # for training
        dataset = dataset.shuffle(128 * batch_size)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    return dataset


def get_batch(fpath1, fpath2, maxlen1, maxlen2, vocab_fpath, batch_size, shuffle=False):
    '''Gets training / evaluation mini-batches
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean

    Returns
    batches
    num_batches: number of mini-batches
    num_samples
    '''
    # YJS changed for ENG / KOREAN
    #    if language=='english':
    sents1, sents2 = load_data(fpath1, fpath2, maxlen1, maxlen2)
    #    else:
    #        sents1, sents2 = load_data_kor(fpath1, fpath2, maxlen1, maxlen2)
    #
    # pdb.set_trace()
    batches = input_fn(sents1, sents2, vocab_fpath, batch_size, shuffle=shuffle)
    num_batches = calc_num_batches(len(sents1), batch_size)
    return batches, num_batches, len(sents1)
