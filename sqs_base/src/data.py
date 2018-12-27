#-*- coding:utf-8 -*-
############################
#File Name: data.py
#Author: chi xiao
#Mail:
#Created Time:
############################
import numpy as np
import sys
import glob
import struct
from tensorflow.core.example import example_pb2

#special token

PARAGTAPH_START = '<p>'
PARAGTAPH_END = '</p>'
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
UNKNOWN_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
DUCUMENT_START = '<d>'
DUCUMENT_END = '</d>'


class vocab(object):

    def __init__(self,vocab_file,max_size):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0

        with open(vocab_file,'r') as f:
            for line in f:
                pair = line.split()
                if(len(pair)!=2):
                    sys.stderr.write("Bad line: %s\n" % line)
                    continue
                if(pair[0] in self._word_to_id):
                    raise ValueError("Duplicated words: %s" % pair[0])
                self._word_to_id[pair[0]] = self._count
                self._id_to_word[self._count] = pair[0]
                self._count += 1

                if self._count > max_size:
                    raise ValueError("Too many words: > %" % max_size)

    def CheckWord(self,word):
        if word not in self._word_to_id:
            return None
        return self._word_to_id[word]

    def WordToId(self,word):
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def IdToWord(self,word_id):
        if word_id not in self._id_to_word:
            raise ValueError("id not found in vocab: %d." % word_id)
        return self._id_to_word[word_id]

    def NumIds(self):
        return self._count


def Pad(ids, pad_id, length):
    assert pad_id is not None
    assert length is not None

    if len(ids) < length:
        a = [pad_id] * (length - len(ids))
        return  ids + a
    else:
        return ids[:length]

def GetWordIds(text, vocab, pad_len=None, pad_id=None):
    ids = []
    for w in text.split():
        i = vocab.WordToId(w)
        if i >= 0:
            ids.append(i)
        else:
            ids.append(vocab.WordToId(UNKNOWN_TOKEN))
    if pad_len is not None:
        return Pad(ids,pad_id,pad_len)
    return ids

def IdsToWords(ids,vocab):
    assert isinstance(ids, list)
    return [vocab.IdToWord(i) for i in ids]


def SnippetGen(text, start_tok, end_tok, inclusive=True):
    cur = 0
    while True:
        try:
            start_p = text.index(start_tok,cur)
            end_p = text.index(end_tok,start_p+1)
            cur = end_p + len(end_tok)
            if inclusive:
                yield text[start_p:cur]
            else:
                yield text[start_p+len(start_tok):end_p]
        except ValueError as e:
            raise StopIteration("no more snippets in the text: %s" % e)

def ToSentence(paragraph,include_token=True):
    s_gen = (paragraph,SENTENCE_START,SENTENCE_END,include_token)
    return [s for s in s_gen]




