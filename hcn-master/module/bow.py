# import modules.util as util
import module.util as util
import numpy as np


'''
    Bag of Words
    
    1. Get a list of words from training set
    2. Get vocabulary, vocab_size
    3. Build bag of words for each utterance
        3.1 iterate through words
        3.2 pick id from vocab
        3.3 set index as 1 in ndarray
'''

class BoW_encoder():

    def __init__(self):
        #用户说的话的单词列表
        self.vocab = self.get_vocab()
        #列表长度
        self.vocab_size = len(self.vocab)

    def get_vocab(self):
        #获得用户说的话，然后将所有的话变成word，并进行排序，返回一个list:['<SILENCE>', 'a', 'actually', 'address', 'am', 'be', 'bombay', 'book', 'british', 'can', 'cheap', 'could', 'cuisine', 'do', 'does', "don't", 'eight', 'else',...
        content = util.read_content()
        #根据空格分开，然后进行排序，排序按照第一个字母的AcsII码进行排序，形成一个list：['<SILENCE>', 'a', 'actually', 'address', 'am', 'be', 'bombay', 'book', 'british', 'can', 'cheap', 'could', 'cuisine', 'do', 'does', "don't", 'eight', 'else',...
        vocab = sorted(set(content.split(' ')))
        # remove empty strings
        return [ item for item in vocab if item ]
#将传入的话进行单词编码，若含有语料库中的单词，那么对应单词的位置就是1，其他是0：
# i'd like to book a table with italian food
# [0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1
#  0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0
#  0 0 1 0 0 0 0 1 0 0 0]
    def encode(self, utterance):
        bow = np.zeros([self.vocab_size], dtype=np.int32)
        for word in utterance.split(' '):
            if word in self.vocab:
                idx = self.vocab.index(word)
                bow[idx] += 1
        return bow
