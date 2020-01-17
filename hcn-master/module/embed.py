from gensim.models import word2vec
import numpy as np


class UtteranceEmbed():
    #得到word2vec的模型，由text8得到的模型

    def __init__(self, fname='data/text8.model', dim=300):
        self.dim = dim
        try:
            # load saved model
            self.model = word2vec.Word2Vec.load(fname)
        except:
            print(':: creating new word2vec model')
            self.create_model()
            self.model = word2vec.Word2Vec.load(fname)
#word的编码，先把每个单词求编码，然后对编码进行平均，返回一个300维的数据；若没有话则返回一个300维的0填充的数组
    def encode(self, utterance):
        embs = [ self.model[word] for word in utterance.split(' ') if word and word in self.model]
        # average of embeddings
        # print(embs)
        if len(embs):
            return np.mean(embs, axis=0)
        else:
            return np.zeros([self.dim],np.float32)

    def create_model(self, fname='text8'):
        sentences = word2vec.Text8Corpus('data/text8')
        model = word2vec.Word2Vec(sentences, size=self.dim)
        model['first']
        model.save('data/text8.model')
        print(':: model saved to data/text8.model')
