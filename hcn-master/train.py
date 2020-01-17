from module.entities import EntityTracker
from module.bow import BoW_encoder
from  module.lstm_net import LSTM_net
from module.embed import UtteranceEmbed
from module.actions import ActionTracker
from module.data_utils import Data
import module.util as util

import numpy as np
import sys


class Trainer():

    def __init__(self):

        et = EntityTracker()
        self.bow_enc = BoW_encoder()
        self.emb = UtteranceEmbed()
        at = ActionTracker(et)
        #获得用户说的话和对应的id，获得段落对话的行数：[('good morning', 3), ("i'd like to book a table with italian food", 7), ('<SILENCE>', 13), ('in paris', 6), ('for six people please', 14)...[{'start': 0, 'end': 21}, {'start': 21, 'end': 40}, {'start': 40, 'end': 59}, {'start': 59, 'end': 72},...
        self.dataset, dialog_indices = Data(et, at).trainset
        self.dialog_indices_tr = dialog_indices[:200]
        self.dialog_indices_dev = dialog_indices[200:250]
        #dim=300,vocab_size=len(self.vocab),num_features=4
        obs_size = self.emb.dim + self.bow_enc.vocab_size + et.num_features
        #获得对话模板
        self.action_templates = at.get_action_templates()
        #对话模板长度
        action_size = at.action_size

        nb_hidden = 128

        self.net = LSTM_net(obs_size=obs_size,
                       action_size=action_size,
                       nb_hidden=nb_hidden)


    def train(self):

        print('\n:: training started\n')
        epochs = 20
        for j in range(epochs):
            # iterate through dialogs
            num_tr_examples = len(self.dialog_indices_tr)
            loss = 0.
            #先是位置，后是其中的数值
            for i,dialog_idx in enumerate(self.dialog_indices_tr):
                # get start and end index
                start, end = dialog_idx['start'], dialog_idx['end']
                # train on dialogue
                loss += self.dialog_train(self.dataset[start:end])
                # print #iteration
                sys.stdout.write('\r{}.[{}/{}]'.format(j+1, i+1, num_tr_examples))

            print('\n\n:: {}.tr loss {}'.format(j+1, loss/num_tr_examples))
            # evaluate every epoch
            accuracy = self.evaluate()
            print(':: {}.dev accuracy {}\n'.format(j+1, accuracy))

            if accuracy > 0.99:
                self.net.save()
                break
        print('函数')

    def dialog_train(self, dialog):
        # create entity tracker
        et = EntityTracker()
        # create action tracker
        at = ActionTracker(et)
        # reset network
        self.net.reset_state()

        loss = 0.
        # iterate through dialog
        for (u,r) in dialog:
            #u是用户说的话，r是位置
            u_ent = et.extract_entities(u)
            #u_ent_features：对et进行编码，返回4维矩阵
            u_ent_features = et.context_features()
            #u_emb：对用户说的话进行编码，得到一个300维度的矩阵
            u_emb = self.emb.encode(u)
            #返回矩阵，每句话对应单词的位置为1，其他为0
            u_bow = self.bow_enc.encode(u)
            # concat features
            #features:三个矩阵首尾相接，形成一个大的一维矩阵
            features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)
            # get action mask 16维度，根据提供的槽位独特设置的
            action_mask = at.action_mask()
            # forward propagation
            #  train step
            loss += self.net.train_step(features, r, action_mask)
        return loss/len(dialog)

    def evaluate(self):
        # create entity tracker
        et = EntityTracker()
        # create action tracker
        at = ActionTracker(et)
        # reset network
        self.net.reset_state()

        dialog_accuracy = 0.
        for dialog_idx in self.dialog_indices_dev:

            start, end = dialog_idx['start'], dialog_idx['end']
            dialog = self.dataset[start:end]
            num_dev_examples = len(self.dialog_indices_dev)

            # create entity tracker
            et = EntityTracker()
            # create action tracker
            at = ActionTracker(et)
            # reset network
            self.net.reset_state()

            # iterate through dialog
            correct_examples = 0
            for (u,r) in dialog:
                # encode utterance

                u_ent = et.extract_entities(u)
                u_ent_features = et.context_features()
                u_emb = self.emb.encode(u)
                u_bow = self.bow_enc.encode(u)
                # concat features
                features = np.concatenate((u_ent_features, u_emb, u_bow), axis=0)
                # get action mask
                action_mask = at.action_mask()
                # forward propagation
                #  train step
                prediction = self.net.forward(features, action_mask)
                print('dddd')
                print(prediction)
                correct_examples += int(prediction == r)
            # get dialog accuracy
            dialog_accuracy += correct_examples/len(dialog)

        return dialog_accuracy/num_dev_examples



if __name__ == '__main__':
    # setup trainer
    trainer = Trainer()
    # start training
    trainer.train()
