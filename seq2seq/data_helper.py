# -*- coding: utf-8 -*-
# @Time    : 2018/9/6 8:44
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : data_helper.py
# @Software: PyCharm

import numpy as np
import time
import tensorflow as tf

class letters_data(object):

    def extract_character_vocab(self, data):
        '''

        :return:

        构造映射表
        '''
        special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
        set_words = list(set([character for line in data.split('\n') for character in line]))
        int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
        vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
        return int_to_vocab, vocab_to_int

    def proprcess(self,source_path,target_path):
        with open(source_path,'r',encoding='utf-8') as f:
            source_data = f.read()
        with open(target_path,'r',encoding='utf-8') as f:
            target_data = f.read()
        # print(source_data.split('\n')[:10])
        # print(target_data.split('\n')[:10])

        source_int_to_letter,source_letter_to_int = self.extract_character_vocab(source_data)
        target_int_to_letter,target_letter_to_int = self.extract_character_vocab(target_data)

        # 对于新来的数据，如果能在词典中找到，则返回对应的索引，否则返回’UNK‘的索引
        source_int = [[source_letter_to_int.get(letter,source_letter_to_int['<UNK>'])for letter in line] for line in source_data.split('\n')]
        target_int = [[target_letter_to_int.get(letter,target_letter_to_int['<UNK>'])for letter in line] + [target_letter_to_int['<EOS>']]for line in target_data.split('\n')]
        return source_int,target_int,source_letter_to_int,target_letter_to_int,source_int_to_letter,target_int_to_letter

    # 模型的输入
    def get_inputs(self):
        '''
        inputs:[batch_sizes,seq_length]
        targets:[batch_sizes,seq_length]
        :return:
        模型输入tensor
        '''
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
        target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
        max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
        source_sequence_length = tf.placeholder(tf.int32,(None,), name='source_sequence_length')

        return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length,source_sequence_length

    def process_decoder_input(self,data,vocab_to_int,batch_size):
        '''
        tf.fill([batch_size,1], vocab_to_int['<GO>']),1表示seq_length=1，即一个字符，并且生成batch个一维的<GO>字符
        :param data:
        :param vocab_to_int:
        :param batch_size:
        :return:
        补充<GO>，并移除最后一个字符
        '''
        ending = tf.strided_slice(data,[0,0],[batch_size,-1],[1,1])
        # tf.fill(dims,input)
        decoder_input = tf.concat([tf.fill([batch_size,1],vocab_to_int['<GO>']),ending],axis=1)
        return decoder_input

    def pad_sentence_batch(self,sentence_batch,pad_int):
        '''
        对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length
        :param sentence_batch:
        :param pad_int: <PAD>对应索引号
        :return:
        '''

        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return [sentence + [pad_int] *(max_sentence - len(sentence))  for sentence in sentence_batch]

    def get_batches(self,targets,sources,batch_size,source_pad_int,target_pad_int):
        '''
        定义生成器，用来获取batch
        :param targets:
        :param sources:
        :param batch_size:
        :param source_pad_int:
        :param target_pad_int:
        :return:
        '''
        for batch_i in range(0,len(sources)//batch_size):
            start_i = batch_i * batch_size
            sources_batch = sources[start_i:start_i + batch_size]
            targets_batch = targets[start_i:start_i + batch_size]
            pad_sources_batch = np.array(self.pad_sentence_batch(sources_batch,source_pad_int))
            pad_target_batch = np.array(self.pad_sentence_batch(targets_batch,target_pad_int))

            #记录每条记录的长度
            targets_lengths = []
            for target in targets_batch:
                targets_lengths.append(len(target))
            source_lengths = []
            for source in sources_batch:
                source_lengths.append(len(source))
            yield pad_target_batch,pad_sources_batch,targets_lengths,source_lengths


if __name__ == '__main__':
    source_path = 'data/letters_source.txt'
    target_path = 'data/letters_target.txt'
    letters_data().proprcess(source_path,target_path)