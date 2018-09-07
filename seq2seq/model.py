# -*- coding: utf-8 -*-
# @Time    : 2018/9/6 8:40
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : model.py
# @Software: PyCharm

from distutils.version import LooseVersion
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from data_helper import letters_data

assert LooseVersion(tf.__version__) >= LooseVersion("1.1")
print("Tensorflow Version:{}".format(tf.__version__))

class seq2seq(object):

    def encoder(self,input_data,rnn_size,num_layers,source_sequence_length,source_vocab_size,
                encoding_embedding_size):
        '''

        :param input_data: 输入tensor
        :param rnn_size: rnn隐层节点数量
        :param num_layers: 堆叠的rnn cell数量
        :param source_sequence_length: 源数据的序列长度
        :param source_vocab_size: 源数据的词典大小
        :param encoding_embedding_size: embedding的大小
        :return:
        '''

        encoder_embed_input = tf.contrib.layers.embed_sequence(input_data,source_vocab_size,encoding_embedding_size)
        lstm_cells = []
        for _ in range(num_layers):
            lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size,initializer=tf.random_uniform_initializer(-0.1,0.1,seed=2))
            lstm_cells.append(lstm_cell)
        cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)
        encoder_output,encoder_state = tf.nn.dynamic_rnn(cell,encoder_embed_input,
                                                         sequence_length=source_sequence_length,dtype=tf.float32)
        return encoder_output,encoder_state

    def decoder(self,target_letter_to_int,decoding_embedding_size,num_layers,batch_size,rnn_size,
                target_sequence_length,max_target_sequence_length,encoder_state,decoder_input):
        '''

        :param target_letter_to_int: target数据的映射表
        :param decoding_embedding_size: embed向量大小
        :param num_layers: 堆叠的RNN单元数量
        :param run_size: RNN单元的隐层结点数量
        :param target_sequence_length: target数据序列长度
        :param max_target_sequence_length: target数据序列最大长度
        :param encoder_state: encoder端编码的状态向量
        :param decoder_input: decoder端输入
        :return:
        '''

        # 1. Embedding
        target_vocab_size = len(target_letter_to_int)
        decode_embeddings = tf.Variable(tf.random_uniform([target_vocab_size,decoding_embedding_size]))
        decoder_embed_input = tf.nn.embedding_lookup(decode_embeddings,decoder_input)

        # 2. 构造Decoder中的RNN单元
        rnn_cells = []
        for _ in range(num_layers):
            decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size,initializer = tf.random_uniform_initializer(-0.1,0.1,seed=2))
            rnn_cells.append(decoder_cell)
        cell = tf.contrib.rnn.MultiRNNCell(rnn_cells)

        # 3.output全连接层
        # 对Decoder层的输出做映射（projection），Dense会把输出变成字典大小，这样才能计算预测出来哪个单词的概率最大
        output_layer = Dense(target_vocab_size,kernel_initializer=tf.truncated_normal_initializer(mean = 0.0,stddev=0.1))

        # 4.Training decoder
        with tf.variable_scope('decode'):
            # A helper for use during training. Only reads inputs.
            # Returned sample_ids are the argmax of the RNN output logits.
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                                sequence_length=target_sequence_length,
                                                                time_major = False)
            # training_helper其实就是decoder的target输入
            training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                               training_helper,
                                                               encoder_state,
                                                               output_layer)
            # return:(final_outputs, final_state, final_sequence_lengths)
            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                              impute_finished = True,
                                                                              maximum_iterations = max_target_sequence_length)
        # 5.Predicting decoder
        # 与training共享参数
        with tf.variable_scope('decode',reuse = True):
            # 创建一个常量tensor并复制为batch_size的大小
            # tf.tile(input, multiples,name=None),其中输出将会重复input输入multiples次,multiples和input的维度一样
            start_tokens = tf.tile(tf.constant([target_letter_to_int['<GO>']],dtype= tf.int32),
                                                [batch_size],name = 'start_tokens')
            # GreedyEmbeddingHelper:__init__(embedding,start_tokens,end_token)
            predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decode_embeddings,
                                                                         start_tokens,
                                                                         target_letter_to_int['<EOS>'])
            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                                 predicting_helper,
                                                                 encoder_state,
                                                                 output_layer)
            # return:(final_outputs, final_state, final_sequence_lengths)
            predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                                impute_finished = True,
                                                                                maximum_iterations = max_target_sequence_length)
        return training_decoder_output,predicting_decoder_output

    # 其实就是将encoder的输出作为decoder的输入
    def seq2seq_model(self,input_data,targets,target_sequence_length,
                      max_target_sequence_length,source_sequence_length,
                      source_vocab_size,target_vocab_size,target_letter_to_int,
                      encoder_embedding_size,decoder_embedding_size,
                      rnn_size,num_layers,batch_size):

        # 获取encoder的状态输出
        _, encoder_state = self.encoder(input_data,
                                        rnn_size,
                                        num_layers,
                                        source_sequence_length,
                                        source_vocab_size,
                                        encoder_embedding_size)
        # 预处理后的decoder输入
        decoder_input = letters_data().process_decoder_input(targets,target_letter_to_int,batch_size)

        # 将状态向量与输入传递给decoder，主要获取上面生成的encoder_state,decoder_input
        training_decoder_output,predicting_output = self.decoder(target_letter_to_int,
                                                                 decoder_embedding_size,
                                                                 num_layers,
                                                                 batch_size,
                                                                 rnn_size,
                                                                 target_sequence_length,
                                                                 max_target_sequence_length,
                                                                 encoder_state,
                                                                 decoder_input)
        return training_decoder_output,predicting_output
