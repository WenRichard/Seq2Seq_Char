# -*- coding: utf-8 -*-
# @Time    : 2018/9/6 15:20
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : train.py
# @Software: PyCharm

import tensorflow as tf
from model import seq2seq
from data_helper import letters_data
import os

class seq2seq_train(object):
    def __init__(self):
        # 超参数
        self.epochs = 60
        self.batch_size = 128
        self.rnn_size = 50
        self.num_layers = 2
        self.encoding_embedding_size = 15
        self.decoding_embedding_size = 15
        self.learning_rate = 0.001

        self.source_path = 'data/letters_source.txt'
        self.target_path = 'data/letters_target.txt'
        self.save_dir = 'E:/NLP/seq2seq'
        self.checkpoint_path = os.path.join(self.save_dir, 'trained_model.ckpt')
        self.source_int,self.target_int,self.source_letter_to_int,self.target_letter_to_int,self.source_int_to_letter,\
        self.target_int_to_letter = letters_data().proprcess(self.source_path,self.target_path)

        # 每隔50轮输出loss
        self.display_step = 50
        # 将数据集分割为train和validation
        self.train_source = self.source_int[self.batch_size:]
        self.train_target = self.target_int[self.batch_size:]
        # 留出一个batch进行验证
        self.valid_source = self.source_int[:self.batch_size]
        self.valid_target = self.target_int[:self.batch_size]

        (self.valid_targets_batch,
         self.valid_sources_batch,
         self.valid_targets_lengths,
         self.valid_sources_lengths) = \
            next(letters_data().get_batches(self.valid_target,
                                            self.valid_source,
                                            self.batch_size,
                                            self.source_letter_to_int['<PAD>'],
                                            self.target_letter_to_int['<PAD>']))


    def train(self):
        train_graph = tf.Graph()
        with train_graph.as_default():
            input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = \
                letters_data().get_inputs()

            training_decoder_output, predicting_decoder_output = seq2seq().seq2seq_model(input_data,
                                                                                         targets,
                                                                                         target_sequence_length,
                                                                                         max_target_sequence_length,
                                                                                         source_sequence_length,
                                                                                         len(self.source_letter_to_int),
                                                                                         len(self.target_letter_to_int),
                                                                                         self.target_letter_to_int,
                                                                                         self.encoding_embedding_size,
                                                                                         self.decoding_embedding_size,
                                                                                         self.rnn_size,
                                                                                         self.num_layers,
                                                                                         self.batch_size)

            training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')

            print(training_logits)
            predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')

            # 用sequence_mask计算了每个句子的权重，该权重作为参数传入loss函数，主要用来忽略句子中pad部分的loss。如果是对pad以后的句子进行loop，
            # 那么输出权重都是1，不符合我们的要求
            masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

            with tf.name_scope('optimization'):
                # A float Tensor of rank 0, 1, or 2 depending on the average_across_timesteps and average_across_batch arguments
                # training_logits：A Tensor of shape [batch_size, sequence_length, num_decoder_symbols]
                # targets：A Tensor of shape [batch_size, sequence_length]
                # masks：A Tensor of shape [batch_size, sequence_length]
                cost = tf.contrib.seq2seq.sequence_loss(
                    training_logits,
                    targets,
                    masks)

                optimizer = tf.train.AdamOptimizer(lr)

                gradients = optimizer.compute_gradients(cost)
                capped_gradients = [(tf.clip_by_value(grad, -5, -5), var) for grad, var in gradients if grad is not None]
                train_op = optimizer.apply_gradients(capped_gradients)



        with tf.Session(graph=train_graph) as sess:
            sess.run(tf.global_variables_initializer())

            for epoch_i in range(1,self.epochs+1):
                # 得到batch的索引和用pad补全好的句子，以及每个batch内每个句子原始长度的列表
                for batch_i, (targets_batch, sources_batch, targets_lengths, source_lengths) in enumerate(
                    letters_data().get_batches(self.train_target,
                                               self.train_source,
                                               self.batch_size,
                                               self.source_letter_to_int['<PAD>'],
                                               self.target_letter_to_int['<PAD>'])):

                    # {}中为传入graph中的参数，max_target_sequence_length可由target_sequence_length得到
                    _, loss = sess.run([train_op, cost],
                                       {input_data: sources_batch,
                                       targets: targets_batch,
                                       lr: self.learning_rate,
                                       target_sequence_length: targets_lengths,
                                       source_sequence_length: source_lengths})

                    if (batch_i) % self.display_step == 0:
                        #计算validation loss
                        validation_loss = sess.run([cost],
                                                   {input_data: self.valid_sources_batch,
                                                    targets: self.valid_targets_batch,
                                                    lr: self.learning_rate,
                                                    target_sequence_length: self.valid_targets_lengths,
                                                    source_sequence_length: self.valid_sources_lengths})

                        print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f} -Validation loss: {:>6.3f}'
                              .format(epoch_i,
                                      self.epochs,
                                      batch_i,
                                      len(self.train_source) // self.batch_size,
                                      loss,
                                      validation_loss[0]))

            saver = tf.train.Saver()
            saver.save(sess, self.checkpoint_path)
            print('Model Trained and Saved')



    def predict(self,input_word):
        sequence_length = 7
        source_to_seq = lambda x:[self.source_letter_to_int.get(word,self.source_letter_to_int['<UNK>']) for word in x] +\
        [self.source_letter_to_int['<PAD>']]*(sequence_length-len(x))
        text = source_to_seq(input_word)
        loaded_graph = tf.Graph()
        with tf.Session(graph=loaded_graph) as sess:
            loader = tf.train.import_meta_graph(self.checkpoint_path + '.meta')
            loader.restore(sess,self.checkpoint_path)

            input_data = loaded_graph.get_tensor_by_name('inputs:0')
            logits = loaded_graph.get_tensor_by_name('predictions: 0')
            source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
            target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')

            answer_logits = sess.run(logits,{input_data:[text]*self.batch_size,
                                             target_sequence_length:[len(input_word)]*self.batch_size,
                                             source_sequence_length:[len(input_word)]*self.batch_size})[0]
        pad = self.source_letter_to_int['<PAD>']

        print('Raw Input : {}'.format(input_word))
        print('\nSource')
        print(' Word Number:        {}'.format([i for i in text]))
        print(' Input Words: {}'.format(''.join([self.source_int_to_letter[i] for i in text])))

        print('\nTarget')
        print(' Word Number:        {}'.format([i for i in answer_logits if i != pad]))
        print(' Output Words: {}'.format(''.join([self.target_int_to_letter[i] for i in answer_logits if i != pad])))



if __name__ == '__main__':
    seq2seq_train().train()
    seq2seq_train().predict('common')


