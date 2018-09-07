# Seq2Seq_Char
Basic Seq2seq
## 简介
主要是借鉴大神“天雨粟”的jupyter代码来改下成data_helper.py,model.py,train.py三个py文件，写的比较粗糙。<br>
源码链接：https://github.com/NELSONZHAO/zhihu/blob/master/basic_seq2seq/Seq2seq_char.ipynb <br>
## 疑惑之处：
有几点比较疑惑的地方：<br>
1.if batch_i % display_step == 0:<br>
这句代码当batch_i == 0时也成立，这个时候也计算了一次loss，所以train后的结果与原博主有很大的不同。结果可在code中的photo文件夹看到。<br>
2.predict的结果也很差，结果可在code中的photo文件夹看到。<br>
