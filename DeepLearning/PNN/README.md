# PNN

## 1、网络结构

![deepfm strcuture](https://github.com/alphaplato/alphaplato/blob/master/image/DeepLearning/pnn.png)

## 2、实现要点

* PNN网络有两部分输入concat后进入DNN网络，这两部分输入的一部分是所有特征embedding，另一部分即pnn设计的输入结构，类FM的二阶处理即向量两两相乘（形成内积或外积）后作为输入，本demo使用内积方式。
* 相当于DeepFM网络去掉FM的部分，在第一层输入时增加FM的处理方式的信号后进入深度网络。

## 参考文献：
* [Product-based Neural Networks for User Response
Prediction](https://arxiv.org/pdf/1611.00144.pdf)