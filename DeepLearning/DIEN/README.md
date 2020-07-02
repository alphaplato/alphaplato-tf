# DIEN

## 1、网络结构

![dien structure](https://github.com/alphaplato/alphaplato/blob/master/image/DeepLearning/dien.png)

## 2、实现要点

* 网络有两部分组成，deepfm的深度部分输入和atention的输出层concat在一起输入深度网络。

* 通过attention进一步利用序列化的特征信息。

* 通过GRU网络提高序列化特征的时间利用率。

* 辅助网络，通过辅助网络提高第一次gru网络的信息提取质量，实现较为复杂，需要重点关注。

* 涉及到改写tensorflow较为基层的api，有一定难度。

* 计算复杂度高，train起来比较耗时。

* 注意padding（补足）值不不取点。

## 参考文献：
* [Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672.pdf)
* [数据集](https://github.com/mouna99/dien/blob/master/data.tar.gz)