# DIN

## 1、网络结构

![din structure](https://github.com/alphaplato/alphaplato/blob/master/image/DeepLearning/din.png)

## 2、实现要点

* 网络有两部分组成，deepfm的深度部分输入和atention的输出层concat在一起输入深度网络。

* 核心思想就是通过attention进一步利用序列化的特征信息。

* 理论看起来很复杂，但实现起来其实并不复杂；核心思想很明确。

* 计算复杂度高，因此train起来比较耗时。

## 参考文献：
* [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978.pdf)
* [数据集](https://github.com/mouna99/dien/blob/master/data.tar.gz)