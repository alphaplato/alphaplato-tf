# DeepMatch

## 1、网络结构

![deepmatch structure](https://github.com/alphaplato/alphaplato/blob/master/image/DeepLearning/deepmatch.png)

## 2、实现要点

* 通过网络学东西用户的embedding信息。

* 将item的信息embedding，并将item进行极限分类。

* 通过随机负采样或层次softmax简化网络及运算，本demo采用随机负采样。

## 参考文献：
* [Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)
* [数据集](https://grouplens.org/datasets/movielens/25m/)