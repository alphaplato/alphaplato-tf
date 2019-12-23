# DeepFM

## 1、网络结构

![deepfm strcuture](https://github.com/alphaplato/alphaplato/blob/master/image/DeepLearning/deepfm.png)

## 2、实现要点

deepfm实现主要由FM和DNN两部分构成，但将FM拆开实际上变成由LR和FM（的embedding部分）和DNN构成。

需要注意：
* 连续向量的处理：原始论文提到“ Each categorical field is represented as a vector of one-hot encoding, and each continuous field is represented as the value itself, or a vector of one-hot encoding after discretization. ”
* 木桶效应：LR、FM和DNN实际上构成联合训练的类多模态网络，其中最差的网络结构对整体的负面影响最大，在实现时可以视情况将LR部分去掉（连续特征不利于LR）。

## 参考文献：
* [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247.pdf)