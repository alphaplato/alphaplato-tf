# DeepFM

## 1、网络结构

## 2、算法实现

deepfm实现主要由FM和DNN两部分构成，但将FM拆开实际上变成由LR和FM（的embedding部分）和DNN构成：

需要注意：
* 连续向量的处理：原始论文提到“ Each categorical field is represented as a vector of one-hot encoding, and each continuous field is represented as the value itself, or a vector of one-hot encoding after discretization. ”
* 木桶
