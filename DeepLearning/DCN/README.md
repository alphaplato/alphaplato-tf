# DeepFM

## 1、网络结构

![deepfm strcuture]()

## 2、算法实现

* 注意公式相乘更简便的操作。

* DCN的有一多层相乘，造成的结果是对非归一化（不同量纲）数据特别敏感，本demo在实现时对每一层的DCN输出是增加了tf.layers.batch_normalization的处理，这个可能会是模型收敛的关键。
