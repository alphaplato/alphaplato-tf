# DeepFM

## 1、网络结构

![deepfm strcuture](https://github.com/alphaplato/alphaplato/blob/master/image/DeepLearning/dcn.png)

## 2、实现要点

* 注意DCN网络公式相乘采用结合律，可规避行程二维张量空间，节省运行内存资源，提高计算效率。。

* DCN的有一多层相乘，所以DCN网络对非归一化（不同量纲）数据特别敏感；本demo在实现时对每一层的DCN输出是增加了tf.layers.batch_normalization的处理，这个可能会是模型收敛的关键。
