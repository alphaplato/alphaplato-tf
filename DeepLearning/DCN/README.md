# DeepFM

## 1、网络结构

![deepfm strcuture](https://github.com/alphaplato/alphaplato/blob/master/image/DeepLearning/dcn.png)

## 2、实现要点

* 注意DCN网络公式相乘采用结合律，可规避形成二维张量空间，节省运行内存资源，提高计算效率。。

* 由于多次预第一层抽取特征相乘，DCN网络的特质之一是对非归一化（不同量纲）数据特别敏感，那么非归一化数据下合理使用tf.layers.batch_normalization的trick就变得重要；本demo在实现时对每一层的DCN输出是增加了tf.layers.batch_normalization的处理，这个可能会是模型收敛的关键。
