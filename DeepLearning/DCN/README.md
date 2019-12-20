# DCN

## 1、网络结构

![deepfm strcuture](https://github.com/alphaplato/alphaplato/blob/master/image/DeepLearning/dcn.png)

## 2、实现要点

* 注意DCN网络公式相乘采用结合律，可规避形成二维张量空间，节省运行内存资源，提高计算效率。。

* 由于多次预第一层抽取特征相乘生成高级特征，DCN网络的特质之一是对非归一化（不同量纲）数据特别敏感，那么归一化数据就极为重要，否则网络极不稳定；还有一个策略就是对数值较大的特征进行压缩，本demo在csv转为record数据时进行了log10压缩，其中这个操作可以配置在json文件指定特征实现。
