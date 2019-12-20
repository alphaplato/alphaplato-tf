# DeepFM

## 1、网络结构

![xdeepfm structure](https://github.com/alphaplato/alphaplato/blob/master/image/DeepLearning/xdeepfm.png)

## 2、实现要点

* 网络有三部分组成，lr、deepfm的深度部分最外层、和xdeep的多次交叉层。

* 多次交叉层的思想史是FM二阶交叉的升级，二阶交叉为隐向量两两外积之后，元素相加；那么升级的思想是原始隐向量两两外积后形成二阶隐向量，二阶隐向量再次和原始隐向量外积相乘变为三阶；以此类推至设定阶次。

* 核心思想就是通过进一步利用特征信息交叉，而且是以特征域为单位进行。

ps:xdeepfm效果上是最好的，个人认为其理论也是所有这些FM升级到深度模型变种中最好的。但是，高阶次的交叉增加了模型的时间复杂度，训练时间成本高，且模型上线时非常挑战serving服务的响应性能。