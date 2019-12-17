# alphaplato

## 1、DeepLearning
   这部分在推荐算法中的总结，面向的是大规模分布式算法，在完成网络结构时采用了tensorflow的feature_column函数分布式特征处理的处理方式。
   在使用深度网络其中需要注意的一点是数据的归一化，因为不同数值特征通常不在一个量级，比如CTR特征和counter计数特征，直接结果是网络不稳定甚至不收敛；所以在网络层级上适当增加batch_normalization的trick非常重要，改函数可以将网络数据高斯归一化。
   
### [1] DeepFM
   注意LR、FM、DEEP三部分网络的组合，在本demo中舍弃了LR的网络。原则是，联合网络当出现特别差的网络时会影响其他网络的训练，即出现木桶效应，本demo实际运行时只保留FM和DEEP部分，提高模型效果。
* DCN
* PNN

## 2、Sentiment_analysis
  这部分是在语义处理中的总结，比如曾经做过多模态的语义召回即和该目录下的做法雷同，主要采用keras处理。这里包含一个很关键的点，语义处理相对工业化的推荐算法从空间上相对来说是最简单的，也就是说没有大规模的（稀疏）特征，这就意味着训练集规模极小，单机就可以搞定，这也是采用keras的原因，即处理小型数据集的算法任务。
  
* Semantic_match
* Sentiment_analysis
* Seq2seq

## 3、FeatureEngineering
  特征工程的反思与总结。
