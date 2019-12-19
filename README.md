# alphaplato

## 1、DeepLearning
   这部分是在推荐算法中的总结，面向的是大规模分布式算法，在完成网络结构时采用了tensorflow的feature_column函数以适配分布式特征处理的情境。
   在使用深度网络其中需要注意的一点是数据的归一化，因为不同数值特征通常不在一个量级，比如CTR特征和counter计数特征，直接结果是网络不稳定甚至不收敛；所以在网络层级上适当增加batch_normalization的trick非常重要，该trick是将网络参数高斯归一化，在层输入前和输出使用是不错的选择。
   
* DeepFM
* DCN
* PNN
* XDeepFM
* DIN
* DIEN

## 2、Sentiment_analysis
  这部分是在语义处理中的总结，比如曾经做过多模态的语义召回即和该目录下的做法雷同，主要采用keras处理。这里包含一个很关键的点，语义处理相对工业化的推荐算法从空间上相对来说是最简单的，也就是说没有大规模的（稀疏）特征，这就意味着训练集规模极小，单机就可以搞定，这也是采用keras的原因，适合处理文本数据集的算法任务。
  语义分析个人经验是明确语义目的，word2vec的预训练是个不错的trick。
  
* Semantic_match
* Sentiment_analysis
* Seq2seq

## 3、FeatureEngineering
  特征工程的反思与总结。

## 4、IdealArchitecture
  算法的工程架构反思与总结。这里会从算法角度，反演整体的架构，对常见场景下的召回、排序环节的工程结构化与设计思路做一些总结。主要包含两个方面：
  * 高效的算法是建立在合理的工程架构之上的。
  * 高效算法工程下的召回、排序的常用的算法实例。