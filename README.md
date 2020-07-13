# alphaplato

## 1、[DeepLearning](https://github.com/alphaplato/alphaplato/tree/master/DeepLearning)
   这部分是在推荐算法中的总结，面向的是生产环境下的算法开发，在完成网络结构时采用了tensorflow的feature_column函数以适配分布式特征处理的情境。开发上，将网络及pipeline在代码层面进行了结构化处理，以提高工程开发效率；技巧上，包含序列化网络、随机负采样等设计。另外，也给出了在工业条件下，tensorflow serving部署的方案及demo。
   
* DeepFM
* DCN
* PNN
* XDeepFM
* DIN
* DIEN
* DeepMatch
* TFServing

## 2、[SentimentAnalysis](https://github.com/alphaplato/alphaplato/tree/master/SemanticAnalysis)
  这部分是在语义处理中的总结，比如曾经做过多模态的语义召回即和该目录下的做法雷同，主要采用keras处理。这里包含一个很关键的点，语义处理相对工业化的推荐算法从空间上相对来说是最简单的，也就是说没有大规模的（稀疏）特征，这就意味着训练集规模极小，单机就可以搞定，这也是本项目大部分算法采用keras的原因，非常适合单机条件下处理文本数据集的算法任务。
  语义分析个人经验是1）要有明确算法的语义目的，2）word2vec预训练，3）好的分词工具。
  
* Semantic_match
* Sentiment_analysis
* Seq2seq
* Show_and_tell

## 3、[FeatureEngineering](https://github.com/alphaplato/alphaplato/tree/master/FeatureEngineering)
  特征工程的反思与总结。

## 4、[IdealArchitecture](https://github.com/alphaplato/alphaplato/tree/master/IdealArchitecture)
  算法的工程架构反思与总结。这里会从算法角度，反演整体的架构，对常见场景下的召回、排序环节的工程结构化与设计思路做一些总结。这部分主要包含两个方面的经验：
  * 合理的算法工程架构。
  * 可适配的高效算法服务组件。

## 5、[ReinforcementLearning](https://github.com/alphaplato/alphaplato/tree/master/ReinforcementLearning)
