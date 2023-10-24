# alphaplato

e-mail：plato.sg.lee@gmail.com

## 1、[DeepLearning](https://github.com/alphaplato/alphaplato/tree/master/DeepLearning)
   这部分是在推荐算法中的总结，面向的是生产环境下的算法开发，在完成网络结构时采用了tensorflow的feature_column函数以适配分布式特征处理的情境。开发上，将网络及pipeline在代码层面进行了结构化处理，以提高工程开发效率；技巧上，包含序列化网络、随机负采样等设计。另外，也给出了在工业条件下，tensorflow serving部署的方案及demo。
   
* DeepFM
* DCN
* PNN
* XDeepFM
* DIN
* DIEN
* ESMM
* DeepMatch
* TFServing

## 2、[SentimentAnalysis](https://github.com/alphaplato/alphaplato/tree/master/SemanticAnalysis)
  这部分是在语义处理中的总结，比如曾经做过的多模态的语义召回、情感分析等。这里包含一个很关键的点，语义处理相对工业化的推荐算法从空间上相对来说是最简单的，也就是说没有大规模的（稀疏）特征，通常训练集规模极小，单机就可以搞定。由于该项目项目大部分算法是本人早起工作积累，主要使用的是keras，其非常适合单机条件下处理文本数据集的算法任务。当然目前算法逐渐转向基础特征和文本、图像甚至视频的多模态网络的方向发展，涉及的数据集自然趋向庞大，网络的trick空间也趋向增大，而tensorflow也逐渐集成了keras的功能，因此开发上更趋灵活。
  语义分析个人经验是1）最好有明确算法的语义目的，2）预训练，3）好的分词工具。
  
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
