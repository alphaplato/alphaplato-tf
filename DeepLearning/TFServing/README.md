# tesorflow serving

主要包含两种部署调用，[docker](https://github.com/alphaplato/alphaplato/blob/master/DeepLearning/TFServing/docker.md)和[k8s](https://github.com/alphaplato/alphaplato/blob/master/DeepLearning/TFServing/k8s.md)，这两种方式均为官方推荐的方式。

docker部署偏向测试使用。k8s部署为生产环境的部署方式之一，在实际的测试中1000条样本在deepFM模型下serving响应耗时约25ms，说明谷歌推荐的生产环境k8s的部署方式响应速度不错，是很实用的。

在k8s部署时考虑到生产环境模型不可能放在本地路径，推荐采用hdfs的存放模型，实现tensorflow serving的分布式部署。因此本项目工程在k8s部署时，部署集成了hdfs的访问功能。