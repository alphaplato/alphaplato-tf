# tesorflow serving

## 一、概述
主要包含两种部署调用，docker和k8s，这两种方式均为官方推荐的方式。

docker部署偏向测试使用。k8s部署为生产环境的部署方式之一，在实际的测试中1000条样本在deepFM模型下serving响应耗时约25ms，说明谷歌推荐的生产环境k8s的部署方式响应速度不错，是很实用的。

在k8s部署时考虑到生产环境模型不可能放在本地路径，推荐采用hdfs的存放模型，实现tensorflow serving的分布式部署。因此本项目工程在k8s部署时，部署集成了hdfs的访问功能。

## 二、docker 部署
docker部署的方式主要提供给算法开发同学使用，提供一个简单的serving用以模拟线上服务标准输入与输出是否符合预期。

准备好模型，依次按照以下步骤进行：
- 将模型拷贝到/tmp/DeepFM/export（路径可自行修改）下
- 启动docker：docker run -p 8501:8501 --mount type=bind,source=/tmp/DeepFM/export,target=/models/DeepFM -e MODEL_NAME=DeepFM -t tensorflow/serving &
- 查看模型：curl http://localhost:8501/v1/models/DeepFM
- 查看metadata：curl http://localhost:8501/v1/models/DeepFM/metadata
- 测试：curl -d '{"instances": [{"age":37,"workclass":"Private","education":"Some-college","education-num":"10","marital-status":"Married-civ-spouse","occupation":"Exec-managerial","relationship":"Husband","race":"Black","sex":"Male","capital-gain":0,"capital-loss":0,"hours-per-week":80,"native-country":"United-States"}]}' -X POST http://localhost:8501/v1/models/DeepFM:predict
## 三、k8s 部署
### 3.1 测试部署
k8s的测试部署主要提供给算法团队、工程团队或者是运维团队测试，是测试算法及serving是否能够正常通信，及在正式部署k8前检验服务的有效性。

首先启动k8s(已有集群可忽略)
> kind create cluster --name Janus

按照以下步骤进行部署：
#### 3.1.1 生成镜像
- docker run -d --name serving_base tensorflow/serving
- docker cp model serving_base:/models/deepfm
- docker commit --change "ENV MODEL_NAME deepfm" serving_base $USER/deepfm_serving
- docker kill serving_base
- docker rm serving_base
#### 3.1.2 测试镜像
- docker run -p 8500:8500 -t $USER/deepfm_serving &
- curl -d '{"instances": [{"age":37,"workclass":"Private","education":"Some-college","education-num":"10","marital-status":"Married-civ-spouse","occupation":"Exec-managerial","relationship":"Husband","race":"Black","sex":"Male","capital-gain":0,"capital-loss":0,"hours-per-week":80,"native-country":"United-States"}]}' -X POST http://localhost:8500/v1/models/DeepFM:predict
#### 3.1.3 上传镜像
- docker tag $USER/deepfm_serving alphaplato/beta:deepfm_serving
- docker push alphaplato/beta:deepfm_serving

测试完成上传保存可以再次使用。

### 3.2 正式部署
假设从头开始生成镜像，基于k8s部署，并集成hdfs的访问路径。
#### 3.2.1 生成镜像
##### A 测试hdfs镜像
* 参照以下内容构建dockerfile，注意内部包含的java和hadoop需要提前准备好。

> MAINTAINER lishuguang@sdiread.com

> FROM ubuntu:16.04

> COPY java /usr/local/java
> COPY hadoop /root/hadoop

> ENV JAVA_HOME /usr/local/java
> ENV HADOOP_HOME /root/hadoop
> ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${JAVA_HOME}/jre/lib/amd64/server

> ENTRYPOINT ["bash"]

* docker build -t tensorflow_serving:1.14-hadoop-test .
* docker run -it --rm tensorflow_serving:1.14-hadoop
* 检测是否能够和目标hdfs系统正常通信，比如采用常用的ls等命令测试，修复响应问题
##### B 生成tensorflow serving镜像

* 构建dockerfile(本路径下)，注意内部包含的java和hadoop需要提前准备好。
* docker build -t tensorflow/serving:tf-serving-hdfs .
#### 3.2.2 测试镜像
- docker run -p 9000:9000 -p 8500:8500 --name=test -e MODEL_NAME=deepfm -e MODEL_BASE_PATH=hdfs://172.16.32.15:4007/export/shidian/rec/sort/models -it tensorflow/serving:tf-serving-hdfs
- grpc测试： sh run_in_docker.sh python3 deepfm_client_grpc.py --server=localhost:8500
- rest api 测试： sh run_in_docker.sh python3 deepfm_client.py  --server=localhost:9000
#### 3.2.3 上传镜像(以腾讯云为例)
- docker tag tensorflow/serving:tf-serving-hdfs $USER/tensorflow:tf-serving-hdfs
- docker push ccr.ccs.tencentyun.com/sd_rec/tensorflow:tf-serving-hdfs
#### 3.2.4 k8s部署
- kubectl create -f deepfm_k8s.yaml
##### 部署测试
* grpc测试： sh run_in_docker.sh python3 deepfm_client_grpc.py --server=k8s_ip:8500
* rest api 测试： sh run_in_docker.sh python3 deepfm_client.py  --server= k8s_ip:9000

## 四 备注
该服务目前只服务单模型，通常情况下改变模型需要新的部署。serving支持多模型并存，可通过配置文件实现，本文档视后续线上使用情况再做补充。
