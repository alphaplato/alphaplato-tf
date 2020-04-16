# tf-serving

## docker 部署

* docker run -p 8501:8501 --mount type=bind,source=/Users/shuguang/alphaplato/alphaplato/DeepLearning/DeepFM/export,target=/models/DeepFM -e MODEL_NAME=DeepFM -t tensorflow/serving &
* curl http://localhost:8501/v1/models/DeepFM
* curl http://localhost:8501/v1/models/DeepFM/metadata

* curl -d '{"instances": [{"age":37,"workclass":"Private","education":"Some-college","education-num":"10","marital-status":"Married-civ-spouse","occupation":"Exec-managerial","relationship":"Husband","race":"Black","sex":"Male","capital-gain":0,"capital-loss":0,"hours-per-week":80,"native-country":"United-States"}]}' -X POST http://localhost:8501/v1/models/DeepFM:predict


## kubernetes 部署

### 启动k8s(已有集群可忽略)
* kind create cluster --name Janus 

### 1、生成镜像
* docker run -d --name serving_base tensorflow/serving
* docker cp model serving_base:/models/deepfm
* docker commit --change "ENV MODEL_NAME deepfm" serving_base $USER/deepfm_serving
* docker kill serving_base
* docker rm serving_base

### 2、测试镜像
* docker run -p 8500:8500 -t $USER/deepfm_serving &
* curl -d '{"instances": [{"age":37,"workclass":"Private","education":"Some-college","education-num":"10","marital-status":"Married-civ-spouse","occupation":"Exec-managerial","relationship":"Husband","race":"Black","sex":"Male","capital-gain":0,"capital-loss":0,"hours-per-week":80,"native-country":"United-States"}]}' -X POST http://localhost:8500/v1/models/DeepFM:predict

### 3、上传镜像
* docker tag $USER/deepfm_serving alphaplato/beta:deepfm_serving
* docker push alphaplato/beta:deepfm_serving
