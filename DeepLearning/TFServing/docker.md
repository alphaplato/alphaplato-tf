# tf-serving

## docker 部署
docker部署的方式主要提供给算法开发同学使用，提供一个简单的serving以模拟线上服务标准输入与输出是否符合预期。
准备好模型，依次按照以下步骤进行：

- docker run -p 8501:8501 --mount type=bind,source=/Users/shuguang/alphaplato/alphaplato/DeepLearning/DeepFM/export,target=/models/DeepFM -e MODEL_NAME=DeepFM -t tensorflow/serving &
- curl http://localhost:8501/v1/models/DeepFM
- curl http://localhost:8501/v1/models/DeepFM/metadata
- curl -d '{"instances": [{"age":37,"workclass":"Private","education":"Some-college","education-num":"10","marital-status":"Married-civ-spouse","occupation":"Exec-managerial","relationship":"Husband","race":"Black","sex":"Male","capital-gain":0,"capital-loss":0,"hours-per-week":80,"native-country":"United-States"}]}' -X POST http://localhost:8501/v1/models/DeepFM:predict