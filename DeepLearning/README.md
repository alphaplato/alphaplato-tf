## 工业化深度学习算法

### 一、算法列表

* DCN
* DeepFM
* PNN
* XDeepFM

* DIN 
* DIEN

* DeepMatch

**已完成框架升级，并将DCN，DeepFM、PNN、XDeepFM更新至最新版；框架升级后，对于算法实现，只需要修改制定的model和fg中的两个函数即可实现数据到nn网络闭环，其他代码复用。大大提高算法开发效率。**

**打包成package，可将不同算法文件文件下的重复文件保存为一份。**

### 二、demo使用
#### 1、执行 python3 csv_to_tfrecord.py
#### 2、执行训练程序 
##### 1) 单机模式：
去掉--dist_mode参数,执行 sh demo.sh 
##### 2）分布式模式：增加--dist_mode=1 --job_name=${1}    
分别执行：
* [1] sh demo.sh ps
* [2] sh demo.sh chief
* [3] sh demo.sh worker    
注意model清除参数的设置。
##### 3）集群模式：
增加--dist_mode=2，执行 sh demo.sh 

#### 3、docker下的使用说明


如果在docker下使用，首先执行如下命令：

> docker container run -p 0.0.0.0:6006:6006 -it -v $PWD:/tmp -w /tmp tensorflow/tensorflow:1.15.2-py3 bash

* -p: Tensorboard will be served in our browser on port of localhost:6006
* -v -w: The directory of docker will be loaded on the directory of localhost (-v hostDir:containerDir -w workDir)


### 三、数据集来源

使用的数据集主要如下：

* 预测收入的数据集，来源于[Adult Data Set](https://archive.ics.uci.edu/ml/datasets/adult)
* 亚马逊[书商点击数据集](https://github.com/mouna99/dien/blob/master/data.tar.gz)
* 电影评分数据集[MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/25m/)
