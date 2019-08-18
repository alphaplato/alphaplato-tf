#!/bin/python3
import os
import numpy as np
import pandas as pd

class dataset(object):
    def __init__(self,
                 ftrain = 'train.txt',
                 ftest = 'test.txt',
                 feature_list = 'feature_list.conf',
                 feature_map = 'feature_map.conf',
                 path='./data/'):
        self.path = path
        self.ftrain = ftrain
        self.ftest = ftest
        self.feature_list = feature_list
        self.feature_map = feature_map
        self.__read_feature_list()
        self.__makesvm()
        self.__to_save()

    def __read_feature_list(self):
        dicrete_feature = []
        continuous_feature = []
        feature_columns = []
        with open(self.feature_list,'r') as fr:
            for line in fr.readlines():
                name,type = line.strip().split('\t')
                if 'D' == type:
                    dicrete_feature.append(name)
                if 'C' == type:
                    continuous_feature.append(name)
                feature_columns.append(name)
        feature_columns.append('label')
        self.dicrete_feature = dicrete_feature
        self.continuous_feature = continuous_feature
        self.feature_columns = feature_columns

    def __makesvm(self):
        train = pd.read_csv(self.ftrain,names=self.feature_columns)
        test = pd.read_csv(self.ftest,header=None,names=self.feature_columns)
        train = train.drop(['fnlwgt'], axis = 1)
        test = test.drop(['fnlwgt'], axis = 1)
        self.continuous_feature.remove('fnlwgt')
        dataset = pd.concat([train,test])
        feature_size = 0
        field_size = 0
        records = []
        fw = open(self.feature_map,'w')
        for name in self.continuous_feature:
            record = '{0}\t{1}\n'.format(feature_size, name)
            records.append(record)
            train[name] = str(feature_size) + ":" + train[name].astype(str)
            test[name] = str(feature_size) + ":" + test[name].astype(str)
            feature_size = feature_size + 1
            field_size = field_size + 1
        #print(feature_size,field_size)   
        
        for name in self.dicrete_feature:
            feature_dict = {}
            values = set(dataset[name])
            for val in values:
                record = '{0}\t{1}={2}\n'.format(feature_size, name, val)
                records.append(record)
                feature_dict[val] = '{0}:1'.format(feature_size)
                feature_size = feature_size + 1
            #print(name,field_size)
            field_size = field_size + 1 
            train[name] = train[name].map(feature_dict)
            test[name] = test[name].map(feature_dict)
        print(feature_size,field_size)
        #print(self.dicrete_feature,self.continuous_feature)

        train['label'] = train['label'].apply(lambda x: 1 if '<=' in str(x) else 0)
        test['label'] = test['label'].apply(lambda x:  1 if '<=' in str(x) else 0)

        fw.writelines(records)
        fw.close()
        self.train = train
        self.test = test

    def __to_save(self):
        self.train.to_csv(self.path + 'train.data',index=0,header=0)
        self.test.to_csv(self.path + 'test.data',index=0,header=0)

if __name__ == '__main__':
    ftrain = '../Example/adult.data'
    ftest = '../Example/adult.test'
    feature_list = 'conf/feature_list.conf'
    feature_map = 'conf/feature_map.conf'
    data = dataset(ftrain,ftest,feature_list,feature_map)
    #print(data.test['label'])
