

import os
import json
import numpy as np

import scipy
from sklearn.metrics import pairwise_distances as sparse_cdist

from .datasets import (GTSRB_truncated, MIOTCD_truncated, Vehicle10_truncated, StanfordCars_truncated)
from .utils import load_gtsrb_data, load_stanford_cars_data, load_miotcd_data, load_vehicle10_data, load_clisa_data, load_tlight10_data, record_net_data_stats

class TLight10Partition:
    def __init__(self, root, partition):
        self.root = root
        self.partition = partition

        percentage = eval(partition[10:]) 

        if not os.path.exists(os.path.join(root, 'tlight-10')):
            data_dir = os.path.join(root, 'tlight-10')
            raise RuntimeError(f'{data_dir} is not a directory')               

        with open(os.path.join(root, 'tlight-10', f'fed_meta_{percentage}.json'), 'r') as file:
            self.fed_meta_info = json.load(file)

        self.net_dataidx_map = {}
        self.traindata_cls_counts = {}
        self.position = {}
        
        self.__build__()

    def __build__(self, ):
        for key in self.fed_meta_info.keys():
            self.net_dataidx_map[eval(key)] = np.array(self.fed_meta_info[key]['index'])
            
            cls = {}
            for item in self.fed_meta_info[key]['class']:
                cls[eval(item)] = self.fed_meta_info[key]['class'][item]
            
            self.traindata_cls_counts[eval(key)] = cls
            self.position[eval(key)] = (self.fed_meta_info[key]['x'], self.fed_meta_info[key]['y'], self.fed_meta_info[key]['city'])
            
        assert len(self.net_dataidx_map) == len(self.traindata_cls_counts)
        assert len(self.traindata_cls_counts) == len(self.position)   

class CLISAPartition:
    def __init__(self, root, partition):
        self.root = root
        self.partition = partition
        
        percentage = eval(partition[10:]) 
        
        if not os.path.exists(os.path.join(root, 'cropped_lisa')):
            data_dir = os.path.join(root, 'cropped_lisa')
            raise RuntimeError(f'{data_dir} is not a directory')        
        
        with open(os.path.join(root, 'cropped_lisa', f'fed_meta_{percentage}.json'), 'r') as file:
            self.fed_meta_info = json.load(file)
        
        self.net_dataidx_map = {}
        self.traindata_cls_counts = {}
        self.position = {}
        
        self.__build__()
    
    def __build__(self):
        for key in self.fed_meta_info.keys():
            self.net_dataidx_map[eval(key)] = np.array(self.fed_meta_info[key]['index'])
            
            cls = {}
            for item in self.fed_meta_info[key]['class']:
                cls[eval(item)] = self.fed_meta_info[key]['class'][item]
            
            self.traindata_cls_counts[eval(key)] = cls
            self.position[eval(key)] = (self.fed_meta_info[key]['x'], self.fed_meta_info[key]['y'], self.fed_meta_info[key]['city'])
            
        assert len(self.net_dataidx_map) == len(self.traindata_cls_counts)
        assert len(self.traindata_cls_counts) == len(self.position)

class GTSRBPartition:
    def __init__(self, root, partition):
        self.root = root
        self.partition = partition
        
        percentage = eval(partition[10:]) 
        
        if not os.path.exists(os.path.join(root, 'gtsrb')):
            data_dir = os.path.join(root, 'gtsrb')
            raise RuntimeError(f'{data_dir} is not a directory')        
        
        with open(os.path.join(root, 'gtsrb', 'GTSRB', f'fed_meta_{percentage}.json'), 'r') as file:
            self.fed_meta_info = json.load(file)
        
        self.net_dataidx_map = {}
        self.traindata_cls_counts = {}
        self.position = {}
        
        self.__build__()
    
    def __build__(self):
        for key in self.fed_meta_info.keys():
            self.net_dataidx_map[eval(key)] = np.array(self.fed_meta_info[key]['index'])
            
            cls = {}
            for item in self.fed_meta_info[key]['class']:
                cls[eval(item)] = self.fed_meta_info[key]['class'][item]
            
            self.traindata_cls_counts[eval(key)] = cls
            self.position[eval(key)] = (self.fed_meta_info[key]['x'], self.fed_meta_info[key]['y'], self.fed_meta_info[key]['city'])
            
        assert len(self.net_dataidx_map) == len(self.traindata_cls_counts)
        assert len(self.traindata_cls_counts) == len(self.position)


class MIOTCDPartition:
    def __init__(self, root, partition):
        self.root = root
        self.partition = partition

        percentage = eval(partition[10:]) 

        if not os.path.exists(os.path.join(root, 'mio-tcd')):
            data_dir = os.path.join(root, 'mio-tcd')
            raise RuntimeError(f'{data_dir} is not a directory')               

        with open(os.path.join(root, 'mio-tcd', f'fed_meta_{percentage}.json'), 'r') as file:
            self.fed_meta_info = json.load(file)

        self.net_dataidx_map = {}
        self.traindata_cls_counts = {}
        self.position = {}
        
        self.__build__()

    def __build__(self, ):
        for key in self.fed_meta_info.keys():
            self.net_dataidx_map[eval(key)] = np.array(self.fed_meta_info[key]['index'])
            
            cls = {}
            for item in self.fed_meta_info[key]['class']:
                cls[eval(item)] = self.fed_meta_info[key]['class'][item]
            
            self.traindata_cls_counts[eval(key)] = cls
            self.position[eval(key)] = (self.fed_meta_info[key]['x'], self.fed_meta_info[key]['y'], self.fed_meta_info[key]['city'])
            
        assert len(self.net_dataidx_map) == len(self.traindata_cls_counts)
        assert len(self.traindata_cls_counts) == len(self.position)        

class Vehicle10Partition:
    def __init__(self, root, partition):
        self.root = root
        self.partition = partition

        percentage = eval(partition[10:]) 

        if not os.path.exists(os.path.join(root, 'vehicle-10')):
            data_dir = os.path.join(root, 'vehicle-10')
            raise RuntimeError(f'{data_dir} is not a directory')               

        with open(os.path.join(root, 'vehicle-10', f'fed_meta_{percentage}.json'), 'r') as file:
            self.fed_meta_info = json.load(file)

        self.net_dataidx_map = {}
        self.traindata_cls_counts = {}
        self.position = {}
        
        self.__build__()

    def __build__(self, ):
        for key in self.fed_meta_info.keys():
            self.net_dataidx_map[eval(key)] = np.array(self.fed_meta_info[key]['index'])
            
            cls = {}
            for item in self.fed_meta_info[key]['class']:
                cls[eval(item)] = self.fed_meta_info[key]['class'][item]
            
            self.traindata_cls_counts[eval(key)] = cls
            self.position[eval(key)] = (self.fed_meta_info[key]['x'], self.fed_meta_info[key]['y'], self.fed_meta_info[key]['city'])
            
        assert len(self.net_dataidx_map) == len(self.traindata_cls_counts)
        assert len(self.traindata_cls_counts) == len(self.position) 

class StanfordCarsPartition:
    def __init__(self, root, partition):
        self.root = root
        self.partition = partition

        percentage = eval(partition[10:]) 

        if not os.path.exists(os.path.join(root, 'stanford_cars')):
            data_dir = os.path.join(root, 'stanford_cars')
            raise RuntimeError(f'{data_dir} is not a directory')               

        with open(os.path.join(root, 'stanford_cars', f'fed_meta_{percentage}.json'), 'r') as file:
            self.fed_meta_info = json.load(file)

        self.net_dataidx_map = {}
        self.traindata_cls_counts = {}
        self.position = {}
        
        self.__build__()

    def __build__(self, ):
        for key in self.fed_meta_info.keys():
            self.net_dataidx_map[eval(key)] = np.array(self.fed_meta_info[key]['index'])
            
            cls = {}
            for item in self.fed_meta_info[key]['class']:
                cls[eval(item)] = self.fed_meta_info[key]['class'][item]
            
            self.traindata_cls_counts[eval(key)] = cls
            self.position[eval(key)] = (self.fed_meta_info[key]['x'], self.fed_meta_info[key]['y'], self.fed_meta_info[key]['city'])
            
        assert len(self.net_dataidx_map) == len(self.traindata_cls_counts)
        assert len(self.traindata_cls_counts) == len(self.position) 

def partition_data_(dataset, datadir, logdir, partition, n_parties, beta=0.4, local_view=False):
    if dataset == 'gtsrb':
        X_train, y_train, X_test, y_test = load_gtsrb_data(datadir, resize=(32, 32))
        gtsrb_partition = GTSRBPartition(root=datadir, partition=partition)
        
        net_dataidx_map = gtsrb_partition.net_dataidx_map
        traindata_cls_counts = gtsrb_partition.traindata_cls_counts
        position = gtsrb_partition.position

        n_parties = len(position)

    elif dataset == 'miotcd':
        X_train, y_train, X_test, y_test = load_miotcd_data(datadir, resize=(32, 32))
        miotcd_partition = MIOTCDPartition(root=datadir, partition=partition)
        
        net_dataidx_map = miotcd_partition.net_dataidx_map
        traindata_cls_counts = miotcd_partition.traindata_cls_counts
        position = miotcd_partition.position

        n_parties = len(position)

    elif dataset == 'vehicle10':
        X_train, y_train, X_test, y_test = load_vehicle10_data(datadir, resize=(32, 32))
        vehicle10_partition = Vehicle10Partition(root=datadir, partition=partition)
        
        net_dataidx_map = vehicle10_partition.net_dataidx_map
        traindata_cls_counts = vehicle10_partition.traindata_cls_counts
        position = vehicle10_partition.position

        n_parties = len(position)

    elif dataset == 'cropped_lisa':
        X_train, y_train, X_test, y_test = load_clisa_data(datadir, resize=(32, 32))
        clisa_partition = CLISAPartition(root=datadir, partition=partition)
        
        net_dataidx_map = clisa_partition.net_dataidx_map
        traindata_cls_counts = clisa_partition.traindata_cls_counts
        position = clisa_partition.position

        n_parties = len(position)

    elif dataset == 'tlight10':
        X_train, y_train, X_test, y_test = load_tlight10_data(datadir, resize=(32, 32))
        tlight10_partition = TLight10Partition(root=datadir, partition=partition)
        
        net_dataidx_map = tlight10_partition.net_dataidx_map
        traindata_cls_counts = tlight10_partition.traindata_cls_counts
        position = tlight10_partition.position

        n_parties = len(position)

    elif dataset == 'stanford_cars':
        X_train, y_train, X_test, y_test = load_stanford_cars_data(datadir, resize=(224, 224))
        stanfordcars_partition = StanfordCarsPartition(root=datadir, partition=partition)
        
        net_dataidx_map = stanfordcars_partition.net_dataidx_map
        traindata_cls_counts = stanfordcars_partition.traindata_cls_counts
        position = stanfordcars_partition.position

        n_parties = len(position)

    # print(f'partition: {net_dataidx_map}')
    print('Data statistics Train:\n %s \n' % str(traindata_cls_counts))
    
    if local_view:
        net_dataidx_map_test = {i: [] for i in range(n_parties)}
        for k_id, stat in traindata_cls_counts.items():
            labels = list(stat.keys())
            for l in labels:
                idx_k = np.where(y_test==l)[0]
                net_dataidx_map_test[k_id].extend(idx_k.tolist())

        testdata_cls_counts = record_net_data_stats(y_test, net_dataidx_map_test, logdir)
        print('Data statistics Test:\n %s \n' % str(testdata_cls_counts))
    else: 
        net_dataidx_map_test = None 
        testdata_cls_counts = None 

    return (X_train, y_train, X_test, y_test, net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts, position)