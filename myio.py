'''
Created on 31 mai 2016

@author: wangxin
'''
import collections
import os 
import wsl_object
import numpy as np
import math
import myio
from gensim.models.lsimodel import stochastic_svd

###HELP FUNCTIONS###
def scale2RowNumber(scale):
    return 1+(100-scale)/10

def file2FloatList(filepath):
    with open(filepath) as fp:
        return [float(line.strip()) for line in fp.readlines()]
###HELP FUNCTIONS###

def fillInstances(scale, filename, feature_root_folder, gaze_root_folder):
    instance_feature = collections.defaultdict(lambda:None)
    instance_gaze = collections.defaultdict(lambda:None)
    
    for d1 in range(scale2RowNumber(scale)):
        for d2 in range(scale2RowNumber(scale)):
            feature_file_fp = os.path.join(feature_root_folder, str(scale), '_'.join([filename, str(d1), str(d2)+'.txt']))
            feature = file2FloatList(feature_file_fp)
            instance_feature[d1*scale2RowNumber(scale)+d2] = feature
            gaze_f = os.path.join(gaze_root_folder, str(scale), '_'.join([filename, str(d1), str(d2)+'.txt']))
            gaze_file_f = open(gaze_f)
            gaze_loss = gaze_file_f.readline()
            instance_gaze[d1*scale2RowNumber(scale)+d2] = gaze_loss
            gaze_file_f.close()
    return instance_feature, instance_gaze


def getBagExample(scale, filename, label, feature_root_folder, gaze_root_folder):
    instances = fillInstances(scale, filename, feature_root_folder, gaze_root_folder)
    bag = wsl_object.GazeImageBag(filename, label, instances[0], instances[1])
    return bag

def load_binary_bags(example_list_root_folder, scale, featureFolder, gazeFolder, stochastic=True):
    import random
    with open(example_list_root_folder) as f:
        example_number = f.readline().strip()
                
        bags = []
        pos_cnt=0
        neg_cnt=0
        print 'Loading images:',
        for cnt, line in enumerate(f):
#             if cnt==1:
#                 break
            example_detail =line.split()
            full_path = example_detail[0]
            filename = full_path.split('/')[-1]
            label = example_detail[1]
            if label == "1":
                pos_cnt+=1
            if label == "0":
                neg_cnt+=1
            instance_number =  example_detail[2]
            bags.append(getBagExample(scale, filename, label, featureFolder, gazeFolder))
            if stochastic:
                random.shuffle(bags)
            if cnt%500==0:
                print str(cnt)+"/"+str(example_number)+",",
        print "positive image:%d, negative image:%d"%(pos_cnt, neg_cnt)
        return bags

def load_multiclass_bags(example_list_root_folder, scale, featureFolder, gazeFolder, stochastic=True):
    import random
    with open(example_list_root_folder) as f:
        example_number = f.readline().strip()
                
        bags = []
        print 'Loading images:',
        for cnt, line in enumerate(f):
#             if cnt==20:
#                 break
            example_detail =line.split()
            full_path = example_detail[0]
            filename = full_path.split('/')[-1]
            label = example_detail[1]
            instance_number =  example_detail[2]
            bags.append(getBagExample(scale, filename, label, featureFolder, gazeFolder))
            if stochastic:
                random.shuffle(bags)
            if cnt%500==0:
                print str(cnt)+"/"+str(example_number)+",",
        return bags

def organize_examples_by_instances(bags, category_number, nb_instance):
    """
    X =     [#instance * #example, #feature length]
    Gazes = [#instance * #example, 1]
    y =     [#example            , #categories]
    """

    X = np.ndarray(shape=(nb_instance * len(bags), len(bags[0].features[0])), dtype=float, order='F') 
    Gazes = np.ndarray(shape=(nb_instance * len(bags),1), dtype=float, order='F')
    y = np.zeros(shape=(len(bags),category_number), dtype=float, order='F')
    for cnt,bag in enumerate(bags):
        for instance_cnt in range(nb_instance):
            X[cnt*nb_instance + instance_cnt,:]= bags[cnt].features[instance_cnt]
            Gazes[cnt*nb_instance + instance_cnt,0]= 1-float(bags[cnt].gazes[instance_cnt])
        y[cnt,int(bags[cnt].label)]= 1
    return [X, y, Gazes]

def organize_examples_by_instances_gaze_multiclass(bags, category_number, nb_instance):
    """
    X =     [#instance * #example, #feature length]
    Gazes = [#instance * #example, #categories]
    y =     [#instance * #example, #categories]
    
    """
    X = np.ndarray(shape=(nb_instance * len(bags), len(bags[0].features[0])), dtype=float, order='F') 
    Gazes = np.zeros(shape=(nb_instance * len(bags),category_number), dtype=float, order='F')
    y = np.zeros(shape=(nb_instance * len(bags),category_number), dtype=float, order='F')
    for cnt,bag in enumerate(bags):
        for instance_cnt in range(nb_instance):
            X[cnt*nb_instance + instance_cnt,:]= bags[cnt].features[instance_cnt]
            Gazes[cnt*nb_instance + instance_cnt,int(bags[cnt].label)]= 1-float(bags[cnt].gazes[instance_cnt])
            y[cnt*nb_instance + instance_cnt,int(bags[cnt].label)] = 1
    return [X, y, Gazes]

def organize_examples_by_instances_classwise(bags, nb_instance):
    """
    X =     [#instance * #example_of_this_category, #feature length]
    Gazes = [#instance * #example_of_this_category, 1]
    y =     [#example_of_this_category            , 1]
    
    """
    
    example_num=0
    for cnt,bag in enumerate(bags):
        if int(bags[cnt].label) == 1:
            example_num+=1
    X = np.ndarray(shape=(nb_instance * example_num, len(bags[0].features[0])), dtype=float, order='F') 
    Gazes = np.zeros(shape=(nb_instance * example_num,1), dtype=float, order='F')
    y = np.ones(shape=(example_num,1), dtype=float, order='F')

    for cnt,bag in enumerate(bags):
        if not int(bags[cnt].label) == 1:
            continue
            for instance_cnt in range(nb_instance):
                X[cnt*nb_instance + instance_cnt,:]= bags[cnt].features[instance_cnt]
                Gazes[cnt*nb_instance + instance_cnt,int(bags[cnt].label)]= 1-float(bags[cnt].gazes[instance_cnt])
    return [X, y, Gazes]


def organize_examples_by_multiclass_bags(bags, category_number, nb_instance):
    X = np.ndarray(shape=(len(bags), nb_instance,
                          len(bags[0].features[0])), dtype=float, order='F') 
    Gazes = np.ndarray(shape=(len(bags), nb_instance), dtype=float, order='F') 
    y = np.zeros(shape=(len(bags),category_number), dtype=float, order='F')
    for cnt,bag in enumerate(bags):
        for instance_cnt in range(nb_instance):
            X[cnt, instance_cnt, :]= bags[cnt].features[instance_cnt]
            Gazes[cnt, instance_cnt] = bags[cnt].gazes[instance_cnt]
        y[cnt,int(bags[cnt].label)]= 1
        
    return [X, y, Gazes]

def organize_examples_by_binaryclass_bags(bags, nb_instance):
    X = np.ndarray(shape=(len(bags), nb_instance,
                          len(bags[0].features[0])), dtype=float, order='F') 
    Gazes = np.ndarray(shape=(len(bags), nb_instance), dtype=float, order='F') 
    y = np.zeros(shape=(len(bags),2), dtype=float, order='F')
    for cnt,bag in enumerate(bags):
        for instance_cnt in range(nb_instance):
            X[cnt, instance_cnt, :]= bags[cnt].features[instance_cnt]
            Gazes[cnt, instance_cnt] = bags[cnt].gazes[instance_cnt]
        y[cnt,int(bags[cnt].label)]= 1
    return [X, y, Gazes]


def batching(v, batch_size):
    start_index=0
    batched_v=[]
    for end_index in range(1,1+int(math.ceil(np.shape(v)[0]/(batch_size+0.0)))):
        batched_v.append(v[start_index*batch_size: min(np.shape(v)[0],end_index*batch_size)])
        start_index=end_index
    return np.array(batched_v)

def batch_batch(X_train , y_train, Gazes_train, X_val, y_val, Gazes_val, train_batch_size):
    batch_X_train = batching(X_train, train_batch_size)
    batch_y_train = batching(y_train, train_batch_size)
    batch_Gazes_train = batching(Gazes_train, train_batch_size)
    batch_X_val = batching(X_val, len(X_val))
    batch_y_val = batching(y_val, len(X_val))
    batch_Gazes_val = batching(Gazes_val, len(X_val))
    return [batch_X_train, batch_y_train, batch_Gazes_train,
            batch_X_val, batch_y_val, batch_Gazes_val]
    
