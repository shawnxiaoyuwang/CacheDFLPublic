# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import matplotlib.pyplot as plt

# plt.style.use("seaborn-white")
import argparse
import random
import torch
cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch import optim
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.multiprocessing as mp
import time
import os    
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import copy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import datetime

from Cache_algorithm import kick_out_timeout_model,kick_out_timeout_model_list,update_model_cache_mixing,update_model_cache_car_to_car_p,prune_cache,update_model_cache_car_to_taxi_p,update_model_cache_taxi_to_taxi_p,update_model_cache_car_to_taxi,update_model_cache_taxi_to_taxi,update_model_cache_distribution, update_model_cache_global,kick_out_timeout_model_cache_info, update_model_cache_fresh,cache_average_process,cache_average_process_fresh,cache_average_process_mixing,update_model_cache,duration_in_future,update_model_cache_only_one,weighted_cache_average_process,update_best_model_cache,cache_average_process_fresh_without_model,update_model_cache_fresh_count, update_model_cache_fresh_v3,cache_average_process_fresh_v3
from aggregation import average_weights,normal_training_process,average_process,normal_train,subgradient_push_process, weighted_average_process
from utils_cnn import test
from model import get_P_matrix,CNNMnist,Cifar10CnnModel,CNNFashion_Mnist,AlexNet,DNN_harbox
from models import ResNet18
from data import get_mnist_iid, get_mnist_taxi_area,get_mnist_imbalance,get_mnist_dirichlet,initial_mnist, update_training_subset,get_dataloader_by_indices, initial_training_subset,get_cifar10_iid, get_cifar10_imbalance,get_cifar10_dirichlet,get_fashionmnist_taxi_area,get_fashionmnist_iid, get_fashionmnist_imbalance,get_fashionmnist_dirichlet,get_harbox_iid,get_harbox_imbalance,get_harbox_dirichlet
from road_sim import generate_roadNet_pair_list_v2, generate_roadNet_pair_list,generate_roadNet_pair_area_list
import seed_setter
Randomseed = seed_setter.set_seed()

# random_seed = 10086
# random.seed(random_seed)
# np.random.seed(random_seed)
np.set_printoptions(precision=4, suppress=True)

# Create the parser
parser = argparse.ArgumentParser(description="Configure script parameters and select algorithm")

# Add arguments
parser.add_argument("--suffix", type=str, default="", help="Suffix in the folder")
parser.add_argument("--note", type=str, default="N/A", help="Special_notes")
# parser.add_argument("--random_seed", type=int, default=10086, help="Random seed")
parser.add_argument("--task", type=str, choices=[
    'mnist', 'fashionmnist', 'cifar10','harbox'], help="Choose dataset task to run")
parser.add_argument("--local_ep", type=int, default=10, help="Number of local epochs")
parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
parser.add_argument("--decay_rate", type=float, default=0.02, help="Decay rate")
parser.add_argument("--decay_round", type=int, default=200, help="Decay round")
parser.add_argument("--car_meet_p", type=float, default=1./9, help="Car meet probability")
parser.add_argument("--alpha_time", type=float, default=0.01, help="Alpha time")
parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for Dirchlet distribution, lower alpha increases heterogeneity")
parser.add_argument("--distribution", type=str, choices=[
    'iid', 'non-iid','dirichlet'], help="Choose data distirbution")
parser.add_argument("--aggregation_metric", type=str, default="mean", help="Aggregation metric")
parser.add_argument("--cache_update_metric", type=str, default="mean", help="Cache update metric")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--sample_size", type=int, default=200, help="Sample size")
parser.add_argument("--hidden_size", type=int, default=200, help="Hidden size")
parser.add_argument("--num_round", type=int, default=2000, help="Number of rounds")
parser.add_argument("--early_stop_round", type=int, default=20, help="Number of rounds test accuracy remain unchanged then stop")
parser.add_argument("--speed", type=float, default=13.59, help="Speed in m/s")
parser.add_argument("--communication_distance", type=int, default=100, help="Communication distance in meters")
parser.add_argument("--epoch_time", type=int, default=60, help="Time to finish on epoch in seconds")
# parser.add_argument("--importance", type=int, default=1, help="Importance")
parser.add_argument("--num_car", type=int, default=100, help="Number of cars")
parser.add_argument("--lr_factor", type=float, default=0.1, help="Learning rate factor")
parser.add_argument("--lr_patience", type=int, default=20, help="Learning rate patience")
parser.add_argument("--cache_size", type=int, default=3, help="Cache size")
# parser.add_argument("--parallel", action='store_true', help="Enable parallel processing")
# parser.add_argument('--non-parallel', dest='parallel', action='store_false')
# parser.set_defaults(parallel=False)
parser.add_argument("--augment", action='store_true', help="Enable augmentation")
parser.add_argument('--no-augment', dest='augment', action='store_false')
parser.set_defaults(augment=True)
parser.add_argument("--shards_allocation", nargs='+', type=int, default=[3,2,1,3,2,1,1,4,1,2]*10, help="Shards allocation")
parser.add_argument("--County", type=str, default="New York", help="County")
parser.add_argument('--kick_out', type= int, default = 3, help =  'Threshold to kick out in cache')
# parser.add_argument("--kick_out", action='store_true', help="Enable kick out")
# parser.add_argument('--no-kick_out', dest='kick_out', action='store_false')
# parser.set_defaults(kick_out=True)
parser.add_argument("--weighted_aggregation", action='store_true', help="Enable weighted aggregation")
parser.add_argument('--no-weighted_aggregation', dest='weighted_aggregation', action='store_false')
parser.set_defaults(weighted_aggregation=True)
parser.add_argument("--algorithm", type=str, choices=[
    'ml', 'cfl', 'dfl','dfl_fair','cache_one',
    'cache_all', 'fresh', 'fresh_update_fresh_by_dp', 'fresh_count',
    'fresh_history', 'best_test_cache','test'
], help="Choose the algorithm to run")
# Parse arguments
args = parser.parse_args()
task = args.task
# args.algorithm = 'cfl'
# Assign values to variables
local_ep = args.local_ep
lr = args.lr
decay_rate = args.decay_rate
decay_round = args.decay_round
car_meet_p = args.car_meet_p
alpha_time = args.alpha_time
alpha = args.alpha
distribution = args.distribution
aggregation_metric = args.aggregation_metric
cache_update_metric = args.cache_update_metric
batch_size = args.batch_size
sample_size = args.sample_size
hidden_size = args.hidden_size
num_round = args.num_round
speed = args.speed
communication_distance = args.communication_distance
# importance = args.importance
early_stop_round = args.early_stop_round
num_car = args.num_car
cache_size = args.cache_size
# parallel = args.parallel
augment = args.augment
shards_allocation = args.shards_allocation
County = args.County
kick_out = args.kick_out
# SEED = args.random_seed
suffix = args.suffix
special_notes = args.note
def write_info(write_dir):
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    with open(write_dir+'/configuration.txt','w') as file:
        file.write('Special suffix = '+str(suffix)+'\n')
        file.write('Special Notes: '+str(special_notes)+'\n')
        file.write('Task: '+str(task)+'\n')
        file.write('Start time: ' + str(date_time.strftime('%Y-%m-%d %H:%M:%S'))+'\n')
        file.write('Random Seed = '+str(Randomseed)+'\n')
        file.write('local_ep = '+str(local_ep)+'\n')
        file.write('lr = '+str(lr)+'\n')
        file.write('decay_round = '+str(decay_round)+'\n')
        file.write('alpha_time = '+str(alpha_time)+'\n')
        file.write('aggregation_metric = '+str(aggregation_metric)+'\n')
        file.write('cache_update_metric = '+str( cache_update_metric)+'\n')
        file.write('batch_size = '+str(batch_size)+'\n')
        file.write('hidden_size = '+str(hidden_size)+'\n')
        file.write('lr_factor = '+str(args.lr_factor)+'\n')
        file.write('lr_patience = '+str(args.lr_patience)+'\n')
        file.write('num_round = '+str(num_round)+'\n')
        file.write('num_car = '+str(num_car)+'\n')
        file.write('epoch time = '+str(args.epoch_time)+'\n')
        file.write('speed = '+str(speed)+'\n')
        file.write('communication_distance = '+str(communication_distance)+'\n')
        file.write('cache_size = '+str(cache_size)+'\n')
        # file.write('parallel = '+str(parallel)+'\n')
        file.write('shards_allocation = '+str(shards_allocation)+'\n')
        file.write('Aggregation weights = '+str(weights)+'\n')
        file.write('County = '+str(County)+'\n')
        file.write('kick_out = '+str(kick_out)+'\n')
        if distribution != 'by_area':
            file.write('alpha = '+str(alpha)+'\n')
            file.write('Data distribution among cars:\n')
            file.write(str(statistic_data)+'\n')
            file.write('Data similarity among cars:\n')
            file.write(str(data_similarity)+'\n')
            file.write('Data_points:\n'+str(data_points)+'\n')
        else:
            file.write('This is by area distribution, no statistic data distribution.'+'\n')
            file.write('Initial data size per car:'+str(initial_size)+'\n')
            file.write('Max data size per car:'+str(max_size)+'\n')
            file.write('Update fraction:'+str(update_fraction)+'\n')
        file.write('mixing table'+str(mixing_table)+'\n')
        file.write('mixing pair'+str(mixing_pair)+'\n')
    pair, area = generate_roadNet_pair_area_list(write_dir,num_car, num_round,communication_distance,args.epoch_time,speed,County,10,car_type_list)
    with open(write_dir+'/pair.txt','w') as file:
        for i in range(num_round):
            file.write('Round:'+str(i)+': \n')
            for j in range(args.epoch_time):
                file.write('Seconds:'+str(j)+': \n')
                file.write(str(pair[i*args.epoch_time+j])+'\n')
    with open(write_dir+'/area.txt','w') as file:
        for i in range(num_car):
            file.write('Car:'+str(i)+': ')
            file.write(str(area[i])+'\n')
    return pair, area
            

def adjust_learning_rate(optimizer, epoch, initial_lr, decay_rate):
    lr = initial_lr
    if epoch>0 and epoch % 100 == 0:
        lr = lr/10
    # lr = initial_lr / (1 + decay_rate * epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

def update_learning_rate(i,learning_rate):
    if i>0 and i % decay_round == 0:
        learning_rate = learning_rate/10
    return learning_rate

def change_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def final_test(model_list, acc_list,class_acc_list):
    for i in range(len(model_list)):
        acc, class_acc = test(model_list[i], test_loader,num_class)
        acc_list[i].append(acc)
        class_acc_list[i].append(class_acc)
    
def final_test_process(model,process_index, result_list = None):
#     acc = test(model, test_loader)
    result_list[process_index] = test(model, test_loader)
    # Signal that the process has completed
#     completion_flag[process_index] = True
    #completion_event.set()
    #return loss, acc
def Centralized_process(suffix_dir,train_loader,test_loader,num_round,local_ep):
    model = []
    test_acc = []
    class_acc_list = []
    loss = []
    optimizer = []
    scheduler = []
    learning_rate = lr
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_cfl'+suffix_dir
    write_info(model_dir)
    for i in range(num_car):
        model.append(copy.deepcopy(global_model))
        model[i].to(device)
        optimizer.append(optim.SGD(params=model[i].parameters(), lr=lr))
        scheduler.append(ReduceLROnPlateau(optimizer[i], mode='max', factor=args.lr_factor, patience=args.lr_patience, verbose=False))
        loss.append([])
    for i in range(num_round):
        print('This is the round:',i)
        for param_group in optimizer[0].param_groups:
            print(param_group['lr'])
        # if i %300== 0:
        #     learning_rate  = learning_rate/5
        #     for index in range(num_car):
        #         for param_group in optimizer[index].param_groups:
        #             param_group['lr'] = learning_rate
        with open(model_dir+'/log.txt','a') as file:
            file.write('This is the round:'+str(i)+'\n')
            for param_group in optimizer[0].param_groups:
                file.write('current lr is: '+str(param_group['lr'])+'\n')
        start_time = time.time()
        #carry out local training
        for index in range(num_car):
            normal_training_process(model[index],optimizer[index],train_loader[index],local_ep,loss[index])
        #upload_central and aggregate
        w = []
        for index in range(num_car):
            w.append(copy.deepcopy(model[index].state_dict()))
        avg_w = average_weights(w,np.array(weights))
        for index in range(num_car):
            model[index].load_state_dict(copy.deepcopy(avg_w))
            model[index].to(device)
        acc, class_acc = test(model[0], test_loader)
        test_acc.append(acc)
        class_acc_list.append(class_acc)
        print('test acc:',acc)
        print('class acc:', class_acc)
        end_time = time.time()
        print(f'{end_time-start_time} [sec] for this epoch')
        if use_lr_scheduler:
            for index in range(num_car):
                scheduler[index].step(acc)
        else:
            learning_rate = update_learning_rate(i, learning_rate)
            for index in range(num_car):
                change_learning_rate(optimizer[index], learning_rate)
        with open(model_dir+'/log.txt','a') as file:
            file.write(f'{end_time-start_time} [sec] for this epoch'+'\n')
            file.write('test acc:'+str(acc)+'\n')
            file.write('class acc:'+str(class_acc)+'\n')
        #test global model accuracy
        with open(model_dir+'/average_acc_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+suffix_dir+'.txt','a') as file:
            file.write(str(i)+':'+str(test_acc[-1])+'\n')
        #write a code to early stop, if test acc remain unchanged for 10 rounds, then stop and return
        if i>early_stop_round and (abs(np.array(test_acc[-early_stop_round:])-test_acc[-1])<1e-7).all():
            print('early stop at round:',i)
            with open(model_dir+'/log.txt','a') as file:
                file.write('early stop at round:'+str(i)+'\n')
            break
    return loss, test_acc , model_dir 
    
def Decentralized_process(suffix_dir,train_loader,test_loader,num_round,local_ep):
    model = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = []
    learning_rate = lr
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+suffix_dir
    pair,area = write_info(model_dir)
    if distribution == 'by_area':
        train_loader = []
        train_indices = []
        for i in range(num_car):
            train_indices.append([])
            train_loader.append([])
    for i in range(num_car):
        model.append(copy.deepcopy(global_model))
        model[i].to(device)
        optimizer.append(optim.SGD(params=model[i].parameters(), lr=learning_rate))
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    for i in range(num_round):
        print('######################################################################')
        print('This is the round:',i)
        # print('lr:',learning_rate)
        if distribution == 'by_area':
            if i == 0:
                for index in range(num_car):
                    train_indices[index] = initial_training_subset(train_dataset, area[index][i],initial_size)
                    train_loader[index] = get_dataloader_by_indices(train_dataset, train_indices[index], batch_size)
            else:
                for index in range(num_car):
                    train_indices[index] = update_training_subset(train_indices[index], train_dataset, area[index][i], max_size, update_fraction)
                    train_loader[index] = get_dataloader_by_indices(train_dataset, train_indices[index], batch_size)
        if i>0 and i % decay_round == 0:
            learning_rate = learning_rate/10
        for index in range(num_car):
            change_learning_rate(optimizer[index], learning_rate)
        for param_group in optimizer[0].param_groups:
            print(param_group['lr'])
        # if i %300== 0:
        #     learning_rate  = learning_rate/5
        #     for index in range(num_car):
        #         for param_group in optimizer[index].param_groups:
        #             param_group['lr'] = learning_rate
        with open(model_dir+'/log.txt','a') as file:
            file.write('This is the round:'+str(i)+'\n')
            for param_group in optimizer[0].param_groups:
                file.write('current lr is: '+str(param_group['lr'])+'\n')
        start_time = time.time()
        #carry out local training
        for index in range(num_car):
            normal_training_process(model[index],optimizer[index],train_loader[index],local_ep,loss[index])
        #print(model[a].state_dict()['fc4.bias'])
                    #meet with each other:
            #print(model[a].state_dict()['fc4.bias'])
        # sect1_time = time.time()
        # print('training for:'+str(sect1_time-start_time))
        # for index in range(num_car):
        #     model[index].load_state_dict(torch.load(model_dir+'/model_'+str(index)+'.pt'))
        #     model[index].to(device)
        
        model_before_aggregation = copy.deepcopy(model)
        # sect2_time = time.time()
        # print('model aggregation for:'+str(sect2_time-sect1_time))
        final_test(model_before_aggregation, acc_global_before_aggregation, class_acc_list_before_aggregation)
        # print(acc_global_before_aggregation[a][-1])
        # sect3_time = time.time()
        # print('test old model for:'+str(sect3_time-sect2_time))

            
        # do model aggregation
        for a,b in pair[i]: 
            weighted_average_process(model[a],model[b],np.array([weights[a],weights[b]]))
            model[a].to(device)
            model[b].to(device)

        end_time = time.time()
        # print('model aggregation for:'+str(end_time-sect3_time))
        
        final_test(model, acc_global, class_acc_list)
        # test_time = time.time()
        # print('test new model for:'+str(test_time-end_time))
        print(f'{end_time-start_time} [sec] for this epoch')
        print('Before/After aggregation acc:')
        for index in range(num_car):
            print('car:',index,'---------------------------------------------------------------')
            print(acc_global_before_aggregation[index][-1],acc_global[index][-1])
            print(class_acc_list_before_aggregation[index][-1])
            print(class_acc_list[index][-1])
            print('Local Cache model version:')
        print('----------------------------------------------------------------------')
        print('Average test acc:',np.average(acc_global,axis=0)[-1])
        print('pair:',pair[i])
        with open(model_dir+'/log.txt','a') as file:
            file.write(f'{end_time-start_time} [sec] for this epoch'+'\n')
            file.write('Before/After aggregation acc:'+'\n')
            for index in range(num_car):
                file.write('car:'+str(index)+'---------------------------------------------------------------'+'\n')
                file.write(str(acc_global_before_aggregation[index][-1])+','+str(acc_global[index][-1])+'\n')
                file.write(str(class_acc_list_before_aggregation[index][-1])+'\n')
                file.write(str(class_acc_list[index][-1])+'\n')
                file.write('Local Cache model version:'+'\n')
            file.write('----------------------------------------------------------------------'+'\n')
            file.write('Average test acc:'+str(np.average(acc_global,axis=0)[-1])+'\n')
        # for index in range(num_car):
        #     if acc_global[index][-1]<acc_global_before_aggregation[index][-1]: 
        #         model[index] = copy.deepcopy(model_before_aggregation[index])
        #         acc_global[index][-1] = acc_global_before_aggregation[index][-1]
        # for a,b in pair[i]:
        #     if acc_global[a][-1]<acc_global_before_aggregation[a][-1] and acc_global[b][-1]<acc_global_before_aggregation[b][-1]:
        #         model[b] = copy.deepcopy(model_before_aggregation[b])
        #         model[a] = copy.deepcopy(model_before_aggregation[a])
        #         acc_global[a][-1] = acc_global_before_aggregation[a][-1]
        #         acc_global[b][-1] = acc_global_before_aggregation[b][-1]
        
        with open(model_dir+'/average_acc_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+suffix_dir+'.txt','a') as file:
            file.write(str(i)+':'+str(np.average(acc_global,axis=0)[-1])+'\n')
        # print('Duration:')
        # print(duration)
        #write a code to early stop, if test acc remain unchanged for 10 rounds, then stop and return
        if i > early_stop_round and (abs(np.average(acc_global,axis=0)[-early_stop_round:]-np.average(acc_global,axis=0)[-1])<1e-7).all():
            print('early stop at round:',i)
            with open(model_dir+'/log.txt','a') as file:
                file.write('early stop at round:'+str(i)+'\n')
            break
    return loss,acc_global,class_acc_list, acc_local   , model_dir


def Decentralized_fair_process(suffix_dir,train_loader,test_loader,num_round,local_ep):
    model = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = []
    learning_rate = lr
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+suffix_dir
    pair,area = write_info(model_dir)
    for i in range(num_car):
        model.append(copy.deepcopy(global_model))
        model[i].to(device)
        optimizer.append(optim.SGD(params=model[i].parameters(), lr=learning_rate))
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    for i in range(num_round):
        print('######################################################################')
        print('This is the round:',i)
        # print('lr:',learning_rate)
        if i>0 and i % decay_round == 0:
            learning_rate = learning_rate/10
        for index in range(num_car):
            change_learning_rate(optimizer[index], learning_rate)
        for param_group in optimizer[0].param_groups:
            print(param_group['lr'])
        # if i %300== 0:
        #     learning_rate  = learning_rate/5
        #     for index in range(num_car):
        #         for param_group in optimizer[index].param_groups:
        #             param_group['lr'] = learning_rate
        with open(model_dir+'/log.txt','a') as file:
            file.write('This is the round:'+str(i)+'\n')
            for param_group in optimizer[0].param_groups:
                file.write('current lr is: '+str(param_group['lr'])+'\n')
        start_time = time.time()
        #carry out local training
        for index in range(num_car):
            normal_training_process(model[index],optimizer[index],train_loader[index],local_ep,loss[index])
        #print(model[a].state_dict()['fc4.bias'])
                    #meet with each other:
            #print(model[a].state_dict()['fc4.bias'])
        # sect1_time = time.time()
        # print('training for:'+str(sect1_time-start_time))
        # for index in range(num_car):
        #     model[index].load_state_dict(torch.load(model_dir+'/model_'+str(index)+'.pt'))
        #     model[index].to(device)
        
        model_before_aggregation = copy.deepcopy(model)
        # sect2_time = time.time()
        # print('model aggregation for:'+str(sect2_time-sect1_time))
        final_test(model_before_aggregation, acc_global_before_aggregation, class_acc_list_before_aggregation)
        # print(acc_global_before_aggregation[a][-1])
        # sect3_time = time.time()
        # print('test old model for:'+str(sect3_time-sect2_time))

            
        # do model aggregation
        temp_weights = copy.deepcopy(weights)
        for a,b in pair[i]: 
            weighted_average_process(model[a],model[b],np.array([temp_weights[a],temp_weights[b]]))
            temp_weights[a] = temp_weights[a] + temp_weights[b]
            temp_weights[b] = temp_weights[a]
            model[a].to(device)
            model[b].to(device)

        end_time = time.time()
        # print('model aggregation for:'+str(end_time-sect3_time))
        
        final_test(model, acc_global, class_acc_list)
        # test_time = time.time()
        # print('test new model for:'+str(test_time-end_time))
        print(f'{end_time-start_time} [sec] for this epoch')
        print('Before/After aggregation acc:')
        for index in range(num_car):
            print('car:',index,'---------------------------------------------------------------')
            print(acc_global_before_aggregation[index][-1],acc_global[index][-1])
            print(class_acc_list_before_aggregation[index][-1])
            print(class_acc_list[index][-1])
            print('Local Cache model version:')
        print('----------------------------------------------------------------------')
        print('Average test acc:',np.average(acc_global,axis=0)[-1])
        print('pair:',pair[i])
        with open(model_dir+'/log.txt','a') as file:
            file.write(f'{end_time-start_time} [sec] for this epoch'+'\n')
            file.write('Before/After aggregation acc:'+'\n')
            for index in range(num_car):
                file.write('car:'+str(index)+'---------------------------------------------------------------'+'\n')
                file.write(str(acc_global_before_aggregation[index][-1])+','+str(acc_global[index][-1])+'\n')
                file.write(str(class_acc_list_before_aggregation[index][-1])+'\n')
                file.write(str(class_acc_list[index][-1])+'\n')
                file.write('Local Cache model version:'+'\n')
            file.write('----------------------------------------------------------------------'+'\n')
            file.write('Average test acc:'+str(np.average(acc_global,axis=0)[-1])+'\n')
        # for index in range(num_car):
        #     if acc_global[index][-1]<acc_global_before_aggregation[index][-1]: 
        #         model[index] = copy.deepcopy(model_before_aggregation[index])
        #         acc_global[index][-1] = acc_global_before_aggregation[index][-1]
        # for a,b in pair[i]:
        #     if acc_global[a][-1]<acc_global_before_aggregation[a][-1] and acc_global[b][-1]<acc_global_before_aggregation[b][-1]:
        #         model[b] = copy.deepcopy(model_before_aggregation[b])
        #         model[a] = copy.deepcopy(model_before_aggregation[a])
        #         acc_global[a][-1] = acc_global_before_aggregation[a][-1]
        #         acc_global[b][-1] = acc_global_before_aggregation[b][-1]
        
        with open(model_dir+'/average_acc_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+suffix_dir+'.txt','a') as file:
            file.write(str(i)+':'+str(np.average(acc_global,axis=0)[-1])+'\n')
        # print('Duration:')
        # print(duration)
        #write a code to early stop, if test acc remain unchanged for 10 rounds, then stop and return
        if i > early_stop_round and (abs(np.average(acc_global,axis=0)[-early_stop_round:]-np.average(acc_global,axis=0)[-1])<1e-7).all():
            print('early stop at round:',i)
            with open(model_dir+'/log.txt','a') as file:
                file.write('early stop at round:'+str(i)+'\n')
            break
    return loss,acc_global,class_acc_list, acc_local   , model_dir
 
def Decentralized_Cache_one_process(suffix_dir,train_loader,test_loader,num_round,local_ep):
    model = []
    local_cache = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = []
    fresh_class_time_table = np.zeros([num_car,num_car])
    # current_model_time_table = np.zeros(num_car)
    # current_model_combination_table = np.eye(num_car)#statistic_data#
    current_class_test = np.zeros([num_car,10])
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir
    pair,area = write_info(model_dir)
    for i in range(num_car):
        local_cache.append({})
        model.append(copy.deepcopy(global_model))
        model[i].to(device)
        optimizer.append(optim.sgd(params=model[i].parameters(), lr=lr))
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    for i in range(num_round):
        # old_model = copy.deepcopy(model)
        print('######################################################################')
        print('This is the round:',i)
        start_time = time.time()
        #carry out local training
        for index in range(num_car):
            normal_training_process(model[index],optimizer[index],train_loader,local_ep,loss[index])
            fresh_class_time_table[index][index] = i
                #print(model[a].state_dict()['fc4.bias'])
                    #meet with each other:
            #print(model[a].state_dict()['fc4.bias'])
            
        # #update time table by decay parameter alpha
        # fresh_class_time_table = alpha_time*i + (1- alpha_time)*fresh_class_time_table
        #update fresh table
        
        # current_model_time_table = alpha_time*i+(1-alpha_time)*current_model_time_table
        # current_model_combination_table = alpha_combination*statistic_data + (1-alpha_combination)*current_model_combination_table## 
        # for index in range(num_car):
        #     model[index].load_state_dict(torch.load(model_dir+'/model_'+str(index)+'.pt'))
        #     model[index].to(device)
        model_before_aggregation = copy.deepcopy(model)
        final_test(model_before_aggregation, acc_global_before_aggregation, class_acc_list_before_aggregation)
        # print(acc_global_before_aggregation[a][-1])
        
        
        # #First put self model in own cache
        # for index in range(num_car):
        #     put_own_model_into_cache(local_cache, model,index,i)
        #information exchange: update trace, cache, diag_fisher_matrix
        
        for a,b in pair[i]: 
            # update_model_cache_fresh(local_cache, model,a,b,i, cache_size, fresh_class_time_table,metric)
            update_model_cache_only_one(local_cache, model[a],model[b], a,b,i,cache_size, kick_out)
            
        
        # do model aggregation
        print('Updated/aggregated model time/combination:')
        # print(current_model_time_table)
        # print(current_model_combination_table)
        # old_model_time_table  = copy.deepcopy(current_model_time_table)
        # old_model_combination_table = copy.deepcopy(current_model_combination_table)
        for index in range(num_car):
            # model[index], fresh_class_time_table[index] = cache_average_process_fresh(model[index],local_cache[index], fresh_class_time_table[index],metric)
            model[index] = cache_average_process(model[index],local_cache[index])
            model[index].to(device)
            # for key in local_cache[index]:
            #     duration[index][key]+=1
        #     if data_similarity[a][b]<0.2:# and np.average([acc_global_before_aggregation[a][-1],acc_global_before_aggregation[b][-1]])<0.5:
        #         model[a],model[b] = exchange_process(model[a],model[b])
        #     else:
        #         model[a],model[b] = average_process(model[a],model[b])
        
        
        # update model property
        
        # print(current_model_combination_table)
                #print('doing aggregate')
                #print(model[a].state_dict()['fc4.bias'])
    #final test acc:
        
        final_test(model, acc_global, class_acc_list)
        end_time = time.time()
        for index in range(num_car):
            current_class_test[index] = class_acc_list[index][-1]
        # print(current_model_combination_table)
        # print(current_class_test)
        print(fresh_class_time_table)
        for index in range(num_car):
            print(class_acc_list[index][-1])
        print(f'{end_time-start_time} [sec] for this epoch')
        print('Before/After aggregation acc:')
        for index in range(num_car):
            print('car:',index,'---------------------------------------------------------------')
            print(acc_global_before_aggregation[index][-1],acc_global[index][-1])
            print(class_acc_list_before_aggregation[index][-1])
            print(class_acc_list[index][-1])
            print('Local Cache model version:')
            for key in local_cache[index]:
                print(key,local_cache[index][key]['time'])#,local_cache[index][key]['fresh_metric'])
        # for index in range(num_car):
        #     if acc_global[index][-1]<acc_global_before_aggregation[index][-1]: 
        #         model[index] = copy.deepcopy(model_before_aggregation[index])
        #         acc_global[index][-1] = acc_global_before_aggregation[index][-1]
        # for a,b in pair[i]:
        #     if acc_global[a][-1]<acc_global_before_aggregation[a][-1] and acc_global[b][-1]<acc_global_before_aggregation[b][-1]:
        #         model[b] = copy.deepcopy(model_before_aggregation[b])
        #         model[a] = copy.deepcopy(model_before_aggregation[a])
        #         acc_global[a][-1] = acc_global_before_aggregation[a][-1]
        #         acc_global[b][-1] = acc_global_before_aggregation[b][-1]
        print('----------------------------------------------------------------------')
        print('Average test acc:',np.average(acc_global,axis=0)[-1])
        print('pair:',pair[i])
        with open(model_dir+'/log.txt','a') as file:
            file.write('fresh_class_time_table\n')
            file.write(str(fresh_class_time_table)+'\n')
            for index in range(num_car):
                file.write(str(index)+':'+str(class_acc_list[index][-1])+'\n')
            file.write(f'{end_time-start_time} [sec] for this epoch'+'\n')
            file.write('Before/After aggregation acc:'+'\n')
            for index in range(num_car):
                file.write('car:'+str(index)+'---------------------------------------------------------------'+'\n')
                file.write(str(acc_global_before_aggregation[index][-1])+','+str(acc_global[index][-1])+'\n')
                file.write(str(class_acc_list_before_aggregation[index][-1])+'\n')
                file.write(str(class_acc_list[index][-1])+'\n')
                file.write('Local Cache model version:'+'\n')
                for key in local_cache[index]:
                    file.write(str(key)+':'+str(local_cache[index][key]['time'])+'\n')#,local_cache[index][key]['fresh_metric'])
            file.write('----------------------------------------------------------------------'+'\n')
            file.write('Average test acc:'+str(np.average(acc_global,axis=0)[-1])+'\n')
        #write a code to early stop, if test acc remain unchanged for 10 rounds, then stop and return
        if i>early_stop_round and (abs(np.average(acc_global,axis=0)[-early_stop_round:]-np.average(acc_global,axis=0)[-1])<1e-7).all():
            print('early stop at round:',i)
            with open(model_dir+'/log.txt','a') as file:
                file.write('early stop at round:'+str(i)+'\n')
            break
        # print('Duration:')
        # print(duration)
    return loss,acc_global,class_acc_list, acc_local   

def Decentralized_Cache_all_process(suffix_dir,train_loader,test_loader,num_round,local_ep):
    model = []
    local_cache = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = []
    learning_rate = lr
    fresh_class_time_table = np.zeros([num_car,num_car])
    # current_model_time_table = np.zeros(num_car)
    # current_model_combination_table = np.eye(num_car)#statistic_data#
    current_class_test = np.zeros([num_car,10])
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir
    pair,area = write_info(model_dir)
    for i in range(num_car):
        local_cache.append({})
        model.append(copy.deepcopy(global_model))
        model[i].to(device)
        optimizer.append(optim.SGD(params=model[i].parameters(), lr=lr))
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    for i in range(num_round):
        # old_model = copy.deepcopy(model)
        print('######################################################################')
        print('This is the round:',i)
        if i>0 and i % decay_round == 0:
            learning_rate = learning_rate/10
        for index in range(num_car):
            change_learning_rate(optimizer[index], learning_rate)
        for param_group in optimizer[0].param_groups:
            print(param_group['lr'])
        # if i %300== 0:
        #     learning_rate  = learning_rate/5
        #     for index in range(num_car):
        #         for param_group in optimizer[index].param_groups:
        #             param_group['lr'] = learning_rate
        with open(model_dir+'/log.txt','a') as file:
            file.write('This is the round:'+str(i)+'\n')
            for param_group in optimizer[0].param_groups:
                file.write('current lr is: '+str(param_group['lr'])+'\n')
        start_time = time.time()
        #carry out local training
        for index in range(num_car):
            # normal_training_process(model[index],optimizer[index],train_loader,local_ep,loss[index])
            normal_training_process(model[index],optimizer[index],train_loader[index],local_ep,loss[index])
            fresh_class_time_table[index][index] = i

        model_before_training = copy.deepcopy(model)
        for seconds in range(args.epoch_time):
            for a,b in pair[i*args.epoch_time+seconds]: 
                update_model_cache(local_cache, model_before_training[a],model_before_training[b],a,b,i, cache_size, kick_out)
                # update_model_cache(local_cache, model_before_training[a],model_before_training[b],a,b,i, cache_size, kick_out)
        
        # do model aggregation
        print('Updated/aggregated model time/combination:')
        for index in range(num_car):
            # model[index] = cache_average_process(model[index],index,local_cache[index],weights)
            model[index] = cache_average_process(model[index],index,i,local_cache[index],weights)
            model[index].to(device)

        final_test(model, acc_global, class_acc_list)

        end_time = time.time()


        for index in range(num_car):
            current_class_test[index] = class_acc_list[index][-1]
        # print(current_model_combination_table)
        # print(current_class_test)
        print(fresh_class_time_table)
        for index in range(num_car):
            print(class_acc_list[index][-1])
        print(f'{end_time-start_time} [sec] for this epoch')
        print('Before/After aggregation acc:')
        for index in range(num_car):
            print('car:',index,'---------------------------------------------------------------')
            print(acc_global[index][-1])
            # print(class_acc_list_before_aggregation[index][-1])
            print(class_acc_list[index][-1])
            print('Local Cache model version:')
            for key in local_cache[index]:
                print(key,local_cache[index][key]['time'])#,local_cache[index][key]['fresh_metric'])
        print('----------------------------------------------------------------------')
        print('Average test acc:',np.average(acc_global,axis=0)[-1])
        print('pair:',pair[i])
        with open(model_dir+'/log.txt','a') as file:
            file.write('fresh_class_time_table\n')
            file.write(str(fresh_class_time_table)+'\n')
            for index in range(num_car):
                file.write(str(index)+':'+str(class_acc_list[index][-1])+'\n')
            file.write(f'{end_time-start_time} [sec] for this epoch'+'\n')
            file.write('Before/After aggregation acc:'+'\n')
            for index in range(num_car):
                file.write('car:'+str(index)+'---------------------------------------------------------------'+'\n')
                file.write(str(acc_global[index][-1])+'\n')
                # file.write(str(class_acc_list_before_aggregation[index][-1])+'\n')
                file.write(str(class_acc_list[index][-1])+'\n')
                file.write('Local Cache model version:'+'\n')
                for key in local_cache[index]:
                    file.write(str(key)+':'+str(local_cache[index][key]['time'])+'\n')#,local_cache[index][key]['fresh_metric'])
            file.write('----------------------------------------------------------------------'+'\n')
            file.write('Average test acc:'+str(np.average(acc_global,axis=0)[-1])+'\n')
        # for index in range(num_car):
        #     if acc_global[index][-1]<acc_global_before_aggregation[index][-1]: 
        #         model[index] = copy.deepcopy(model_before_aggregation[index])
        #         acc_global[index][-1] = acc_global_before_aggregation[index][-1]
        # for a,b in pair[i]:
        #     if acc_global[a][-1]<acc_global_before_aggregation[a][-1] and acc_global[b][-1]<acc_global_before_aggregation[b][-1]:
        #         model[b] = copy.deepcopy(model_before_aggregation[b])
        #         model[a] = copy.deepcopy(model_before_aggregation[a])
        #         acc_global[a][-1] = acc_global_before_aggregation[a][-1]
        #         acc_global[b][-1] = acc_global_before_aggregation[b][-1]
        
        with open(model_dir+'/average_acc_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir+'.txt','a') as file:
            file.write(str(i)+':'+str(np.average(acc_global,axis=0)[-1])+'\n')
        #write a code to early stop, if test acc remain unchanged for 10 rounds, then stop and return
        if i>early_stop_round and (abs(np.average(acc_global,axis=0)[-early_stop_round:]-np.average(acc_global,axis=0)[-1])<1e-7).all():
            print('early stop at round:',i)
            with open(model_dir+'/log.txt','a') as file:
                file.write('early stop at round:'+str(i)+'\n')
            break
        
        
        
     
    return loss,acc_global,class_acc_list, acc_local, model_dir 


def Decentralized_Cache_data_distribution_process(suffix_dir,train_loader,test_loader,num_round,local_ep):
    model = []
    local_cache = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = []
    learning_rate = lr
    fresh_class_time_table = np.zeros([num_car,num_car])
    # current_model_time_table = np.zeros(num_car)
    # current_model_combination_table = np.eye(num_car)#statistic_data#
    current_class_test = np.zeros([num_car,10])
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir
    pair,area = write_info(model_dir)
    for i in range(num_car):
        local_cache.append({})
        model.append(copy.deepcopy(global_model))
        model[i].to(device)
        optimizer.append(optim.SGD(params=model[i].parameters(), lr=lr))
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    for i in range(num_round):
        # old_model = copy.deepcopy(model)
        print('######################################################################')
        print('This is the round:',i)
        if i>0 and i % decay_round == 0:
            learning_rate = learning_rate/10
        for index in range(num_car):
            change_learning_rate(optimizer[index], learning_rate)
        for param_group in optimizer[0].param_groups:
            print(param_group['lr'])
        # if i %300== 0:
        #     learning_rate  = learning_rate/5
        #     for index in range(num_car):
        #         for param_group in optimizer[index].param_groups:
        #             param_group['lr'] = learning_rate
        with open(model_dir+'/log.txt','a') as file:
            file.write('This is the round:'+str(i)+'\n')
            for param_group in optimizer[0].param_groups:
                file.write('current lr is: '+str(param_group['lr'])+'\n')
        start_time = time.time()
        #carry out local training
        for index in range(num_car):
            normal_training_process(model[index],optimizer[index],train_loader,local_ep,loss[index])
            fresh_class_time_table[index][index] = i

        model_before_training = copy.deepcopy(model)
        for seconds in range(args.epoch_time):
            for a,b in pair[i*args.epoch_time+seconds]: 
                # update_model_cache(local_cache, model_before_training[a],model_before_training[b],a,b,i, cache_size, kick_out)
                update_model_cache_distribution(local_cache, model_before_training[a],model_before_training[b],a,b,i, cache_size, args.kick_out,statistic_data,0.9)
                
        for key in local_cache[0]:
            print(key,local_cache[0][key]['time'],local_cache[0][key]['distribution'],local_cache[0][key]['cache_score'])
        # do model aggregation
        print('Updated/aggregated model time/combination:')
        for index in range(num_car):
            model[index] = cache_average_process(model[index],index,local_cache[index],weights)
            model[index].to(device)

        final_test(model, acc_global, class_acc_list)

        end_time = time.time()


        for index in range(num_car):
            current_class_test[index] = class_acc_list[index][-1]
        # print(current_model_combination_table)
        # print(current_class_test)
        print(fresh_class_time_table)
        for index in range(num_car):
            print(class_acc_list[index][-1])
        print(f'{end_time-start_time} [sec] for this epoch')
        print('Before/After aggregation acc:')
        for index in range(num_car):
            print('car:',index,'---------------------------------------------------------------')
            print(acc_global_before_aggregation[index][-1],acc_global[index][-1])
            print(class_acc_list_before_aggregation[index][-1])
            print(class_acc_list[index][-1])
            print('Local Cache model version:')
            for key in local_cache[index]:
                print(key,local_cache[index][key]['time'])#,local_cache[index][key]['fresh_metric'])
        print('----------------------------------------------------------------------')
        print('Average test acc:',np.average(acc_global,axis=0)[-1])
        print('pair:',pair[i])
        with open(model_dir+'/log.txt','a') as file:
            file.write('fresh_class_time_table\n')
            file.write(str(fresh_class_time_table)+'\n')
            for index in range(num_car):
                file.write(str(index)+':'+str(class_acc_list[index][-1])+'\n')
            file.write(f'{end_time-start_time} [sec] for this epoch'+'\n')
            file.write('Before/After aggregation acc:'+'\n')
            for index in range(num_car):
                file.write('car:'+str(index)+'---------------------------------------------------------------'+'\n')
                file.write(str(acc_global_before_aggregation[index][-1])+','+str(acc_global[index][-1])+'\n')
                file.write(str(class_acc_list_before_aggregation[index][-1])+'\n')
                file.write(str(class_acc_list[index][-1])+'\n')
                file.write('Local Cache model version:'+'\n')
                for key in local_cache[index]:
                    file.write(str(key)+':'+str(local_cache[index][key]['time'])+'\n')#,local_cache[index][key]['fresh_metric'])
            file.write('----------------------------------------------------------------------'+'\n')
            file.write('Average test acc:'+str(np.average(acc_global,axis=0)[-1])+'\n')
        # for index in range(num_car):
        #     if acc_global[index][-1]<acc_global_before_aggregation[index][-1]: 
        #         model[index] = copy.deepcopy(model_before_aggregation[index])
        #         acc_global[index][-1] = acc_global_before_aggregation[index][-1]
        # for a,b in pair[i]:
        #     if acc_global[a][-1]<acc_global_before_aggregation[a][-1] and acc_global[b][-1]<acc_global_before_aggregation[b][-1]:
        #         model[b] = copy.deepcopy(model_before_aggregation[b])
        #         model[a] = copy.deepcopy(model_before_aggregation[a])
        #         acc_global[a][-1] = acc_global_before_aggregation[a][-1]
        #         acc_global[b][-1] = acc_global_before_aggregation[b][-1]
        
        with open(model_dir+'/average_acc_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir+'.txt','a') as file:
            file.write(str(i)+':'+str(np.average(acc_global,axis=0)[-1])+'\n')
        #write a code to early stop, if test acc remain unchanged for 10 rounds, then stop and return
        if i>early_stop_round and (abs(np.average(acc_global,axis=0)[-early_stop_round:]-np.average(acc_global,axis=0)[-1])<1e-7).all():
            print('early stop at round:',i)
            with open(model_dir+'/log.txt','a') as file:
                file.write('early stop at round:'+str(i)+'\n')
            break
        
        
        
     
    return loss,acc_global,class_acc_list, acc_local, model_dir 

def Decentralized_Cache_global_process(suffix_dir,train_loader,test_loader,num_round,local_ep):
    model = []
    local_cache = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = []
    learning_rate = lr
    fresh_class_time_table = np.zeros([num_car,num_car])
    # current_model_time_table = np.zeros(num_car)
    # current_model_combination_table = np.eye(num_car)#statistic_data#
    current_class_test = np.zeros([num_car,10])
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir
    pair,area = write_info(model_dir)
    for i in range(num_car):
        local_cache.append({})
        model.append(copy.deepcopy(global_model))
        model[i].to(device)
        optimizer.append(optim.SGD(params=model[i].parameters(), lr=lr))
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    for i in range(num_round):
        # old_model = copy.deepcopy(model)
        print('######################################################################')
        print('This is the round:',i)
        if i>0 and i % decay_round == 0:
            learning_rate = learning_rate/10
        for index in range(num_car):
            change_learning_rate(optimizer[index], learning_rate)
        for param_group in optimizer[0].param_groups:
            print(param_group['lr'])
        # if i %300== 0:
        #     learning_rate  = learning_rate/5
        #     for index in range(num_car):
        #         for param_group in optimizer[index].param_groups:
        #             param_group['lr'] = learning_rate
        with open(model_dir+'/log.txt','a') as file:
            file.write('This is the round:'+str(i)+'\n')
            for param_group in optimizer[0].param_groups:
                file.write('current lr is: '+str(param_group['lr'])+'\n')
        start_time = time.time()
        #carry out local training
        for index in range(num_car):
            normal_training_process(model[index],optimizer[index],train_loader,local_ep,loss[index])
            fresh_class_time_table[index][index] = i
                #print(model[a].state_dict()['fc4.bias'])
                    #meet with each other:
            #print(model[a].state_dict()['fc4.bias'])
            
        # #update time table by decay parameter alpha
        # fresh_class_time_table = alpha_time*i + (1- alpha_time)*fresh_class_time_table
        #update fresh table
        
        # current_model_time_table = alpha_time*i+(1-alpha_time)*current_model_time_table
        # current_model_combination_table = alpha_combination*statistic_data + (1-alpha_combination)*current_model_combination_table## 
        # for index in range(num_car):
        #     model[index].load_state_dict(torch.load(model_dir+'/model_'+str(index)+'.pt'))
        #     model[index].to(device)
        model_before_aggregation = copy.deepcopy(model)
        final_test(model_before_aggregation, acc_global_before_aggregation, class_acc_list_before_aggregation)
        # print(acc_global_before_aggregation[a][-1])
        
        
        # #First put self model in own cache
        # for index in range(num_car):
        #     put_own_model_into_cache(local_cache, model,index,i)
        #information exchange: update trace, cache, diag_fisher_matrix
        
        for a,b in pair[i]: 
            # update_model_cache_fresh(local_cache, model,a,b,i, cache_size, fresh_class_time_table,metric)
            update_model_cache(local_cache,  model[a],model[b],a,b,i, cache_size, kick_out)
            # update_model_cache_only_one(local_cache,model, a,b,i,cache_size)
        if kick_out == True:
            for index in range(num_car):
                local_cache[index] = kick_out_timeout_model(local_cache[index],i-cache_size)
        cache_info = np.zeros([cache_size,num_car])
        for index in range(num_car):
            for key in local_cache[index]:
                cache_info[i-local_cache[index][key]['time']][key] += 1 
        print('cache_info:',cache_info)
        
        # do model aggregation
        print('Updated/aggregated model time/combination:')
        # print(current_model_time_table)
        # print(current_model_combination_table)
        # old_model_time_table  = copy.deepcopy(current_model_time_table)
        # old_model_combination_table = copy.deepcopy(current_model_combination_table)
        for index in range(num_car):
            # model[index], fresh_class_time_table[index] = cache_average_process_fresh(model[index],local_cache[index], fresh_class_time_table[index],metric)
            model[index] = cache_average_process(model[index],index,local_cache[index],weights)
            model[index].to(device)
            # for key in local_cache[index]:
            #     duration[index][key]+=1
        #     if data_similarity[a][b]<0.2:# and np.average([acc_global_before_aggregation[a][-1],acc_global_before_aggregation[b][-1]])<0.5:
        #         model[a],model[b] = exchange_process(model[a],model[b])
        #     else:
        #         model[a],model[b] = average_process(model[a],model[b])
        
        
        
        # update model property
        
        # print(current_model_combination_table)
                #print('doing aggregate')
                #print(model[a].state_dict()['fc4.bias'])
        #final test acc:
        # final_test(model, acc_global, class_acc_list)

        end_time = time.time()


        for index in range(num_car):
            current_class_test[index] = class_acc_list[index][-1]
        # print(current_model_combination_table)
        # print(current_class_test)
        print(fresh_class_time_table)
        for index in range(num_car):
            print(class_acc_list[index][-1])
        print(f'{end_time-start_time} [sec] for this epoch')
        print('Before/After aggregation acc:')
        for index in range(num_car):
            print('car:',index,'---------------------------------------------------------------')
            print(acc_global_before_aggregation[index][-1],acc_global[index][-1])
            print(class_acc_list_before_aggregation[index][-1])
            print(class_acc_list[index][-1])
            print('Local Cache model version:')
            for key in local_cache[index]:
                print(key,local_cache[index][key]['time'])#,local_cache[index][key]['fresh_metric'])
        print('----------------------------------------------------------------------')
        print('Average test acc:',np.average(acc_global,axis=0)[-1])
        print('pair:',pair[i])
        with open(model_dir+'/log.txt','a') as file:
            file.write('fresh_class_time_table\n')
            file.write(str(fresh_class_time_table)+'\n')
            for index in range(num_car):
                file.write(str(index)+':'+str(class_acc_list[index][-1])+'\n')
            file.write(f'{end_time-start_time} [sec] for this epoch'+'\n')
            file.write('Before/After aggregation acc:'+'\n')
            for index in range(num_car):
                file.write('car:'+str(index)+'---------------------------------------------------------------'+'\n')
                file.write(str(acc_global_before_aggregation[index][-1])+','+str(acc_global[index][-1])+'\n')
                file.write(str(class_acc_list_before_aggregation[index][-1])+'\n')
                file.write(str(class_acc_list[index][-1])+'\n')
                file.write('Local Cache model version:'+'\n')
                for key in local_cache[index]:
                    file.write(str(key)+':'+str(local_cache[index][key]['time'])+'\n')#,local_cache[index][key]['fresh_metric'])
            file.write('----------------------------------------------------------------------'+'\n')
            file.write('Average test acc:'+str(np.average(acc_global,axis=0)[-1])+'\n')
        # for index in range(num_car):
        #     if acc_global[index][-1]<acc_global_before_aggregation[index][-1]: 
        #         model[index] = copy.deepcopy(model_before_aggregation[index])
        #         acc_global[index][-1] = acc_global_before_aggregation[index][-1]
        # for a,b in pair[i]:
        #     if acc_global[a][-1]<acc_global_before_aggregation[a][-1] and acc_global[b][-1]<acc_global_before_aggregation[b][-1]:
        #         model[b] = copy.deepcopy(model_before_aggregation[b])
        #         model[a] = copy.deepcopy(model_before_aggregation[a])
        #         acc_global[a][-1] = acc_global_before_aggregation[a][-1]
        #         acc_global[b][-1] = acc_global_before_aggregation[b][-1]
        
        with open(model_dir+'/average_acc_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir+'.txt','a') as file:
            file.write(str(i)+':'+str(np.average(acc_global,axis=0)[-1])+'\n')
        #write a code to early stop, if test acc remain unchanged for 10 rounds, then stop and return
        if i>early_stop_round and (abs(np.average(acc_global,axis=0)[-early_stop_round:]-np.average(acc_global,axis=0)[-1])<1e-7).all():
            print('early stop at round:',i)
            with open(model_dir+'/log.txt','a') as file:
                file.write('early stop at round:'+str(i)+'\n')
            break
        
        
        
     
    return loss,acc_global,class_acc_list, acc_local, model_dir 

def Decentralized_Cache_test_taxi():
    model = []
    local_cache = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = []
    # fresh_class_time_table = np.zeros([num_car,num_car])
    # fresh_history = np.zeros([num_car,num_round])
    # current_model_time_table = np.zeros(num_car)
    # current_model_combination_table = np.eye(num_car)#statistic_data#
    # current_class_test = np.zeros([num_car,10])
    model_dir = './result/test/taxi'
    pair,area = write_info(model_dir)

    for i in range(num_car):
        local_cache.append({})
        model.append(copy.deepcopy(global_model))
        optimizer.append(optim.SGD(params=model[i].parameters(), lr=lr))
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    cache_info_dynamic = np.ones([num_car])
    for i in range(num_round):
        # old_model = copy.deepcopy(model)
        # print('######################################################################')
        print('This is the round:',i)
        for ep in range(local_ep):
            start_time = time.time()
            #carry out local training
            # for index in range(num_car):
                    # normal_training_process(model[index],optimizer[index],train_loader,test_loader,local_ep,index,loss[index],acc_global[index],acc_local[index],model_dir)
                    # fresh_class_time_table[index][index] = i
                    #print(model[a].state_dict()['fc4.bias'])
                        #meet with each other:
                #print(model[a].state_dict()['fc4.bias'])
                
            # #update time table by decay parameter alpha
            # fresh_class_time_table = alpha_time*i + (1- alpha_time)*fresh_class_time_table
            #update fresh table
            
            # current_model_time_table = alpha_time*i+(1-alpha_time)*current_model_time_table
            # current_model_combination_table = alpha_combination*statistic_data + (1-alpha_combination)*current_model_combination_table## 
            # for index in range(num_car):
                # model[index].load_state_dict(torch.load(model_dir+'/model_'+str(index)+'.pt'))
            # model_before_aggregation = copy.deepcopy(model)
            # final_test(model_before_aggregation, acc_global_before_aggregation, class_acc_list_before_aggregation)
            # print(acc_global_before_aggregation[a][-1])
            
            
            # #First put self model in own cache
            # for index in range(num_car):
            #     put_own_model_into_cache(local_cache, model,index,i)
            #information exchange: update trace, cache, diag_fisher_matrix
            
            model_before_training = copy.deepcopy(model)
            if kick_out == True:
                for index in range(num_car):
                    local_cache[index] = kick_out_timeout_model(local_cache[index],i-args.kick_out)
                    
            for seconds in range(args.epoch_time):
                for a,b in pair[i*args.epoch_time+seconds]: 
                    if car_type_list[a] == car_type_list[b] or car_type_list[a] == 0 or car_type_list[b] == 0: 
                        update_model_cache(local_cache, model_before_training[a],model_before_training[b],a,b,i, cache_size, kick_out)
                    # if car_type_list[a] == car_type_list[b]: 
                    #     if car_type_list[a] != 0:
                    #         update_model_cache_car_to_car_p(local_cache, model_before_training[a],model_before_training[b],a,b,i, cache_size, kick_out,car_type_list, type_limits_car)
                    #     else:
                    #         update_model_cache_taxi_to_taxi_p(local_cache, a,b,i, cache_size, kick_out,car_type_list,type_limits_taxi)
                    # elif car_type_list[a] == 0:
                    #     update_model_cache_car_to_taxi_p(local_cache, model_before_training[b],b,a,i, cache_size, cache_size, kick_out,car_type_list,type_limits_car,type_limits_taxi)
                    # elif car_type_list[b] == 0:
                    #     update_model_cache_car_to_taxi_p(local_cache, model_before_training[a],a,b,i, cache_size, cache_size, kick_out,car_type_list,type_limits_car,type_limits_taxi)
                    # update_model_cache_distribution(local_cache, model_before_training[a],model_before_training[b],a,b,i, cache_size, args.kick_out,statistic_data,max_std,0.9)
            # for key in local_cache[0]:
            #     print(key,local_cache[0][key]['time'],local_cache[0][key]['distribution'],local_cache[0][key]['cache_score'])
            cache_info = np.zeros([num_car])
            for index in range(num_car):
                # cache_info_by_time[0][index] += 1 
                cache_info[index] += 1
                for key in local_cache[index]:
                    # print(local_cache[index][key]['time'])
                    cache_info[key] += 1
            cache_age = 0
            cache_num = 0
            for index in range(num_car):
                cache_num += len(local_cache[index])
                for key in local_cache[index]:
                    cache_age += i-local_cache[index][key]['time']
            avg_cache_age = cache_age/cache_num
            with open(model_dir+'/cache_age_cache_num_'+str('cache_all' )+'_'+str(cache_size)+'_'+str(args.epoch_time)+'_'+str(args.kick_out)+'.txt','a') as file:
                file.write(str(i)+':')
                file.write(str(avg_cache_age)+'\t'+str(cache_num/num_car)+'\n')
            # with open(model_dir+'/cache_info.txt','a') as file:
            #     file.write('This is the round:'+str(i)+'\n')
            #     file.write(str(cache_info)+'\n')


            ################### code to test cache info
            # if balance == True:
            #     for a,b in pair[i]: 
            #         cache_info_dynamic = update_model_cache_global(local_cache,  model[a],model[b],a,b,i,cache_size, cache_info_dynamic,kick_out)
                    
            #     if kick_out == True:
            #         for index in range(num_car):
            #             local_cache[index],cache_info_dynamic = kick_out_timeout_model_cache_info(local_cache[index],i-cache_size, cache_info_dynamic)
            #     with open(model_dir+'/cache_info_dynamic.txt','a') as file:
            #         file.write('This is the round:'+str(i)+'\n')
            #         file.write(str(cache_info_dynamic)+'\n')
            # else:
            #     for a,b in pair[i]: 
            #         # update_model_cache_fresh(local_cache, model,a,b,i, cache_size, fresh_class_time_table,metric)
            #         update_model_cache(local_cache,  model[a],model[b],a,b,i, cache_size, kick_out)
            #         # update_model_cache_only_one(local_cache,model, a,b,i,cache_size)
            #     if kick_out == True:
            #         for index in range(num_car):
            #             local_cache[index] = kick_out_timeout_model(local_cache[index],i-cache_size)


        
            #     cache_info = np.zeros([num_car])
            #     for index in range(num_car):
            #         # cache_info_by_time[0][index] += 1 
            #         cache_info[index] += 1
            #         for key in local_cache[index]:
            #             # print(local_cache[index][key]['time'])
            #             cache_info[key] += 1
            #             # cache_info_by_time[i-local_cache[index][key]['time']][key] += 1 
            #     # print('cache_info:',cache_info)
            #     with open(model_dir+'/cache_info.txt','a') as file:
            #         file.write('This is the round:'+str(i)+'\n')
            #         file.write(str(cache_info)+'\n')
            
        
        # with open(model_dir+'/cache_info_by_time.txt','a') as file:
        #     file.write('This is the round:'+str(i)+'\n')
        #     for info in cache_info_by_time:
        #         file.write(str(info)+'\n')
        # do model aggregation
        # print('Updated/aggregated model time/combination:')
        # for index in range(num_car):
        #     fresh_class_time_table[index] = cache_average_process_fresh_without_model(local_cache[index], fresh_class_time_table[index],aggregation_metric)
        end_time = time.time()
        
        # update model property
        
        # print(current_model_combination_table)
                #print('doing aggregate')
                #print(model[a].state_dict()['fc4.bias'])
    #final test acc:
        
        # final_test(model, acc_global, class_acc_list)
        # for index in range(num_car):
        #     current_class_test[index] = class_acc_list[index][-1]
        # print(current_model_combination_table)
        # print(current_class_test)
        # print(fresh_class_time_table)
        # for index in range(num_car):
        #     print(class_acc_list[index][-1])
        # print(f'{end_time-start_time} [sec] for this epoch')
        # print('Before/After aggregation acc:')
        # for index in range(num_car):
        #     print('car:',index,'---------------------------------------------------------------')
        #     # print(acc_global_before_aggregation[index][-1],acc_global[index][-1])
        #     # print(class_acc_list_before_aggregation[index][-1])
        #     # print(class_acc_list[index][-1])
        #     print('Local Cache model version:')
        #     for key in local_cache[index]:
        #         print(key,local_cache[index][key]['time'])
                
        # for index in range(num_car):
        #     if acc_global[index][-1]<acc_global_before_aggregation[index][-1]: 
        #         model[index] = copy.deepcopy(model_before_aggregation[index])
        #         acc_global[index][-1] = acc_global_before_aggregation[index][-1]
        # for a,b in pair[i]:
        #     if acc_global[a][-1]<acc_global_before_aggregation[a][-1] and acc_global[b][-1]<acc_global_before_aggregation[b][-1]:
        #         model[b] = copy.deepcopy(model_before_aggregation[b])
        #         model[a] = copy.deepcopy(model_before_aggregation[a])
        #         acc_global[a][-1] = acc_global_before_aggregation[a][-1]
        #         acc_global[b][-1] = acc_global_before_aggregation[b][-1]
        # print('----------------------------------------------------------------------')
        # print('Average test acc:',np.average(acc_global,axis=0)[-1])
        # print('pair:',pair[i])
        # print('Duration:')
        # print(duration)
    return loss,[0,0,0],class_acc_list, acc_local 

def Decentralized_Cache_test_taxi_priority():
    model = []
    local_cache = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = []
    model_dir = './result/test/taxi'
    pair,area = write_info(model_dir)

    for i in range(num_car):
        local_cache.append({})
        model.append(copy.deepcopy(global_model))
        optimizer.append(optim.SGD(params=model[i].parameters(), lr=lr))
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    cache_info_dynamic = np.ones([num_car])
    for i in range(num_round):
        # old_model = copy.deepcopy(model)
        # print('######################################################################')
        print('This is the round:',i)
        for ep in range(local_ep):

            model_before_training = copy.deepcopy(model)
            if kick_out == True:
                for index in range(num_car):
                    local_cache[index] = kick_out_timeout_model(local_cache[index],i-args.kick_out)
                    
            for seconds in range(args.epoch_time):
                for a,b in pair[i*args.epoch_time+seconds]: 
                    if car_type_list[a] == car_type_list[b] or car_type_list[a] == 0 or car_type_list[b] == 0: 
                        update_model_cache_car_to_car_p(local_cache, model_before_training[a],model_before_training[b],a,b,i, cache_size, kick_out,car_area_list, type_limits_car)
                    # if car_type_list[a] == car_type_list[b]: 
                    #     if car_type_list[a] != 0:
                    #         update_model_cache_car_to_car_p(local_cache, model_before_training[a],model_before_training[b],a,b,i, cache_size, kick_out,car_type_list, type_limits_car)
                    #     else:
                    #         update_model_cache_taxi_to_taxi_p(local_cache, a,b,i, cache_size, kick_out,car_type_list,type_limits_taxi)
                    # elif car_type_list[a] == 0:
                    #     update_model_cache_car_to_taxi_p(local_cache, model_before_training[b],b,a,i, cache_size, cache_size, kick_out,car_type_list,type_limits_car,type_limits_taxi)
                    # elif car_type_list[b] == 0:
                    #     update_model_cache_car_to_taxi_p(local_cache, model_before_training[a],a,b,i, cache_size, cache_size, kick_out,car_type_list,type_limits_car,type_limits_taxi)
                    # update_model_cache_distribution(local_cache, model_before_training[a],model_before_training[b],a,b,i, cache_size, args.kick_out,statistic_data,max_std,0.9)
            # for key in local_cache[0]:
            #     print(key,local_cache[0][key]['time'],local_cache[0][key]['distribution'],local_cache[0][key]['cache_score'])
            cache_info = np.zeros([num_car])
            for index in range(num_car):
                # cache_info_by_time[0][index] += 1 
                cache_info[index] += 1
                for key in local_cache[index]:
                    # print(local_cache[index][key]['time'])
                    cache_info[key] += 1
            cache_age = 0
            cache_num = 0
            for index in range(num_car):
                cache_num += len(local_cache[index])
                for key in local_cache[index]:
                    cache_age += i-local_cache[index][key]['time']
            avg_cache_age = cache_age/cache_num
            with open(model_dir+'/cache_age_cache_num_'+str('cache_all' )+'_'+str(cache_size)+'_'+str(args.epoch_time)+'_'+str(args.kick_out)+'.txt','a') as file:
                file.write(str(i)+':')
                file.write(str(avg_cache_age)+'\t'+str(cache_num/num_car)+'\n')
            with open(model_dir+'/log.txt','a') as file:
            # file.write('fresh_class_time_table\n')
            # file.write(str(fresh_class_time_table)+'\n')
                # for index in range(num_car):
                #     file.write(str(index)+':'+str(class_acc_list[index][-1])+'\n')
                # file.write(f'{end_time-start_time} [sec] for this epoch'+'\n')
                # file.write('Before/After aggregation acc:'+'\n')
                for index in range(num_car):
                    file.write('car:'+str(index)+'---------------------------------------------------------------'+'\n')
                    # file.write(str(acc_global[index][-1])+'\n')
                    # # file.write(str(acc_global_before_aggregation[index][-1])+','+str(acc_global[index][-1])+'\n')
                    # # file.write(str(class_acc_list_before_aggregation[index][-1])+'\n')
                    # file.write(str(class_acc_list[index][-1])+'\n')
                    file.write('Local Cache model version:'+'\n')
                    for key in local_cache[index]:
                        file.write(str(key)+':'+str(local_cache[index][key]['time'])+'\t'+str(local_cache[index][key]['car_type'])+'\n')#,local_cache[index][key]['fresh_metric'])

    return loss,[0,0,0],class_acc_list, acc_local 


def Decentralized_Cache_test():
    model = []
    local_cache = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = []
    model_dir = './result/test/cache_all'
    pair,area = write_info(model_dir)

    for i in range(num_car):
        local_cache.append({})
        model.append(copy.deepcopy(global_model))
        optimizer.append(optim.SGD(params=model[i].parameters(), lr=lr))
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    for i in range(10):
        local_cache.append({})
    cache_info_dynamic = np.ones([num_car])
    for i in range(num_round):
        # old_model = copy.deepcopy(model)
        # print('######################################################################')
        print('This is the round:',i)
        for ep in range(local_ep):
            start_time = time.time()
            
            model_before_training = copy.deepcopy(model)
            if kick_out == True:
                for index in range(num_car):
                    local_cache[index] = kick_out_timeout_model(local_cache[index],i-args.kick_out)
                    
            for seconds in range(args.epoch_time):
                for a,b in pair[i*args.epoch_time+seconds]: 
                    update_model_cache(local_cache, model_before_training[a],model_before_training[b],a,b,i, cache_size, kick_out)
            cache_info = np.zeros([num_car])
            for index in range(num_car):
                # cache_info_by_time[0][index] += 1 
                cache_info[index] += 1
                for key in local_cache[index]:
                    # print(local_cache[index][key]['time'])
                    cache_info[key] += 1
            cache_age = 0
            cache_num = 0
            for index in range(num_car):
                cache_num += len(local_cache[index])
                for key in local_cache[index]:
                    cache_age += i-local_cache[index][key]['time']
            avg_cache_age = cache_age/cache_num
            with open(model_dir+'/cache_age_cache_num_'+str('cache_all' )+'_'+str(cache_size)+'_'+str(args.epoch_time)+'_'+str(args.kick_out)+'.txt','a') as file:
                file.write(str(i)+':')
                file.write(str(avg_cache_age)+'\t'+str(cache_num/num_car)+'\n')
        end_time = time.time()

    return loss,[0,0,0],class_acc_list, acc_local 
def Decentralized_Cache_test_mixing():
    model = []
    local_cache = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = []

    model_dir = './result/test/mixing'
    pair,area = write_info(model_dir)

    for i in range(num_car):
        local_cache.append([])
        model.append(copy.deepcopy(global_model))
        optimizer.append(optim.SGD(params=model[i].parameters(), lr=lr))
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    cache_info_dynamic = np.ones([num_car])
    for i in range(num_round):
        # old_model = copy.deepcopy(model)
        # print('######################################################################')
        print('This is the round:',i)
        for ep in range(local_ep):
            start_time = time.time()
            
            model_before_training = copy.deepcopy(model)
            if kick_out == True:
                for index in range(num_car):
                    local_cache[index] = kick_out_timeout_model_list(local_cache[index],i-args.kick_out)
                    
            for seconds in range(args.epoch_time):
                for a,b in pair[i*args.epoch_time+seconds]: 
                    update_model_cache_mixing(local_cache, model_before_training[a],model_before_training[b],a,b,i, cache_size)
                    # update_model_cache_distribution(local_cache, model_before_training[a],model_before_training[b],a,b,i, cache_size, args.kick_out,statistic_data,max_std,0.9)
            # for key in local_cache[0]:
            #     print(key,local_cache[0][key]['time'],local_cache[0][key]['distribution'],local_cache[0][key]['cache_score'])
            # cache_info = np.zeros([num_car])
            # for index in range(num_car):
            #     # cache_info_by_time[0][index] += 1 
            #     cache_info[index] += 1
            #     for key in local_cache[index]:
            #         # print(local_cache[index][key]['time'])
            #         cache_info[key] += 1
            cache_age = 0
            cache_num = 0
            for index in range(args.num_car):
                for slot in range(len(local_cache[index])):
                    cache_num += len(local_cache[index][slot]['models'])
                    for key in local_cache[index][slot]['models']:
                        cache_age += i-local_cache[index][slot]['models'][key]['time']
            avg_cache_age = cache_age/cache_num
            with open(model_dir+'/cache_age_cache_num'+'_'+str(cache_size)+'_'+str(args.epoch_time)+'_'+str(args.kick_out)+'.txt','a') as file:
                file.write(str(i)+':')
                file.write(str(avg_cache_age)+'\t'+str(cache_num/num_car)+'\n')
            
    return loss,[0,0,0],class_acc_list, acc_local 


def Decentralized_fresh_Cache_process(suffix_dir,train_loader,test_loader,num_round,local_ep):
    model = []
    local_cache = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = []
    fresh_class_time_table = np.zeros([num_car,num_car])
    learning_rate = lr
    # current_model_time_table = np.zeros(num_car)
    # current_model_combination_table = np.eye(num_car)#statistic_data#
    current_class_test = np.zeros([num_car,10])
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir
    pair,area = write_info(model_dir)
    # model_dir = 'dfl_model_fresh_'+str(aggregation_metric)+'_'+str(cache_update_metric)
    for i in range(num_car):
        local_cache.append({})
        model.append(copy.deepcopy(global_model))
        model[i].to(device)
        optimizer.append(optim.SGD(params=model[i].parameters(), lr=lr))
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    for i in range(num_round):
        # old_model = copy.deepcopy(model)
        print('######################################################################')
        print('This is the round:',i)
        if i>0 and i % decay_round == 0:
            learning_rate = learning_rate/10
        for index in range(num_car):
            change_learning_rate(optimizer[index], learning_rate)
        for param_group in optimizer[0].param_groups:
            print(param_group['lr'])
        # if i %300== 0:
        #     learning_rate  = learning_rate/5
        #     for index in range(num_car):
        #         for param_group in optimizer[index].param_groups:
        #             param_group['lr'] = learning_rate
        with open(model_dir+'/log.txt','a') as file:
            file.write('This is the round:'+str(i)+'\n')
            for param_group in optimizer[0].param_groups:
                file.write('current lr is: '+str(param_group['lr'])+'\n')
        start_time = time.time()
        #carry out local training
        for index in range(num_car):
            normal_training_process(model[index],optimizer[index],train_loader,local_ep,loss[index])
            fresh_class_time_table[index][index] = i#fresh_class_time_table[index][index] + 1
                #print(model[a].state_dict()['fc4.bias'])
                    #meet with each other:
            #print(model[a].state_dict()['fc4.bias'])
            
        # #update time table by decay parameter alpha
        # fresh_class_time_table = alpha_time*i + (1- alpha_time)*fresh_class_time_table
        #update fresh table
        
        # current_model_time_table = alpha_time*i+(1-alpha_time)*current_model_time_table
        # current_model_combination_table = alpha_combination*statistic_data + (1-alpha_combination)*current_model_combination_table## 
        # for index in range(num_car):
        #     model[index].load_state_dict(torch.load(model_dir+'/model_'+str(index)+'.pt'))
        #     model[index].to(device)
        model_before_aggregation = copy.deepcopy(model)
        final_test(model_before_aggregation, acc_global_before_aggregation, class_acc_list_before_aggregation)
        # print(acc_global_before_aggregation[a][-1])
        
        
        # #First put self model in own cache
        # for index in range(num_car):
        #     put_own_model_into_cache(local_cache, model,index,i)
        #information exchange: update trace, cache, diag_fisher_matrix
        
        for a,b in pair[i]: 
            update_model_cache_fresh(local_cache,  model[a],model[b],a,b,i, cache_size, fresh_class_time_table, cache_update_metric, args.kick_out)
            # update_model_cache(local_cache, model,a,b,i, cache_size)
            # update_model_cache_only_one(local_cache,model, a,b,i,cache_size)
            
        
        # do model aggregation
        print('Updated/aggregated model time/combination:')
        # print(current_model_time_table)
        # print(current_model_combination_table)
        # old_model_time_table  = copy.deepcopy(current_model_time_table)
        # old_model_combination_table = copy.deepcopy(current_model_combination_table)
        for index in range(num_car):
            model[index], fresh_class_time_table[index] = cache_average_process_fresh(model[index],index,local_cache[index], fresh_class_time_table[index],aggregation_metric,weights)
            model[index].to(device)
            # model[index] = cache_average_process(model[index],local_cache[index])

            # for key in local_cache[index]:
            #     duration[index][key]+=1
        #     if data_similarity[a][b]<0.2:# and np.average([acc_global_before_aggregation[a][-1],acc_global_before_aggregation[b][-1]])<0.5:
        #         model[a],model[b] = exchange_process(model[a],model[b])
        #     else:
        #         model[a],model[b] = average_process(model[a],model[b])
        
        # update model property
        
        # print(current_model_combination_table)
                #print('doing aggregate')
                #print(model[a].state_dict()['fc4.bias'])
    #final test acc:
        
        final_test(model, acc_global, class_acc_list)

        end_time = time.time()

        for index in range(num_car):
            current_class_test[index] = class_acc_list[index][-1]
        # print(current_model_combination_table)
        # print(current_class_test)
        print(fresh_class_time_table)
        for index in range(num_car):
            print(class_acc_list[index][-1])
        print(f'{end_time-start_time} [sec] for this epoch')
        print('Before/After aggregation acc:')
        for index in range(num_car):
            print('car:',index,'---------------------------------------------------------------')
            print(acc_global_before_aggregation[index][-1],acc_global[index][-1])
            print(class_acc_list_before_aggregation[index][-1])
            print(class_acc_list[index][-1])
            print('Local Cache model version:')
            for key in local_cache[index]:
                print(key,local_cache[index][key]['time'],local_cache[index][key]['fresh'])
        print('----------------------------------------------------------------------')
        print('Average test acc:',np.average(acc_global,axis=0)[-1])
        print('pair:',pair[i])
        with open(model_dir+'/log.txt','a') as file:
            file.write('fresh_class_time_table\n')
            file.write(str(fresh_class_time_table)+'\n')
            for index in range(num_car):
                file.write(str(index)+':'+str(class_acc_list[index][-1])+'\n')
            file.write(f'{end_time-start_time} [sec] for this epoch'+'\n')
            file.write('Before/After aggregation acc:'+'\n')
            for index in range(num_car):
                file.write('car:'+str(index)+'---------------------------------------------------------------'+'\n')
                file.write(str(acc_global_before_aggregation[index][-1])+','+str(acc_global[index][-1])+'\n')
                file.write(str(class_acc_list_before_aggregation[index][-1])+'\n')
                file.write(str(class_acc_list[index][-1])+'\n')
                file.write('Local Cache model version:'+'\n')
                for key in local_cache[index]:
                    file.write(str(key)+':'+str(local_cache[index][key]['time'])+','+str(local_cache[index][key]['fresh'])+'\n')#,local_cache[index][key]['fresh_metric'])
            file.write('----------------------------------------------------------------------'+'\n')
            file.write('Average test acc:'+str(np.average(acc_global,axis=0)[-1])+'\n')
        # for index in range(num_car):
        #     if acc_global[index][-1]<acc_global_before_aggregation[index][-1]: 
        #         model[index] = copy.deepcopy(model_before_aggregation[index])
        #         acc_global[index][-1] = acc_global_before_aggregation[index][-1]
        # for a,b in pair[i]:
        #     if acc_global[a][-1]<acc_global_before_aggregation[a][-1] and acc_global[b][-1]<acc_global_before_aggregation[b][-1]:
        #         model[b] = copy.deepcopy(model_before_aggregation[b])
        #         model[a] = copy.deepcopy(model_before_aggregation[a])
        #         acc_global[a][-1] = acc_global_before_aggregation[a][-1]
        #         acc_global[b][-1] = acc_global_before_aggregation[b][-1]
        
        with open(model_dir+'/average_acc_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir+'.txt','a') as file:
            file.write(str(i)+':'+str(np.average(acc_global,axis=0)[-1])+'\n')
        #write a code to early stop, if test acc remain unchanged for 10 rounds, then stop and return
        if i>early_stop_round and (abs(np.average(acc_global,axis=0)[-early_stop_round:]-np.average(acc_global,axis=0)[-1])<1e-7).all():
            print('early stop at round:',i)
            with open(model_dir+'/log.txt','a') as file:
                file.write('early stop at round:'+str(i)+'\n')
            break
    return loss,acc_global,class_acc_list, acc_local , model_dir

def Decentralized_fresh_with_datapoints_Cache_process(suffix_dir,train_loader,test_loader,num_round,local_ep):
    model = []
    local_cache = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = []
    fresh_class_time_table = np.zeros([num_car,num_car])
    learning_rate = lr
    # current_model_time_table = np.zeros(num_car)
    # current_model_combination_table = np.eye(num_car)#statistic_data#
    current_class_test = np.zeros([num_car,10])
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir
    pair,area = write_info(model_dir)
    # model_dir = 'dfl_model_fresh_'+str(aggregation_metric)+'_'+str(cache_update_metric)
    for i in range(num_car):
        local_cache.append({})
        model.append(copy.deepcopy(global_model))
        model[i].to(device)
        optimizer.append(optim.SGD(params=model[i].parameters(), lr=lr))
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    for i in range(num_round):
        # old_model = copy.deepcopy(model)
        print('######################################################################')
        print('This is the round:',i)
        if i>0 and i % decay_round == 0:
            learning_rate = learning_rate/10
        for index in range(num_car):
            change_learning_rate(optimizer[index], learning_rate)
        for param_group in optimizer[0].param_groups:
            print(param_group['lr'])
        # if i %300== 0:
        #     learning_rate  = learning_rate/5
        #     for index in range(num_car):
        #         for param_group in optimizer[index].param_groups:
        #             param_group['lr'] = learning_rate
        with open(model_dir+'/log.txt','a') as file:
            file.write('This is the round:'+str(i)+'\n')
            for param_group in optimizer[0].param_groups:
                file.write('current lr is: '+str(param_group['lr'])+'\n')
        start_time = time.time()
        #carry out local training
        for index in range(num_car):
            normal_training_process(model[index],optimizer[index],train_loader,local_ep,loss[index])
            fresh_class_time_table[index][index] = fresh_class_time_table[index][index] + 1
                #print(model[a].state_dict()['fc4.bias'])
                    #meet with each other:
            #print(model[a].state_dict()['fc4.bias'])
            
        # #update time table by decay parameter alpha
        # fresh_class_time_table = alpha_time*i + (1- alpha_time)*fresh_class_time_table
        #update fresh table
        
        # current_model_time_table = alpha_time*i+(1-alpha_time)*current_model_time_table
        # current_model_combination_table = alpha_combination*statistic_data + (1-alpha_combination)*current_model_combination_table## 
        # for index in range(num_car):
        #     model[index].load_state_dict(torch.load(model_dir+'/model_'+str(index)+'.pt'))
        #     model[index].to(device)
        model_before_aggregation = copy.deepcopy(model)
        final_test(model_before_aggregation, acc_global_before_aggregation, class_acc_list_before_aggregation)
        # print(acc_global_before_aggregation[a][-1])
        
        
        # #First put self model in own cache
        # for index in range(num_car):
        #     put_own_model_into_cache(local_cache, model,index,i)
        #information exchange: update trace, cache, diag_fisher_matrix
        
        for a,b in pair[i]: 
            update_model_cache_fresh(local_cache,  model[a],model[b],a,b,i, cache_size, fresh_class_time_table, cache_update_metric, args.kick_out)
            # update_model_cache(local_cache, model,a,b,i, cache_size)
            # update_model_cache_only_one(local_cache,model, a,b,i,cache_size)
            
        
        # do model aggregation
        print('Updated/aggregated model time/combination:')
        # print(current_model_time_table)
        # print(current_model_combination_table)
        # old_model_time_table  = copy.deepcopy(current_model_time_table)
        # old_model_combination_table = copy.deepcopy(current_model_combination_table)
        for index in range(num_car):
            model[index], fresh_class_time_table[index] = cache_average_process_fresh(model[index],local_cache[index], fresh_class_time_table[index],aggregation_metric)
            model[index].to(device)
            # model[index] = cache_average_process(model[index],local_cache[index])

            # for key in local_cache[index]:
            #     duration[index][key]+=1
        #     if data_similarity[a][b]<0.2:# and np.average([acc_global_before_aggregation[a][-1],acc_global_before_aggregation[b][-1]])<0.5:
        #         model[a],model[b] = exchange_process(model[a],model[b])
        #     else:
        #         model[a],model[b] = average_process(model[a],model[b])
        
        
        # update model property
        
        # print(current_model_combination_table)
                #print('doing aggregate')
                #print(model[a].state_dict()['fc4.bias'])
    #final test acc:
        
        final_test(model, acc_global, class_acc_list)

        end_time = time.time()


        for index in range(num_car):
            current_class_test[index] = class_acc_list[index][-1]
        # print(current_model_combination_table)
        # print(current_class_test)
        print(fresh_class_time_table)
        for index in range(num_car):
            print(class_acc_list[index][-1])
        print(f'{end_time-start_time} [sec] for this epoch')
        print('Before/After aggregation acc:')
        for index in range(num_car):
            print('car:',index,'---------------------------------------------------------------')
            print(acc_global_before_aggregation[index][-1],acc_global[index][-1])
            print(class_acc_list_before_aggregation[index][-1])
            print(class_acc_list[index][-1])
            print('Local Cache model version:')
            for key in local_cache[index]:
                print(key,local_cache[index][key]['time'],local_cache[index][key]['fresh'])
        print('----------------------------------------------------------------------')
        print('Average test acc:',np.average(acc_global,axis=0)[-1])
        print('pair:',pair[i])
        with open(model_dir+'/log.txt','a') as file:
            file.write('fresh_class_time_table\n')
            file.write(str(fresh_class_time_table)+'\n')
            for index in range(num_car):
                file.write(str(index)+':'+str(class_acc_list[index][-1])+'\n')
            file.write(f'{end_time-start_time} [sec] for this epoch'+'\n')
            file.write('Before/After aggregation acc:'+'\n')
            for index in range(num_car):
                file.write('car:'+str(index)+'---------------------------------------------------------------'+'\n')
                file.write(str(acc_global_before_aggregation[index][-1])+','+str(acc_global[index][-1])+'\n')
                file.write(str(class_acc_list_before_aggregation[index][-1])+'\n')
                file.write(str(class_acc_list[index][-1])+'\n')
                file.write('Local Cache model version:'+'\n')
                for key in local_cache[index]:
                    file.write(str(key)+':'+str(local_cache[index][key]['time'])+','+str(local_cache[index][key]['fresh'])+'\n')#,local_cache[index][key]['fresh_metric'])
            file.write('----------------------------------------------------------------------'+'\n')
            file.write('Average test acc:'+str(np.average(acc_global,axis=0)[-1])+'\n')
        # for index in range(num_car):
        #     if acc_global[index][-1]<acc_global_before_aggregation[index][-1]: 
        #         model[index] = copy.deepcopy(model_before_aggregation[index])
        #         acc_global[index][-1] = acc_global_before_aggregation[index][-1]
        # for a,b in pair[i]:
        #     if acc_global[a][-1]<acc_global_before_aggregation[a][-1] and acc_global[b][-1]<acc_global_before_aggregation[b][-1]:
        #         model[b] = copy.deepcopy(model_before_aggregation[b])
        #         model[a] = copy.deepcopy(model_before_aggregation[a])
        #         acc_global[a][-1] = acc_global_before_aggregation[a][-1]
        #         acc_global[b][-1] = acc_global_before_aggregation[b][-1]
        
        with open(model_dir+'/average_acc_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir+'.txt','a') as file:
            file.write(str(i)+':'+str(np.average(acc_global,axis=0)[-1])+'\n')
        #write a code to early stop, if test acc remain unchanged for 10 rounds, then stop and return
        if i>early_stop_round and (abs(np.average(acc_global,axis=0)[-early_stop_round:]-np.average(acc_global,axis=0)[-1])<1e-7).all():
            print('early stop at round:',i)
            with open(model_dir+'/log.txt','a') as file:
                file.write('early stop at round:'+str(i)+'\n')
            break
    return loss,acc_global,class_acc_list, acc_local , model_dir

def Decentralized_fresh_count_Cache_process(suffix_dir,train_loader, test_loader,num_round,local_ep):
    model = []
    local_cache = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = []
    learning_rate = lr
    fresh_class_time_table = np.zeros([num_car,num_car])
    cache_statistic_table_by_version = np.zeros([num_car,num_round])
    cache_statistic_table_by_car_history = np.zeros([num_car,num_round])
    # current_model_time_table = np.zeros(num_car)
    # current_model_combination_table = np.eye(num_car)#statistic_data#
    current_class_test = np.zeros([num_car,10])
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir
    pair,area = write_info(model_dir)
    # model_dir = 'dfl_model_fresh_count_2'
    for i in range(num_car):
        local_cache.append({})
        model.append(copy.deepcopy(global_model))
        model[i].to(device)
        optimizer.append(optim.SGD(params=model[i].parameters(), lr=lr))
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    for i in range(num_round):
        # old_model = copy.deepcopy(model)
        print('######################################################################')
        print('This is the round:',i)
        if i>0 and i % decay_round == 0:
            learning_rate = learning_rate/10
        for index in range(num_car):
            change_learning_rate(optimizer[index], learning_rate)
        for param_group in optimizer[0].param_groups:
            print(param_group['lr'])
        # if i %300== 0:
        #     learning_rate  = learning_rate/5
        #     for index in range(num_car):
        #         for param_group in optimizer[index].param_groups:
        #             param_group['lr'] = learning_rate
        with open(model_dir+'/log.txt','a') as file:
            file.write('This is the round:'+str(i)+'\n')
            for param_group in optimizer[0].param_groups:
                file.write('current lr is: '+str(param_group['lr'])+'\n')
        start_time = time.time()
        #carry out local training
        for index in range(num_car):
            normal_training_process(model[index],optimizer[index],train_loader,local_ep,loss[index])
            fresh_class_time_table[index][index] = i
                #print(model[a].state_dict()['fc4.bias'])
                    #meet with each other:
            #print(model[a].state_dict()['fc4.bias'])
            
        # #update time table by decay parameter alpha
        # fresh_class_time_table = alpha_time*i + (1- alpha_time)*fresh_class_time_table
        #update fresh table
        
        # current_model_time_table = alpha_time*i+(1-alpha_time)*current_model_time_table
        # current_model_combination_table = alpha_combination*statistic_data + (1-alpha_combination)*current_model_combination_table## 
        # for index in range(num_car):
        #     model[index].load_state_dict(torch.load(model_dir+'/model_'+str(index)+'.pt'))
        #     model[index].to(device)
        model_before_aggregation = copy.deepcopy(model)
        final_test(model_before_aggregation, acc_global_before_aggregation, class_acc_list_before_aggregation)
        # print(acc_global_before_aggregation[a][-1])
        
        
        # #First put self model in own cache
        # for index in range(num_car):
        #     put_own_model_into_cache(local_cache, model,index,i)
        #information exchange: update trace, cache, diag_fisher_matrix
        
        for a,b in pair[i]: 
            update_model_cache_fresh_count(local_cache,  model[a],model[b],a,b,i, cache_size, fresh_class_time_table, cache_statistic_table_by_version, args.kick_out)
            # update_model_cache(local_cache, model,a,b,i, cache_size)
            # update_model_cache_only_one(local_cache,model, a,b,i,cache_size)
            
        
        # do model aggregation
        print('Updated/aggregated model time/combination:')
        # print(current_model_time_table)
        # print(current_model_combination_table)
        # old_model_time_table  = copy.deepcopy(current_model_time_table)
        # old_model_combination_table = copy.deepcopy(current_model_combination_table)
        for index in range(num_car):
            # model[index], fresh_class_time_table[index] = cache_average_process_fresh(model[index],local_cache[index], fresh_class_time_table[index],aggregation_metric)
            model[index] = cache_average_process(model[index],local_cache[index])
            model[index].to(device)
            # for key in local_cache[index]:
            #     duration[index][key]+=1
        #     if data_similarity[a][b]<0.2:# and np.average([acc_global_before_aggregation[a][-1],acc_global_before_aggregation[b][-1]])<0.5:
        #         model[a],model[b] = exchange_process(model[a],model[b])
        #     else:
        #         model[a],model[b] = average_process(model[a],model[b])
        
        
        # update model property
        
        # print(current_model_combination_table)
                #print('doing aggregate')
                #print(model[a].state_dict()['fc4.bias'])
    #final test acc:
        
        
        final_test(model, acc_global, class_acc_list)

        end_time = time.time()

        for index in range(num_car):
            current_class_test[index] = class_acc_list[index][-1]
        # print(current_model_combination_table)
        # print(current_class_test)
        print(fresh_class_time_table)
        for index in range(num_car):
            print(class_acc_list[index][-1])
        print(f'{end_time-start_time} [sec] for this epoch')
        print('Before/After aggregation acc:')
        for index in range(num_car):
            print('car:',index,'---------------------------------------------------------------')
            print(acc_global_before_aggregation[index][-1],acc_global[index][-1])
            print(class_acc_list_before_aggregation[index][-1])
            print(class_acc_list[index][-1])
            print('Local Cache model version:')
            for key in local_cache[index]:
                print(key,local_cache[index][key]['time'],local_cache[index][key]['cache_score'])
                cache_statistic_table_by_car_history[index][i] += 1
                cache_statistic_table_by_version[key][i] += 1
        print('current cached update time')
        print(cache_statistic_table_by_version[:,i])
        print('----------------------------------------------------------------------')
        print('Average test acc:',np.average(acc_global,axis=0)[-1])
        print('pair:',pair[i])
        with open(model_dir+'/log.txt','a') as file:
            file.write('fresh_class_time_table\n')
            file.write(str(fresh_class_time_table)+'\n')
            for index in range(num_car):
                file.write(str(index)+':'+str(class_acc_list[index][-1])+'\n')
            file.write(f'{end_time-start_time} [sec] for this epoch'+'\n')
            file.write('Before/After aggregation acc:'+'\n')
            for index in range(num_car):
                file.write('car:'+str(index)+'---------------------------------------------------------------'+'\n')
                file.write(str(acc_global_before_aggregation[index][-1])+','+str(acc_global[index][-1])+'\n')
                file.write(str(class_acc_list_before_aggregation[index][-1])+'\n')
                file.write(str(class_acc_list[index][-1])+'\n')
                file.write('Local Cache model version:'+'\n')
                for key in local_cache[index]:
                    file.write(str(key)+':'+str(local_cache[index][key]['time'])+','+str(local_cache[index][key]['cache_score'])+'\n')#,local_cache[index][key]['fresh_metric'])
            file.write('current cached update time:\n')
            file.write(str(cache_statistic_table_by_version[:,i])+'\n')
            file.write('----------------------------------------------------------------------'+'\n')
            file.write('Average test acc:'+str(np.average(acc_global,axis=0)[-1])+'\n')
        # for index in range(num_car):
        #     if acc_global[index][-1]<acc_global_before_aggregation[index][-1]: 
        #         model[index] = copy.deepcopy(model_before_aggregation[index])
        #         acc_global[index][-1] = acc_global_before_aggregation[index][-1]
        # for a,b in pair[i]:
        #     if acc_global[a][-1]<acc_global_before_aggregation[a][-1] and acc_global[b][-1]<acc_global_before_aggregation[b][-1]:
        #         model[b] = copy.deepcopy(model_before_aggregation[b])
        #         model[a] = copy.deepcopy(model_before_aggregation[a])
        #         acc_global[a][-1] = acc_global_before_aggregation[a][-1]
        #         acc_global[b][-1] = acc_global_before_aggregation[b][-1]
        
        with open(model_dir+'/average_acc_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir+'.txt','a') as file:
            file.write(str(i)+':'+str(np.average(acc_global,axis=0)[-1])+'\n')
    return loss,acc_global,class_acc_list, acc_local, model_dir



def Decentralized_fresh_history_Cache_process(suffix_dir,train_loader, test_loader,num_round,local_ep):
    model = []
    local_cache = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = []
    fresh_time_table = np.zeros(num_car)
    cache_statistic_table_by_version = np.zeros([num_car,num_round])
    cache_statistic_table_by_car_history = np.zeros([num_car,num_round])
    # current_model_time_table = np.zeros(num_car)
    # current_model_combination_table = np.eye(num_car)#statistic_data#
    current_class_test = np.zeros([num_car,10])
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir
    pair,area = write_info(model_dir)
    # model_dir = 'dfl_model_fresh_2'
    for i in range(num_car):
        local_cache.append({})
        model.append(copy.deepcopy(global_model))
        model[i].to(device)
        optimizer.append(optim.SGD(params=model[i].parameters(), lr=lr))
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    for i in range(num_round):
        # old_model = copy.deepcopy(model)
        print('######################################################################')
        print('This is the round:',i)
        start_time = time.time()
        #carry out local training
        for index in range(num_car):
            normal_training_process(model[index],optimizer[index],train_loader,local_ep,loss[index])
            fresh_time_table[index] += 1*alpha_time
                #print(model[a].state_dict()['fc4.bias'])
                    #meet with each other:
            #print(model[a].state_dict()['fc4.bias'])
            
        # #update time table by decay parameter alpha
        # fresh_class_time_table = alpha_time*i + (1- alpha_time)*fresh_class_time_table
        #update fresh table
        
        # current_model_time_table = alpha_time*i+(1-alpha_time)*current_model_time_table
        # current_model_combination_table = alpha_combination*statistic_data + (1-alpha_combination)*current_model_combination_table## 
        # for index in range(num_car):
        #     model[index].load_state_dict(torch.load(model_dir+'/model_'+str(index)+'.pt'))
        #     model[index].to(device)
        model_before_aggregation = copy.deepcopy(model)
        final_test(model_before_aggregation, acc_global_before_aggregation, class_acc_list_before_aggregation)
        # print(acc_global_before_aggregation[a][-1])
        
        
        # #First put self model in own cache
        # for index in range(num_car):
        #     put_own_model_into_cache(local_cache, model,index,i)
        #information exchange: update trace, cache, diag_fisher_matrix
        
        for a,b in pair[i]: 
            update_model_cache_fresh_v3(local_cache,  model[a],model[b],a,b,i, cache_size, fresh_time_table,  args.kick_out)
            # update_model_cache(local_cache, model,a,b,i, cache_size)
            # update_model_cache_only_one(local_cache,model, a,b,i,cache_size)
            
        
        # do model aggregation
        print('Updated/aggregated model time/combination:')
        # print(current_model_time_table)
        # print(current_model_combination_table)
        # old_model_time_table  = copy.deepcopy(current_model_time_table)
        # old_model_combination_table = copy.deepcopy(current_model_combination_table)
        for index in range(num_car):
            model[index], fresh_time_table[index] = cache_average_process_fresh_v3(model[index],local_cache[index], fresh_time_table[index],aggregation_metric)
            # model[index] = cache_average_process(model[index],local_cache[index])

            # for key in local_cache[index]:
            #     duration[index][key]+=1
        #     if data_similarity[a][b]<0.2:# and np.average([acc_global_before_aggregation[a][-1],acc_global_before_aggregation[b][-1]])<0.5:
        #         model[a],model[b] = exchange_process(model[a],model[b])
        #     else:
        #         model[a],model[b] = average_process(model[a],model[b])
        
        
        # update model property
        
        # print(current_model_combination_table)
                #print('doing aggregate')
                #print(model[a].state_dict()['fc4.bias'])
    #final test acc:
        
        final_test(model, acc_global, class_acc_list)

        end_time = time.time()

        for index in range(num_car):
            current_class_test[index] = class_acc_list[index][-1]
        # print(current_model_combination_table)
        # print(current_class_test)
        print(fresh_time_table)
        for index in range(num_car):
            print(class_acc_list[index][-1])
        print(f'{end_time-start_time} [sec] for this epoch')
        print('Before/After aggregation acc:')
        for index in range(num_car):
            print('car:',index,'---------------------------------------------------------------')
            print(acc_global_before_aggregation[index][-1],acc_global[index][-1])
            print(class_acc_list_before_aggregation[index][-1])
            print(class_acc_list[index][-1])
            print('Local Cache model version:')
            for key in local_cache[index]:
                print(key,local_cache[index][key]['time'],local_cache[index][key]['fresh'])
                cache_statistic_table_by_car_history[index][i] += 1
                cache_statistic_table_by_version[key][i] += 1
        print('current cached update time')
        print(cache_statistic_table_by_version[:,i])
        # for index in range(num_car):
        #     if acc_global[index][-1]<acc_global_before_aggregation[index][-1]: 
        #         model[index] = copy.deepcopy(model_before_aggregation[index])
        #         acc_global[index][-1] = acc_global_before_aggregation[index][-1]
        # for a,b in pair[i]:
        #     if acc_global[a][-1]<acc_global_before_aggregation[a][-1] and acc_global[b][-1]<acc_global_before_aggregation[b][-1]:
        #         model[b] = copy.deepcopy(model_before_aggregation[b])
        #         model[a] = copy.deepcopy(model_before_aggregation[a])
        #         acc_global[a][-1] = acc_global_before_aggregation[a][-1]
        #         acc_global[b][-1] = acc_global_before_aggregation[b][-1]
        print('----------------------------------------------------------------------')
        print('Average test acc:',np.average(acc_global,axis=0)[-1])
        print('pair:',pair[i])
        # print('Duration:')
        # print(duration)
    return loss,acc_global,class_acc_list, acc_local, model_dir

def Decentralized_best_Cache_process(suffix_dir,train_loader,test_loader,num_round,local_ep):
    model = []
    local_cache = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = []
    fresh_class_time_table = np.zeros([num_car,num_car])
    test_score = []
    # current_model_time_table = np.zeros(num_car)
    # current_model_combination_table = np.eye(num_car)#statistic_data#
    current_class_test = np.zeros([num_car,10])
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir
    for i in range(num_car):
        local_cache.append({})
        model.append(copy.deepcopy(global_model))
        model[i].to(device)
        optimizer.append(optim.SGD(params=model[i].parameters(), lr=lr))
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
        test_score.append([])
    for i in range(num_round):
        # old_model = copy.deepcopy(model)
        print('######################################################################')
        print('This is the round:',i)
        start_time = time.time()
        #carry out local training
        for index in range(num_car):
            normal_training_process(model[index],optimizer[index],train_loader,local_ep,loss[index])
            fresh_class_time_table[index][index] = i
                #print(model[a].state_dict()['fc4.bias'])
                    #meet with each other:
            #print(model[a].state_dict()['fc4.bias'])
            
        # #update time table by decay parameter alpha
        # fresh_class_time_table = alpha_time*i + (1- alpha_time)*fresh_class_time_table
        #update fresh table
        
        # current_model_time_table = alpha_time*i+(1-alpha_time)*current_model_time_table
        # current_model_combination_table = alpha_combination*statistic_data + (1-alpha_combination)*current_model_combination_table## 
        # for index in range(num_car):
        #     model[index].load_state_dict(torch.load(model_dir+'/model_'+str(index)+'.pt'))
        #     model[index].to(device)
        model_before_aggregation = copy.deepcopy(model)
        final_test(model_before_aggregation, acc_global_before_aggregation, class_acc_list_before_aggregation)
        # print(acc_global_before_aggregation[a][-1])
        
        
        # #First put self model in own cache
        # for index in range(num_car):
        #     put_own_model_into_cache(local_cache, model,index,i)
        #information exchange: update trace, cache, diag_fisher_matrix
        
        for a,b in pair[i]: 
            # update_model_cache_fresh(local_cache, model,a,b,i, cache_size, fresh_class_time_table, cache_update_metric, args.kick_out)
            # update_model_cache(local_cache, model,a,b,i, cache_size)
            # update_model_cache_only_one(local_cache,model, a,b,i,cache_size)
            update_best_model_cache(local_cache,  model[a],model[b],a,b,i, cache_size, test_score)
        
        # do model aggregation
        print('Updated/aggregated model time/combination:')
        # print(current_model_time_table)
        # print(current_model_combination_table)
        # old_model_time_table  = copy.deepcopy(current_model_time_table)
        # old_model_combination_table = copy.deepcopy(current_model_combination_table)
        for index in range(num_car):
            # model[index], fresh_class_time_table[index] = cache_average_process_fresh(model[index],local_cache[index], fresh_class_time_table[index],aggregation_metric)
            model[index] = cache_average_process(model[index],local_cache[index])
            model[index].to(device)
            # for key in local_cache[index]:
            #     duration[index][key]+=1
        #     if data_similarity[a][b]<0.2:# and np.average([acc_global_before_aggregation[a][-1],acc_global_before_aggregation[b][-1]])<0.5:
        #         model[a],model[b] = exchange_process(model[a],model[b])
        #     else:
        #         model[a],model[b] = average_process(model[a],model[b])
        
        
        # update model property
        
        # print(current_model_combination_table)
                #print('doing aggregate')
                #print(model[a].state_dict()['fc4.bias'])
    #final test acc:
        
        final_test(model, acc_global, class_acc_list)

        end_time = time.time()


        for index in range(num_car):
            test_score[index] = copy.deepcopy(class_acc_list_before_aggregation[index][-1])
        for index in range(num_car):
            current_class_test[index] = class_acc_list[index][-1]
        # print(current_model_combination_table)
        # print(current_class_test)
        # print(fresh_class_time_table)
        for index in range(num_car):
            print(class_acc_list[index][-1])
        print(f'{end_time-start_time} [sec] for this epoch')
        print('Before/After aggregation acc:')
        for index in range(num_car):
            print('car:',index,'---------------------------------------------------------------')
            print(acc_global_before_aggregation[index][-1],acc_global[index][-1])
            print(class_acc_list_before_aggregation[index][-1])
            print(class_acc_list[index][-1])
            print('Local Cache model version:')
            for key in local_cache[index]:
                print(key,local_cache[index][key]['time'])#,local_cache[index][key]['fresh_metric'])
        # for index in range(num_car):
        #     if acc_global[index][-1]<acc_global_before_aggregation[index][-1]: 
        #         model[index] = copy.deepcopy(model_before_aggregation[index])
        #         acc_global[index][-1] = acc_global_before_aggregation[index][-1]
        # for a,b in pair[i]:
        #     if acc_global[a][-1]<acc_global_before_aggregation[a][-1] and acc_global[b][-1]<acc_global_before_aggregation[b][-1]:
        #         model[b] = copy.deepcopy(model_before_aggregation[b])
        #         model[a] = copy.deepcopy(model_before_aggregation[a])
        #         acc_global[a][-1] = acc_global_before_aggregation[a][-1]
        #         acc_global[b][-1] = acc_global_before_aggregation[b][-1]
        print('----------------------------------------------------------------------')
        print('Average test acc:',np.average(acc_global,axis=0)[-1])
        print('pair:',pair[i])
        # print('Duration:')
        # print(duration)
    return loss,acc_global,class_acc_list, acc_local 
    
def ml_process(suffix_dir, train_loader,test_loader, num_round,local_ep):
    model = copy.deepcopy(global_model)
    acc = []
    loss = []
    optimizer = optim.SGD(params=model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_factor, patience=args.lr_patience, verbose=False)
    learning_rate = lr
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+str(Randomseed)+'_'+args.algorithm+suffix_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    for i in range(num_round):
        print('This is the round:',i)
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        with open(model_dir+'/log.txt','a') as file:
            file.write('This is the round:'+str(i)+'\n')
            for param_group in optimizer.param_groups:
                file.write('current lr is: '+str(param_group['lr'])+'\n')
        start_time = time.time()
        #carry out local training
        for _ in tqdm(range(local_ep),disable=True):
            loss.append(normal_train(model, optimizer, full_loader))
        current_acc, class_acc = test(model, test_loader,num_class)
        acc.append(current_acc) 
        end_time = time.time()
        print(f'{end_time-start_time} [sec] for this epoch')
        print('Acc:',current_acc)
        print('Class acc:',class_acc)
        if use_lr_scheduler:
            scheduler.step(current_acc)
        else:
            learning_rate = update_learning_rate(i, learning_rate)
            change_learning_rate(optimizer, learning_rate)
        with open(model_dir+'/log.txt','a') as file:
            file.write(f'{end_time-start_time} [sec] for this epoch'+'\n')
            file.write('Acc:'+str(current_acc)+'\n')
            file.write('Class acc:'+str(class_acc)+'\n')
        with open(model_dir+'/acc_'+task+'_'+str(Randomseed)+'_'+args.algorithm+suffix_dir+'.txt','a') as file:
            file.write(str(i)+':'+str(current_acc)+'\n')
        #write a code to early stop, if test acc remain unchanged for 10 rounds, then stop and return
        if i>early_stop_round and (abs(np.array(acc[-early_stop_round:])-acc[-1])<1e-7).all():
            print('early stop at round:',i)
            with open(model_dir+'/log.txt','a') as file:
                file.write('early stop at round:'+str(i)+'\n')
            break
    return loss,acc    


if __name__ == '__main__':
    task = 'fashionmnist'
    distribution = 'taxi_area'
    args.algorithm = 'test_taxi_priority'
    # distribution = 'iid'
    # args.algorithm = 'test_mixing'
    # args.algorithm = 'cache_all'
    cache_size = 3
    kick_out = True
    args.kick_out = 5
    if distribution == 'taxi_area':
        kick_out = False
    args.epoch_time = 120
    num_car = 100
    num_round = 1000

    balance = False
    test_ratio = 0.1
    use_lr_scheduler = True
    data_distribution = distribution
    car_type_list =  [0]*num_car
    type_limits_taxi = {'1':4,'2':3,'3':3}
    type_limits_car = {'1':4,'2':3,'3':3}
    # target_labels = [[0,1,2,3],[4,5,6],[7,8,9]]
    target_labels = [[9,0,1,2],[3,4,5,6],[6,7,8,9]]
    # target_labels = [[7,8,9,0,1,2,3],[1,2,3,4,5,6],[4,5,6,7,8,9]]
    # type_limits_car = {'taxi':5,'car':5}
    if args.algorithm == 'ml':
        distribution = 'iid'
    if task == 'mnist':
        hidden_size = 64
        global_model = CNNMnist(1,10)
        num_class = 10
        #MNIST
        if distribution == 'iid':
            train_loader, sub_test_loader, test_loader, full_loader =  get_mnist_iid(num_car,batch_size, test_ratio)
        elif distribution == 'non-iid':
            train_loader, sub_test_loader, test_loader, full_loader =  get_mnist_imbalance(shards_allocation, num_car, batch_size, test_ratio)
        elif distribution == 'dirichlet':
            train_loader, sub_test_loader, test_loader, full_loader =  get_mnist_dirichlet(alpha, num_car,batch_size, test_ratio)
            data_distribution = data_distribution+'_'+str(alpha)
        elif distribution == 'by_area':
            train_dataset, sub_test_loader, test_loader, full_loader =  initial_mnist(batch_size, test_ratio)
            initial_size = int(len(train_dataset)/num_car)
            max_size = int(len(train_dataset)/num_car)
            update_fraction = 0.1
        elif distribution == 'taxi_area':
                car_type_list = []
                car_type_list += [1]*30
                car_type_list += [0]*4
                car_type_list += [2]*30
                car_type_list += [0]*3
                car_type_list += [3]*30
                car_type_list += [0]*3
                # num_car -= 10 
                car_area_list = []
                car_area_list += [1]*34
                car_area_list += [2]*33
                car_area_list += [3]*33
                train_loader, sub_test_loader, test_loader, full_loader =  get_mnist_taxi_area(shards_allocation,  batch_size,test_ratio,car_area_list,target_labels)
        else:
            raise ValueError('Error')
    elif task == 'cifar10':
        # global_model = Cifar10CnnModel()
        # global_model = AlexNet(10)
        global_model = ResNet18()
        num_class = 10
        # cifar10
        if distribution == 'iid':
            train_loader, sub_test_loader, test_loader, full_loader =  get_cifar10_iid(num_car,batch_size,test_ratio)
        elif distribution == 'non-iid':
            train_loader, sub_test_loader, test_loader, full_loader =  get_cifar10_imbalance(shards_allocation, num_car, batch_size,test_ratio)
        elif distribution == 'dirichlet':
            train_loader, sub_test_loader, test_loader, full_loader =  get_cifar10_dirichlet(alpha, num_car,batch_size,test_ratio)
            data_distribution = data_distribution+'_'+str(alpha)
        else:
            raise ValueError('Error')
    elif task == 'fashionmnist':
        global_model = CNNFashion_Mnist()
        num_class = 10
        # FashionMNIST
        # train_loader, test_loader, full_loader =  get_fashionmnist_dataset_iid(shards_per_user, num_car)
        if distribution == 'iid':
            train_loader, sub_test_loader, test_loader, full_loader =  get_fashionmnist_iid(num_car,batch_size, test_ratio)
        elif distribution == 'non-iid':
            train_loader, sub_test_loader, test_loader, full_loader =  get_fashionmnist_imbalance(shards_allocation, num_car, batch_size, test_ratio)
        elif distribution == 'dirichlet':
            train_loader, sub_test_loader, test_loader, full_loader =  get_fashionmnist_dirichlet(alpha, num_car,batch_size, test_ratio)
            data_distribution = data_distribution+'_'+str(alpha)
        elif distribution == 'taxi_area':
                car_type_list = []
                car_type_list += [1]*30
                car_type_list += [0]*4
                car_type_list += [2]*30
                car_type_list += [0]*3
                car_type_list += [3]*30
                car_type_list += [0]*3
                # num_car -= 10 
                car_area_list = []
                car_area_list += [1]*34
                car_area_list += [2]*33
                car_area_list += [3]*33
                train_loader, sub_test_loader, test_loader, full_loader =  get_fashionmnist_taxi_area(shards_allocation,  batch_size,test_ratio,car_area_list,target_labels)
        else:
            raise ValueError('Error')
    elif task == 'harbox':
        global_model = DNN_harbox()
        num_class = 5
        # HARBox
        if distribution == 'iid':
            train_loader, sub_test_loader, test_loader, full_loader =  get_harbox_iid(num_car,batch_size, test_ratio)
        elif distribution == 'non-iid':
            train_loader, sub_test_loader, test_loader, full_loader =  get_harbox_imbalance(shards_allocation, num_car, batch_size, test_ratio)
        elif distribution == 'dirichlet':
            train_loader, sub_test_loader, test_loader, full_loader =  get_harbox_dirichlet(alpha, num_car,batch_size, test_ratio)
            data_distribution = data_distribution+'_'+str(alpha)
        else:
            raise ValueError('Error')
    else:
            raise ValueError('Error')
    print(len(sub_test_loader.dataset))
    print(len(test_loader.dataset))
    
    
   
    
    # global_model.to(device)
    #statistic data distribution
    if distribution == 'by_area':
        print('This is by area distribution, no statistic data distribution.')
        print('Initial data size per car:',initial_size)
        print('Max data size per car:',max_size)
        print('Update fraction:',update_fraction)
        weights = [1]*num_car
    else:
        statistic_data = np.zeros([num_car,num_class])
        for i in range(num_car):
            for input, target in train_loader[i]:
                for item in target:
                    statistic_data[i][item] += 1
        print('Data distribution among cars:')
        print(statistic_data)
        max_std = np.zeros(num_car)
        for i in range(num_car):
            for j in range(num_car):
                if max_std[i] < np.var(statistic_data[i]- statistic_data[j]):
                    max_std[i] = np.var(statistic_data[i]- statistic_data[j])
        data_points = np.sum(statistic_data,axis = 1)
        print(sum(data_points))
        print(data_points/sum(np.array(data_points)))
        if args.weighted_aggregation == True:
            weights = data_points
        else:
            weights = [1]*num_car    
        data_similarity = cosine_similarity(statistic_data)
        print(data_similarity)
    
    numbers = list(range(num_car))
    random.shuffle(numbers)
    mixing_pair = []
    mixing_table = []
    for i in range(cache_size):
        mixing_pair.append([])
    for i in range(num_car):
        mixing_pair[i%cache_size].append(numbers[i])
        mixing_table.append([])
    for slot in range(cache_size):
        for key in mixing_pair[slot]:
            mixing_table[key] = slot
    print('mixing table',mixing_table)
    print('mixing pair',mixing_pair)
    
    numbers = list(range(num_car))
    pair = []
    
    # # forced p2p pairing
    # for i in range(1000):
    #     random.shuffle(numbers)
    #     pairs = [(numbers[i], numbers[i + 1]) for i in range(0, num_car, 2)]
    #     pair.append(pairs)
        
    # Bernulli pairing
    # for k in range(1000):
    #     random.shuffle(numbers)
    #     pairs = []
    #     for i in range(num_car):
    #         for j in range(i+1, num_car):
    #             if random.random() < car_meet_p:
    #                 pairs.append((i,j))
    #     random.shuffle(pairs)
    #     pair.append(pairs)
    
    # use road sim to generate car pairings.

    start_time = time.time()
    date_time = datetime.datetime.fromtimestamp(start_time)
    # Use the algorithm argument to run the corresponding function
    if args.algorithm == 'ml':
        loss, acc = ml_process(suffix, train_loader,test_loader,num_round,local_ep)
    elif args.algorithm == 'cfl':
        fl_loss, fl_test_acc, model_dir = Centralized_process(suffix,train_loader,test_loader,num_round,local_ep)
    # elif args.algorithm == 'centralized_EWC_process':
    #     fl_loss, fl_acc_global, fl_acc_local, fl_test_acc, model_dir = centralized_EWC_process(args.train_loader, args.test_loader, args.num_round, args.local_ep)
    #     print(fl_test_acc)
    elif args.algorithm == 'dfl':
        dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local, model_dir = Decentralized_process(suffix,train_loader,test_loader,num_round,local_ep)
    elif args.algorithm == 'dfl_fair':
            dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local, model_dir = Decentralized_fair_process(suffix,train_loader,test_loader,num_round,local_ep)
    elif args.algorithm == 'cache_one':
        dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local, model_dir = Decentralized_Cache_one_process(suffix,train_loader,test_loader,num_round,local_ep)
    elif args.algorithm == 'cache_all':
        dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local, model_dir = Decentralized_Cache_all_process(suffix,train_loader,test_loader,num_round,local_ep)
    elif args.algorithm == 'fresh':
        dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_loca, model_dirl  = Decentralized_fresh_Cache_process(suffix,train_loader,test_loader,num_round,local_ep)
    elif args.algorithm == 'fresh_update_fresh_by_dp':
        dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local, model_dir  = Decentralized_fresh_with_datapoints_Cache_process(suffix,train_loader,test_loader,num_round,local_ep)
    elif args.algorithm == 'fresh_count':
        dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local, model_dir  = Decentralized_fresh_count_Cache_process(suffix,train_loader,test_loader,num_round,local_ep)
    elif args.algorithm == 'fresh_history':
        dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local, model_dir  = Decentralized_fresh_history_Cache_process(suffix,train_loader,test_loader,num_round,local_ep)
    elif args.algorithm == 'best_test_cache':
        dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local, model_dir  = Decentralized_best_Cache_process(suffix,train_loader,test_loader,num_round,local_ep)
    elif args.algorithm == 'test':
        dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local = Decentralized_Cache_test()
    elif args.algorithm == 'test_taxi':
        dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local = Decentralized_Cache_test_taxi()
    elif args.algorithm == 'test_taxi_priority':
        dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local = Decentralized_Cache_test_taxi_priority()
    elif args.algorithm == 'test_mixing':
        dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local = Decentralized_Cache_test_mixing()
    else:
        raise ValueError('Error')
    if args.algorithm == 'ml':
        print(acc)
    elif args.algorithm == 'cfl':
        print(fl_test_acc)
    elif args.algorithm != 'test':
        print(np.average(dfl_acc_global,axis=0))
        end_time = time.time()
        total_training_time = end_time-start_time
        hours, remainder = divmod(total_training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        with open(model_dir+'/summary.txt','w') as file:
            file.write('Start time: ' + str(date_time.strftime('%Y-%m-%d %H:%M:%S'))+'\n')
            date_time = datetime.datetime.fromtimestamp(end_time)
            file.write('End time: ' + str(date_time.strftime('%Y-%m-%d %H:%M:%S'))+'\n')
            file.write(f'Total training time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds \n')
            for i in range(num_car):
                file.write('This is car '+str(i)+'\n')
                file.write(str(dfl_acc_global[i])+'\n')
            file.write('This is the average acc:\n')
            file.write(str(np.average(dfl_acc_global,axis=0))+'\n')
        print(f"Total training time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")
     
    
    
