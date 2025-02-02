# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import matplotlib.pyplot as plt

# plt.style.use("seaborn-white")
from mpi4py import MPI
import pickle
from mpi4py.util import pkl5
import argparse
import random
import torch
cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch import optim
from tqdm import tqdm
import time
import os    
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import copy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import datetime
import math
from Cache_algorithm import update_model_cache_mixing,update_model_cache_plus, update_model_cache_fresh,cache_average_process,cache_average_process_plus,cache_average_process_fresh,cache_average_process_mixing,update_model_cache,duration_in_future,update_model_cache_only_one,weighted_cache_average_process,update_best_model_cache,cache_average_process_fresh_without_model,update_model_cache_fresh_count, update_model_cache_fresh_v3,cache_average_process_fresh_v3
from aggregation import average_weights,normal_training_process,average_process,normal_train,subgradient_push_process, weighted_average_process
from utils_cnn import test
from model import get_P_matrix,CNNMnist,Cifar10CnnModel,CNNFashion_Mnist,AlexNet
from models import ResNet18
from data import get_mnist_iid, get_mnist_imbalance,get_mnist_dirichlet, get_cifar10_iid, get_cifar10_imbalance,get_cifar10_dirichlet,get_fashionmnist_iid, get_fashionmnist_imbalance,get_fashionmnist_dirichlet
from road_sim import generate_roadNet_pair_list_v2, generate_roadNet_pair_list
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
    'mnist', 'fashionmnist', 'cifar10'], help="Choose dataset task to run")
parser.add_argument("--local_ep", type=int, default=1, help="Number of local epochs")
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
parser.add_argument("--early_stop_round", type=int, default=30, help="Number of rounds test accuracy remain unchanged then stop")
parser.add_argument("--speed", type=float, default=13.59, help="Speed in m/s")
parser.add_argument("--communication_distance", type=int, default=100, help="Communication distance in meters")
parser.add_argument("--communication_interval", type=int, default=60, help="Communication interval in seconds")
# parser.add_argument("--importance", type=int, default=1, help="Importance")
parser.add_argument("--num_car", type=int, default=100, help="Number of cars")
parser.add_argument("--cache_size", type=int, default=3, help="Cache size")
# parser.add_argument("--parallel", action='store_true', help="Enable parallel processing")
# parser.add_argument('--non-parallel', dest='parallel', action='store_false')
# parser.set_defaults(parallel=False)
parser.add_argument("--augment", action='store_true', help="Enable augmentation")
parser.add_argument('--no-augment', dest='augment', action='store_false')
parser.set_defaults(augment=True)
parser.add_argument("--shards_allocation", nargs='+', type=int, default=[3,2,1,3,2,1,1,4,1,2]*10, help="Shards allocation")
parser.add_argument("--County", type=str, default="New York", help="County")
parser.add_argument("--kick_out", action='store_true', help="Enable kick out")
parser.add_argument('--no-kick_out', dest='kick_out', action='store_false')
parser.set_defaults(kick_out=True)
parser.add_argument("--weighted_aggregation", action='store_true', help="Enable weighted aggregation")
parser.add_argument('--no-weighted_aggregation', dest='weighted_aggregation', action='store_false')
parser.set_defaults(weighted_aggregation=True)
parser.add_argument("--algorithm", type=str, choices=[
    'ml', 'cfl', 'dfl','cache_one','cache_plus',
    'cache_all', 'fresh', 'fresh_update_fresh_by_dp', 'fresh_count',
    'fresh_history', 'best_test_cache'
], help="Choose the algorithm to run")
# Parse arguments
args = parser.parse_args()
task = args.task
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
communication_interval = args.communication_interval
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
    if not os.path.exists(write_dir+'/rank_log'):
        os.makedirs(write_dir+'/rank_log')
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
        file.write('num_round = '+str(num_round)+'\n')
        file.write('num_car = '+str(num_car)+'\n')
        file.write('communication_interval = '+str(communication_interval)+'\n')
        file.write('speed = '+str(speed)+'\n')
        file.write('communication_distance = '+str(communication_distance)+'\n')
        file.write('cache_size = '+str(cache_size)+'\n')
        # file.write('parallel = '+str(parallel)+'\n')
        file.write('shards_allocation = '+str(shards_allocation)+'\n')
        file.write('Aggregation weights = '+str(weights)+'\n')
        file.write('County = '+str(County)+'\n')
        file.write('kick_out = '+str(kick_out)+'\n')
        file.write('Data distribution among cars:\n')
        file.write(str(statistic_data)+'\n')
        file.write('Data similarity among cars:\n')
        file.write(str(data_similarity)+'\n')
        file.write('Data_points:\n'+str(data_points)+'\n')
        # file.write('mixing table'+str(mixing_table)+'\n')
        # file.write('mixing pair'+str(mixing_pair)+'\n')
    pair = generate_roadNet_pair_list_v2(write_dir,num_car, num_round,communication_distance,communication_interval,speed,County)
    with open(write_dir+'/pair.txt','w') as file:
        for i in range(num_round):
            file.write('Round:'+str(i)+': ')
            file.write(str(pair[i])+'\n')
    return pair
            
# Example: serialize a PyTorch model
def serialize_model(model):
    return pickle.dumps(model)

# Example: deserialize the byte stream back to a PyTorch model
def deserialize_model(byte_stream):
    return pickle.loads(byte_stream)


def adjust_learning_rate(optimizer, epoch, initial_lr, decay_rate):
    lr = initial_lr
    if epoch>0 and epoch % 100 == 0:
        lr = lr/10
    # lr = initial_lr / (1 + decay_rate * epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

def update_learning_rate(i,learning_rate):
    flag_for_lr_change = False
    if i>0 and i % decay_round == 0:
        learning_rate = learning_rate/10
        flag_for_lr_change = True
    return learning_rate, flag_for_lr_change

def clean_up_local_cache(local_cache):
    # for index in range(num_car):
    #     local_cache[index] = {}
    return local_cache

def change_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def final_test(model_list, acc_list,class_acc_list):
    for i in range(len(model_list)):
        acc, class_acc = test(model_list[i], test_loader)
        acc_list[i].append(acc)
        class_acc_list[i].append(class_acc)
        
def mpi_send_model_host(model_list):
    for i in range(1,size):
        # send_model_group = []
        for j in client_rank_mapping[i]:
            comm.send(serialize_model(model_list[j].state_dict()), dest=i, tag=11)
            # send_model_group.append(serialize_model(model_list[j].state_dict()))
        # comm.send(send_model_group, dest=i, tag=11)
        
def mpi_receive_model_host(model_list):
    for i in range(1,size):
        # recv_model_group = comm.recv(source=i, tag=12)
        for index in range(len(client_rank_mapping[i])):
            j = client_rank_mapping[i][index]
            # model_list[j].load_state_dict(deserialize_model(recv_model_group[index]))
            model_list[j].load_state_dict(deserialize_model(comm.recv(source=i, tag=12)))
            
def mpi_train_host(model_list,optimizer,train_loader,local_ep,loss,model_dir):
    # send model to ranks
    mpi_send_model_host(model_list)
    #train model on host
    time_1 = time.time()
    for j in client_rank_mapping[0]:
        model_list[j].to(device)
        normal_training_process(model_list[j],optimizer[j],train_loader[j],local_ep,loss[j])
        # print(model_list[j].state_dict()['conv1.weight'][0][0][0])
    time_2 = time.time()
    print('Host training time:'+str(time_2-time_1))
    with open(model_dir+'/log.txt','a') as file:
        file.write('Host training time:'+str(time_2-time_1)+'\n')
    # receive trained model
    mpi_receive_model_host(model_list)
    # receive training loss
    for i in range(1,size):
        loss_group = comm.recv(source=i, tag=15)
        for index in range(len(client_rank_mapping[i])):
            j = client_rank_mapping[i][index]
            loss[j] += loss_group[index]
    # print('received loss:')
    # print(loss)
        
def mpi_test_host(model_list, acc_list,class_acc_list,after_aggregation,model_dir):
    time_1 = time.time()
    if after_aggregation:
        mpi_send_model_host(model_list)
    time_2 = time.time()
    print('mpi_send model time:'+str(time_2-time_1))
    #Test the models belong to host
    for j in client_rank_mapping[0]:
        model_list[j].to(device)
        acc, class_acc = test(model_list[j], test_loader)
        acc_list[j].append(acc)
        class_acc_list[j].append(class_acc)
    time_3 = time.time()
    print('Host  test time:'+str(time_3-time_2))
    # Receive models' test result from all workers
    for i in range(1,size):
        acc_group = comm.recv(source=i, tag=13)
        # print('Received acc_list')
        # print(acc_group)
        class_acc_group = comm.recv( source=i, tag=14)
        for index in range(len(client_rank_mapping[i])):
            j = client_rank_mapping[i][index]
            acc_list[j].append(acc_group[index])
            class_acc_list[j].append(class_acc_group[index])
    time_4 = time.time()
    print('receive time:'+str(time_4-time_3))
    with open(model_dir+'/log.txt','a') as file:
        file.write('mpi_send model time:'+str(time_2-time_1)+'\n')
        file.write('Host  test time:'+str(time_3-time_2)+'\n')
        file.write('receive time:'+str(time_4-time_3)+'\n')
        
def mpi_test_rank(model_group,after_aggregation):   
    acc_group = []
    class_group = []
    if after_aggregation:
        # receive model for inference from master processes
        # recv_model_group = comm.recv(source=0, tag=11)
        for index in range(len(model_group)):
            model_group[index].load_state_dict(deserialize_model(comm.recv(source=0, tag=11)))
        #    model_group[index].load_state_dict(deserialize_model(recv_model_group[index]))
    for index in range(len(model_group)):
        model_group[index].to(device)
        acc, class_acc = test(model_group[index], test_loader)
        acc_group.append(acc)
        class_group.append(class_acc)
    # Send update back to master
    with open(model_dir+'/rank_log/log_'+str(rank)+'.txt','a') as file:
        file.write('The acc groups is:')
        file.write(str(acc_group)+'\n')
        # file.write('The class groups is\n')
        # file.write(str(class_group)+'\n')
    comm.send(acc_group, dest=0, tag=13)
    comm.send(class_group, dest=0, tag=14)
  
def Centralized_process(suffix_dir,train_loader,test_loader,num_round,local_ep):
    model = []
    test_acc = []
    class_acc_list = []
    loss = []
    optimizer = {}
    learning_rate = lr
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+suffix_dir
    write_info(model_dir)
    for i in range(1,size):
        comm.send(model_dir, dest=i, tag=7)
    for i in range(num_car):
        model.append(copy.deepcopy(global_model))
        loss.append([])
    for j in client_rank_mapping[0]:
        optimizer[j] = optim.SGD(params=model[j].parameters(), lr=learning_rate)
    for i in range(num_round):
        print('This is the round:',i)
        learning_rate, flag_lr = update_learning_rate(i, learning_rate)
        if flag_lr:
            for j in client_rank_mapping[0]:
                change_learning_rate(optimizer[j], learning_rate)
        for param_group in optimizer[j].param_groups:
            print(param_group['lr'])
        # if i %300== 0:
        #     learning_rate  = learning_rate/5
        #     for index in range(num_car):
        #         for param_group in optimizer[index].param_groups:
        #             param_group['lr'] = learning_rate
        with open(model_dir+'/log.txt','a') as file:
            file.write('This is the round:'+str(i)+'\n')
            for param_group in optimizer[j].param_groups:
                file.write('current lr is: '+str(param_group['lr'])+'\n')
        start_time = time.time()
        #carry out mpi training
        mpi_train_host(model,optimizer,train_loader,local_ep,loss,model_dir)
        #upload_central and aggregate
        w = []
        for index in range(num_car):
            w.append(copy.deepcopy(model[index].state_dict()))
        avg_w = average_weights(w,np.array(weights))
        for index in range(num_car):
            model[index].load_state_dict(copy.deepcopy(avg_w))
        end_time = time.time()
        print(f'{end_time-start_time} [sec] for this epoch')
        acc, class_acc = test(model[0], test_loader)
        test_acc.append(acc)
        class_acc_list.append(class_acc)
        print('test acc:',acc)
        print('class acc:', class_acc)
        with open(model_dir+'/log.txt','a') as file:
            file.write(f'{end_time-start_time} [sec] for this epoch'+'\n')
            file.write('test acc:'+str(acc)+'\n')
            file.write('class acc:'+str(class_acc)+'\n')
        #test global model accuracy
        with open(model_dir+'/average_acc_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+suffix_dir+'.txt','a') as file:
            file.write(str(i)+':'+str(test_acc[-1])+'\n')
        #write a code to early stop, if test acc remain unchanged for 10 rounds, then stop and return
        if i>early_stop_round and (abs(np.array(test_acc[-early_stop_round:])-test_acc[-1])<1e-10).all():
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
    optimizer = {}
    learning_rate = lr
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+suffix_dir
    pair = write_info(model_dir)
    for i in range(1,size):
        comm.send(model_dir, dest=i, tag=7)
        
    # Distribute model to all clients 
    for i in range(num_car):
        model.append(copy.deepcopy(global_model))
        # optimizer.append([])
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    for j in client_rank_mapping[0]:
        optimizer[j] = optim.SGD(params=model[j].parameters(), lr=learning_rate)
    # mpi_send_model_host(model)
    for i in range(num_round):
        print('######################################################################')
        print('This is the round:',i)
        # print('lr:',learning_rate)
        learning_rate, flag_lr = update_learning_rate(i, learning_rate)
        if flag_lr:
            for j in client_rank_mapping[0]:
                change_learning_rate(optimizer[j], learning_rate)
        for param_group in optimizer[j].param_groups:
            print(param_group['lr'])
        # if i %300== 0:
        #     learning_rate  = learning_rate/5
        #     for index in range(num_car):
        #         for param_group in optimizer[index].param_groups:
        #             param_group['lr'] = learning_rate
        with open(model_dir+'/log.txt','a') as file:
            file.write('This is the round:'+str(i)+'\n')
            for param_group in optimizer[j].param_groups:
                file.write('current lr is: '+str(param_group['lr'])+'\n')
        start_time = time.time()
        #carry out mpi training
        mpi_train_host(model,optimizer,train_loader,local_ep,loss,model_dir)
                
        # model_before_aggregation = copy.deepcopy(model)
        
        # mpi_test_host(model_before_aggregation, acc_global_before_aggregation, class_acc_list_before_aggregation,False,model_dir)

            
        # do model aggregation
        for a,b in pair[i]: 
            weighted_average_process(model[a],model[b],np.array([weights[a],weights[b]]))
        mpi_test_host(model, acc_global, class_acc_list,True,model_dir)
        end_time = time.time()
        print(f'{end_time-start_time} [sec] for this epoch')
        # print('Before/After aggregation acc:')
        # for index in range(num_car):
        #     print('car:',index,'---------------------------------------------------------------')
        #     print(acc_global_before_aggregation[index][-1],acc_global[index][-1])
        #     print(class_acc_list_before_aggregation[index][-1])
        #     print(class_acc_list[index][-1])
        #     print('Local Cache model version:')
        print('----------------------------------------------------------------------')
        print('Average test acc:',np.average(acc_global,axis=0)[-1])
        print('Variance test acc:',np.var(acc_global,axis=0)[-1])
        print('pair:',pair[i])
        with open(model_dir+'/log.txt','a') as file:
            file.write(f'{end_time-start_time} [sec] for this epoch'+'\n')
            file.write('Before/After aggregation acc:'+'\n')
            for index in range(num_car):
                file.write('car:'+str(index)+'---------------------------------------------------------------'+'\n')
                file.write(str(acc_global[index][-1])+'\n')
                # file.write(str(acc_global_before_aggregation[index][-1])+','+str(acc_global[index][-1])+'\n')
                # file.write(str(class_acc_list_before_aggregation[index][-1])+'\n')
                file.write(str(class_acc_list[index][-1])+'\n')
                file.write('Local Cache model version:'+'\n')
            file.write('----------------------------------------------------------------------'+'\n')
            file.write('Average test acc:'+str(np.average(acc_global,axis=0)[-1])+'\n')
            file.write('Variance test acc:'+str(np.var(acc_global,axis=0)[-1])+'\n')

        
        with open(model_dir+'/average_acc_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+suffix_dir+'.txt','a') as file:
            file.write(str(i)+':'+str(np.average(acc_global,axis=0)[-1])+'\t'+str(np.var(acc_global,axis=0)[-1])+'\n')
        #write a code to early stop, if test acc remain unchanged for 10 rounds, then stop and return
        if i>early_stop_round and (abs(np.average(acc_global,axis=0)[-early_stop_round:]-np.average(acc_global,axis=0)[-1])<1e-10).all():
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
    optimizer = {}
    learning_rate = lr
    fresh_class_time_table = np.zeros([num_car,num_car])
    # current_model_time_table = np.zeros(num_car)
    # current_model_combination_table = np.eye(num_car)#statistic_data#
    current_class_test = np.zeros([num_car,10])
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir
    pair = write_info(model_dir)
    for i in range(1,size):
        comm.send(model_dir, dest=i, tag=7)
        
    # Distribute model to all clients 
    for i in range(num_car):
        local_cache.append({})
        model.append(copy.deepcopy(global_model))
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    for j in client_rank_mapping[0]:
        optimizer[j] = optim.SGD(params=model[j].parameters(), lr=learning_rate)
    # mpi_send_model_host(model)
    for i in range(num_round):
        # old_model = copy.deepcopy(model)
        print('######################################################################')
        print('This is the round:',i)
        learning_rate, flag_lr = update_learning_rate(i, learning_rate)
        if flag_lr:
            clean_up_local_cache(local_cache)
            for j in client_rank_mapping[0]:
                change_learning_rate(optimizer[j], learning_rate)
        for param_group in optimizer[j].param_groups:
            print(param_group['lr'])
        with open(model_dir+'/log.txt','a') as file:
            file.write('This is the round:'+str(i)+'\n')
            for param_group in optimizer[j].param_groups:
                file.write('current lr is: '+str(param_group['lr'])+'\n')
        start_time = time.time()
        #carry out mpi training
        mpi_train_host(model,optimizer,train_loader,local_ep,loss,model_dir)
                
        # model_before_aggregation = copy.deepcopy(model)
        
        # mpi_test_host(model_before_aggregation, acc_global_before_aggregation, class_acc_list_before_aggregation,False,model_dir)
        
        for a,b in pair[i]: 
            update_model_cache_only_one(local_cache,model[a],model[b], a,b,i,cache_size, kick_out)
            
        
        # do model aggregation
        print('Updated/aggregated model time/combination:')

        for index in range(num_car):
            model[index] = cache_average_process(model[index],index,local_cache[index],weights)

        mpi_test_host(model, acc_global, class_acc_list, True,model_dir)

        end_time = time.time()
        for index in range(num_car):
            current_class_test[index] = class_acc_list[index][-1]
        # print(current_model_combination_table)
        # print(current_class_test)
        print(fresh_class_time_table)
        for index in range(num_car):
            print(class_acc_list[index][-1])
        print(f'{end_time-start_time} [sec] for this epoch')
        # print('Before/After aggregation acc:')
        # for index in range(num_car):
        #     print('car:',index,'---------------------------------------------------------------')
        #     print(acc_global_before_aggregation[index][-1],acc_global[index][-1])
        #     print(class_acc_list_before_aggregation[index][-1])
        #     print(class_acc_list[index][-1])
        #     print('Local Cache model version:')
        #     for key in local_cache[index]:
        #         print(key,local_cache[index][key]['time'])#,local_cache[index][key]['fresh_metric'])
        # print('----------------------------------------------------------------------')
        print('Average test acc:',np.average(acc_global,axis=0)[-1])
        print('Variance test acc:',np.var(acc_global,axis=0)[-1])
        print('pair:',pair[i])
        with open(model_dir+'/log.txt','a') as file:
            file.write(f'{end_time-start_time} [sec] for this epoch'+'\n')
            file.write('Before/After aggregation acc:'+'\n')
            for index in range(num_car):
                file.write('car:'+str(index)+'---------------------------------------------------------------'+'\n')
                file.write(str(acc_global[index][-1])+'\n')
                # file.write(str(acc_global_before_aggregation[index][-1])+','+str(acc_global[index][-1])+'\n')
                # file.write(str(class_acc_list_before_aggregation[index][-1])+'\n')
                file.write(str(class_acc_list[index][-1])+'\n')
                file.write('Local Cache model version:'+'\n')
                for key in local_cache[index]:
                    file.write(str(key)+':'+str(local_cache[index][key]['time'])+'\n')#,local_cache[index][key]['fresh_metric'])
            file.write('----------------------------------------------------------------------'+'\n')
            file.write('Average test acc:'+str(np.average(acc_global,axis=0)[-1])+'\n')
            file.write('Variance test acc:'+str(np.var(acc_global,axis=0)[-1])+'\n')
        
        with open(model_dir+'/average_acc_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir+'.txt','a') as file:
            file.write(str(i)+':'+str(np.average(acc_global,axis=0)[-1])+'\t'+str(np.var(acc_global,axis=0)[-1])+'\n')
        #write a code to early stop, if test acc remain unchanged for 10 rounds, then stop and return
        if i>early_stop_round and (abs(np.average(acc_global,axis=0)[-early_stop_round:]-np.average(acc_global,axis=0)[-1])<1e-10).all():
            print('early stop at round:',i)
            with open(model_dir+'/log.txt','a') as file:
                file.write('early stop at round:'+str(i)+'\n')
            break
    return loss,acc_global,class_acc_list, acc_local   , model_dir

def Decentralized_Cache_all_process(suffix_dir,train_loader,test_loader,num_round,local_ep):
    model = []
    local_cache = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = {}
    learning_rate = lr
    fresh_class_time_table = np.zeros([num_car,num_car])
    # current_model_time_table = np.zeros(num_car)
    # current_model_combination_table = np.eye(num_car)#statistic_data#
    current_class_test = np.zeros([num_car,10])
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir
    pair = write_info(model_dir)
    for i in range(1,size):
        comm.send(model_dir, dest=i, tag=7)
        
    # Distribute model to all clients 
    for i in range(num_car):
        local_cache.append({})
        model.append(copy.deepcopy(global_model))
        # optimizer.append([])
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    for j in client_rank_mapping[0]:
        optimizer[j] = optim.SGD(params=model[j].parameters(), lr=learning_rate)
    # mpi_send_model_host(model)
    
    for i in range(num_round):
        print('######################################################################')
        print('This is the round:',i)
        learning_rate, flag_lr = update_learning_rate(i, learning_rate)
        if flag_lr:
            clean_up_local_cache(local_cache)
            for j in client_rank_mapping[0]:
                change_learning_rate(optimizer[j], learning_rate)
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
        #carry out mpi training
        mpi_train_host(model,optimizer,train_loader,local_ep,loss,model_dir)
        
        # model_before_aggregation = copy.deepcopy(model)
        
        # mpi_test_host(model_before_aggregation, acc_global_before_aggregation, class_acc_list_before_aggregation,False,model_dir)
        
        for a,b in pair[i]: 
            update_model_cache(local_cache, model[a],model[b],a,b,i, cache_size, kick_out)
            
        
        # do model aggregation
        print('Updated/aggregated model time/combination:')
        for index in range(num_car):
            model[index] = cache_average_process(model[index],index,local_cache[index],weights)
        


        mpi_test_host(model, acc_global, class_acc_list, True,model_dir)


        end_time = time.time()
        

        
        for index in range(num_car):
            current_class_test[index] = class_acc_list[index][-1]
        # print(current_model_combination_table)
        # print(current_class_test)
        print(fresh_class_time_table)
        for index in range(num_car):
            print(class_acc_list[index][-1])
        print(f'{end_time-start_time} [sec] for this epoch')
        # print('Before/After aggregation acc:')
        # for index in range(num_car):
        #     print('car:',index,'---------------------------------------------------------------')
        #     print(acc_global_before_aggregation[index][-1],acc_global[index][-1])
        #     print(class_acc_list_before_aggregation[index][-1])
        #     print(class_acc_list[index][-1])
        #     print('Local Cache model version:')
        #     for key in local_cache[index]:
        #         print(key,local_cache[index][key]['time'])#,local_cache[index][key]['fresh_metric'])
        # print('----------------------------------------------------------------------')
        print('Average test acc:',np.average(acc_global,axis=0)[-1])
        print('Variance test acc:',np.var(acc_global,axis=0)[-1])
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
                # file.write(str(acc_global_before_aggregation[index][-1])+','+str(acc_global[index][-1])+'\n')
                # file.write(str(class_acc_list_before_aggregation[index][-1])+'\n')
                file.write(str(class_acc_list[index][-1])+'\n')
                file.write('Local Cache model version:'+'\n')
                for key in local_cache[index]:
                    file.write(str(key)+':'+str(local_cache[index][key]['time'])+'\n')#,local_cache[index][key]['fresh_metric'])
            file.write('----------------------------------------------------------------------'+'\n')
            file.write('Average test acc:'+str(np.average(acc_global,axis=0)[-1])+'\n')
            file.write('Variance test acc:'+str(np.var(acc_global,axis=0)[-1])+'\n')
        with open(model_dir+'/average_acc_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir+'.txt','a') as file:
           file.write(str(i)+':'+str(np.average(acc_global,axis=0)[-1])+'\t'+str(np.var(acc_global,axis=0)[-1])+'\n')
        
        
        
        
        #write a code to early stop, if test acc remain unchanged for 10 rounds, then stop and return
        if i>early_stop_round and (abs(np.average(acc_global,axis=0)[-early_stop_round:]-np.average(acc_global,axis=0)[-1])<1e-10).all():
            print('early stop at round:',i)
            with open(model_dir+'/log.txt','a') as file:
                file.write('early stop at round:'+str(i)+'\n')
            break
    return loss,acc_global,class_acc_list, acc_local, model_dir 

def Decentralized_Cache_plus_process(suffix_dir,train_loader,test_loader,num_round,local_ep):
    model = []
    local_cache = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = {}
    learning_rate = lr
    fresh_class_time_table = np.zeros([num_car,num_car])
    # current_model_time_table = np.zeros(num_car)
    # current_model_combination_table = np.eye(num_car)#statistic_data#
    current_class_test = np.zeros([num_car,10])
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir
    pair = write_info(model_dir)
    for i in range(1,size):
        comm.send(model_dir, dest=i, tag=7)
        
    # Distribute model to all clients 
    for i in range(num_car):
        local_cache.append({})
        model.append(copy.deepcopy(global_model))
        # optimizer.append([])
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    for j in client_rank_mapping[0]:
        optimizer[j] = optim.SGD(params=model[j].parameters(), lr=learning_rate)
    # mpi_send_model_host(model)
    
    for i in range(num_round):
        print('######################################################################')
        print('This is the round:',i)
        learning_rate, flag_lr = update_learning_rate(i, learning_rate)
        if flag_lr:
            clean_up_local_cache(local_cache)
            for j in client_rank_mapping[0]:
                change_learning_rate(optimizer[j], learning_rate)
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
        #carry out mpi training
        mpi_train_host(model,optimizer,train_loader,local_ep,loss,model_dir)
        
        # model_before_aggregation = copy.deepcopy(model)
        
        # mpi_test_host(model_before_aggregation, acc_global_before_aggregation, class_acc_list_before_aggregation,False,model_dir)
        # do model aggregation
            
        for a,b in pair[i]: 
            #do model aggregation first, then do the cache update
            weighted_average_process(model[a],model[b],np.array([weights[a],weights[b]]))
            update_model_cache(local_cache, model[a],model[b],a,b,i, cache_size, kick_out)
            
        
        # do model aggregation
        print('Updated/aggregated model time/combination:')
        for index in range(num_car):
            model[index] = cache_average_process(model[index],index,local_cache[index],weights)
        


        mpi_test_host(model, acc_global, class_acc_list, True,model_dir)


        end_time = time.time()
        

        
        for index in range(num_car):
            current_class_test[index] = class_acc_list[index][-1]
        # print(current_model_combination_table)
        # print(current_class_test)
        print(fresh_class_time_table)
        for index in range(num_car):
            print(class_acc_list[index][-1])
        print(f'{end_time-start_time} [sec] for this epoch')
        # print('Before/After aggregation acc:')
        # for index in range(num_car):
        #     print('car:',index,'---------------------------------------------------------------')
        #     print(acc_global_before_aggregation[index][-1],acc_global[index][-1])
        #     print(class_acc_list_before_aggregation[index][-1])
        #     print(class_acc_list[index][-1])
        #     print('Local Cache model version:')
        #     for key in local_cache[index]:
        #         print(key,local_cache[index][key]['time'])#,local_cache[index][key]['fresh_metric'])
        # print('----------------------------------------------------------------------')
        print('Average test acc:',np.average(acc_global,axis=0)[-1])
        print('Variance test acc:',np.var(acc_global,axis=0)[-1])
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
                # file.write(str(acc_global_before_aggregation[index][-1])+','+str(acc_global[index][-1])+'\n')
                # file.write(str(class_acc_list_before_aggregation[index][-1])+'\n')
                file.write(str(class_acc_list[index][-1])+'\n')
                file.write('Local Cache model version:'+'\n')
                for key in local_cache[index]:
                    file.write(str(key)+':'+str(local_cache[index][key]['time'])+'\n')#,local_cache[index][key]['fresh_metric'])
            file.write('----------------------------------------------------------------------'+'\n')
            file.write('Average test acc:'+str(np.average(acc_global,axis=0)[-1])+'\n')
            file.write('Variance test acc:'+str(np.var(acc_global,axis=0)[-1])+'\n')
        with open(model_dir+'/average_acc_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir+'.txt','a') as file:
            file.write(str(i)+':'+str(np.average(acc_global,axis=0)[-1])+'\t'+str(np.var(acc_global,axis=0)[-1])+'\n')
        
        
        
        
        #write a code to early stop, if test acc remain unchanged for 10 rounds, then stop and return
        if i>early_stop_round and (abs(np.average(acc_global,axis=0)[-early_stop_round:]-np.average(acc_global,axis=0)[-1])<1e-10).all():
            print('early stop at round:',i)
            with open(model_dir+'/log.txt','a') as file:
                file.write('early stop at round:'+str(i)+'\n')
            break
    return loss,acc_global,class_acc_list, acc_local, model_dir 

def Decentralized_Cache_test(train_loader,test_loader,num_round,local_ep):
    model = []
    local_cache = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = {}
    learning_rate = lr
    fresh_class_time_table = np.zeros([num_car,num_car])
    fresh_history = np.zeros([num_car,num_round])
    # current_model_time_table = np.zeros(num_car)
    # current_model_combination_table = np.eye(num_car)#statistic_data#
    current_class_test = np.zeros([num_car,10])
    # model_dir = 'dfl_model_all_cache'
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
    for i in range(num_round):
        # old_model = copy.deepcopy(model)
        print('######################################################################')
        print('This is the round:',i)
        learning_rate, flag_lr = update_learning_rate(i, learning_rate)
        if flag_lr:
            clean_up_local_cache(local_cache)
            for j in client_rank_mapping[0]:
                change_learning_rate(optimizer[j], learning_rate)
        start_time = time.time()
        #carry out local training
        for index in range(num_car):
                # normal_training_process(model[index],optimizer[index],train_loader[index],local_ep,loss[index])
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
            # model[index].load_state_dict(torch.load(model_dir+'/model_'+str(index)+'.pt'))
        # model_before_aggregation = copy.deepcopy(model)
        # final_test(model_before_aggregation, acc_global_before_aggregation, class_acc_list_before_aggregation)
        # print(acc_global_before_aggregation[a][-1])
        
        
        # #First put self model in own cache
        # for index in range(num_car):
        #     put_own_model_into_cache(local_cache, model,index,i)
        #information exchange: update trace, cache, diag_fisher_matrix
        
        for a,b in pair[i]: 
            update_model_cache_fresh(local_cache, model[a],model[b],a,b,i, cache_size, fresh_class_time_table, cache_update_metric, kick_out)
            # update_model_cache(local_cache, model,a,b,i, cache_size, kick_out)
            # update_model_cache_only_one(local_cache,model, a,b,i,cache_size)
        for index in range(num_car):
            if cache_update_metric =='mean':
                fresh_history[index][i] = fresh_class_time_table[index].mean()
            elif cache_update_metric == 'min':
                fresh_history[index][i] = fresh_class_time_table[index].min()
        
        # do model aggregation
        print('Updated/aggregated model time/combination:')
        for index in range(num_car):
            fresh_class_time_table[index] = cache_average_process_fresh_without_model(local_cache[index], fresh_class_time_table[index],aggregation_metric)
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
        print(fresh_class_time_table)
        # for index in range(num_car):
        #     print(class_acc_list[index][-1])
        print(f'{end_time-start_time} [sec] for this epoch')
        print('Before/After aggregation acc:')
        for index in range(num_car):
            print('car:',index,'---------------------------------------------------------------')
            # print(acc_global_before_aggregation[index][-1],acc_global[index][-1])
            # print(class_acc_list_before_aggregation[index][-1])
            # print(class_acc_list[index][-1])
            print('Local Cache model version:')
            for key in local_cache[index]:
                print(key,local_cache[index][key]['time'],local_cache[index][key]['fresh_metric'])
                
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
        # print('Average test acc:',np.average(acc_global,axis=0)[-1])
        # print('Variance test acc:',np.var(acc_global,axis=0)[-1])
        print('pair:',pair[i])
        # print('Duration:')
        # print(duration)
    return loss,[0,0,0],class_acc_list, acc_local , fresh_history


def Decentralized_fresh_Cache_process(suffix_dir,train_loader,test_loader,num_round,local_ep):
    model = []
    local_cache = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = {}
    fresh_class_time_table = np.zeros([num_car,num_car])
    learning_rate = lr
    # current_model_time_table = np.zeros(num_car)
    # current_model_combination_table = np.eye(num_car)#statistic_data#
    current_class_test = np.zeros([num_car,10])
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir
    pair = write_info(model_dir)
    for i in range(1,size):
        comm.send(model_dir, dest=i, tag=7)
        
    # Distribute model to all clients 
    for i in range(num_car):
        local_cache.append({})
        model.append(copy.deepcopy(global_model))
        # optimizer.append([])
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    for j in client_rank_mapping[0]:
        optimizer[j] = optim.SGD(params=model[j].parameters(), lr=learning_rate)
    # mpi_send_model_host(model)
    for i in range(num_round):
        # old_model = copy.deepcopy(model)
        print('######################################################################')
        print('This is the round:',i)
        learning_rate, flag_lr = update_learning_rate(i, learning_rate)
        if flag_lr:
            clean_up_local_cache(local_cache)
            for j in client_rank_mapping[0]:
                change_learning_rate(optimizer[j], learning_rate)
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
        #carry out mpi training
        mpi_train_host(model,optimizer,train_loader,local_ep,loss,model_dir)
        for index in range(num_car):
            fresh_class_time_table[index][index] = i
        # model_before_aggregation = copy.deepcopy(model)
        
        # mpi_test_host(model_before_aggregation, acc_global_before_aggregation, class_acc_list_before_aggregation,False,model_dir)
        
        for a,b in pair[i]: 
            update_model_cache_fresh(local_cache, model[a],model[b],a,b,i, cache_size, fresh_class_time_table, cache_update_metric, kick_out)

            
        
        # do model aggregation
        print('Updated/aggregated model time/combination:')
        for index in range(num_car):
            model[index], fresh_class_time_table[index] = cache_average_process_fresh(model[index],index,local_cache[index], fresh_class_time_table[index],aggregation_metric,weights)
        
        
        
        mpi_test_host(model, acc_global, class_acc_list, True,model_dir)

        end_time = time.time()

        for index in range(num_car):
            current_class_test[index] = class_acc_list[index][-1]
        # print(current_model_combination_table)
        # print(current_class_test)
        print(fresh_class_time_table)
        for index in range(num_car):
            print(class_acc_list[index][-1])
        print(f'{end_time-start_time} [sec] for this epoch')
        # print('Before/After aggregation acc:')
        # for index in range(num_car):
        #     print('car:',index,'---------------------------------------------------------------')
        #     print(acc_global_before_aggregation[index][-1],acc_global[index][-1])
        #     print(class_acc_list_before_aggregation[index][-1])
        #     print(class_acc_list[index][-1])
        #     print('Local Cache model version:')
        #     for key in local_cache[index]:
        #         print(key,local_cache[index][key]['time'],local_cache[index][key]['fresh'])
        # print('----------------------------------------------------------------------')
        print('Average test acc:',np.average(acc_global,axis=0)[-1])
        print('Variance test acc:',np.var(acc_global,axis=0)[-1])
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
                # file.write(str(acc_global_before_aggregation[index][-1])+','+str(acc_global[index][-1])+'\n')
                # file.write(str(class_acc_list_before_aggregation[index][-1])+'\n')
                file.write(str(class_acc_list[index][-1])+'\n')
                file.write('Local Cache model version:'+'\n')
                for key in local_cache[index]:
                    file.write(str(key)+':'+str(local_cache[index][key]['time'])+','+str(local_cache[index][key]['fresh'])+'\n')#,local_cache[index][key]['fresh_metric'])
            file.write('----------------------------------------------------------------------'+'\n')
            file.write('Average test acc:'+str(np.average(acc_global,axis=0)[-1])+'\n')
            file.write('Variance test acc:'+str(np.var(acc_global,axis=0)[-1])+'\n')
        with open(model_dir+'/average_acc_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir+'.txt','a') as file:
            file.write(str(i)+':'+str(np.average(acc_global,axis=0)[-1])+'\t'+str(np.var(acc_global,axis=0)[-1])+'\n')
        #write a code to early stop, if test acc remain unchanged for 10 rounds, then stop and return
        if i>early_stop_round and (abs(np.average(acc_global,axis=0)[-early_stop_round:]-np.average(acc_global,axis=0)[-1])<1e-10).all():
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
    optimizer = {}
    fresh_class_time_table = np.zeros([num_car,num_car])
    learning_rate = lr
    # current_model_time_table = np.zeros(num_car)
    # current_model_combination_table = np.eye(num_car)#statistic_data#
    current_class_test = np.zeros([num_car,10])
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir
    pair = write_info(model_dir)
    for i in range(1,size):
        comm.send(model_dir, dest=i, tag=7)
        
    # Distribute model to all clients 
    for i in range(num_car):
        local_cache.append({})
        model.append(copy.deepcopy(global_model))
        # optimizer.append([])
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    for j in client_rank_mapping[0]:
        optimizer[j] = optim.SGD(params=model[j].parameters(), lr=learning_rate)
    # mpi_send_model_host(model)
    for i in range(num_round):
        # old_model = copy.deepcopy(model)
        print('######################################################################')
        print('This is the round:',i)
        learning_rate, flag_lr = update_learning_rate(i, learning_rate)
        if flag_lr:
            clean_up_local_cache(local_cache)
            for j in client_rank_mapping[0]:
                change_learning_rate(optimizer[j], learning_rate)
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
        #carry out mpi training
        mpi_train_host(model,optimizer,train_loader,local_ep,loss,model_dir)
        for index in range(num_car):
            fresh_class_time_table[index][index] = fresh_class_time_table[index][index] + 1
        # model_before_aggregation = copy.deepcopy(model)
        
        # mpi_test_host(model_before_aggregation, acc_global_before_aggregation, class_acc_list_before_aggregation,False,model_dir)
            
        
        for a,b in pair[i]: 
            update_model_cache_fresh(local_cache, model[a],model[b],a,b,i, cache_size, fresh_class_time_table, cache_update_metric, kick_out)
            
        
        # do model aggregation
        print('Updated/aggregated model time/combination:')
        for index in range(num_car):
            model[index], fresh_class_time_table[index] = cache_average_process_fresh(model[index],local_cache[index], fresh_class_time_table[index],aggregation_metric)
        
        
        
        mpi_test_host(model, acc_global, class_acc_list, True,model_dir)

        end_time = time.time()


        for index in range(num_car):
            current_class_test[index] = class_acc_list[index][-1]
        print(fresh_class_time_table)
        for index in range(num_car):
            print(class_acc_list[index][-1])
        print(f'{end_time-start_time} [sec] for this epoch')
        print('Before/After aggregation acc:')
        # for index in range(num_car):
        #     print('car:',index,'---------------------------------------------------------------')
        #     print(acc_global_before_aggregation[index][-1],acc_global[index][-1])
        #     print(class_acc_list_before_aggregation[index][-1])
        #     print(class_acc_list[index][-1])
        #     print('Local Cache model version:')
        #     for key in local_cache[index]:
        #         print(key,local_cache[index][key]['time'],local_cache[index][key]['fresh'])
        # print('----------------------------------------------------------------------')
        print('Average test acc:',np.average(acc_global,axis=0)[-1])
        print('Variance test acc:',np.var(acc_global,axis=0)[-1])
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
                # file.write(str(acc_global_before_aggregation[index][-1])+','+str(acc_global[index][-1])+'\n')
                # file.write(str(class_acc_list_before_aggregation[index][-1])+'\n')
                file.write(str(class_acc_list[index][-1])+'\n')
                file.write('Local Cache model version:'+'\n')
                for key in local_cache[index]:
                    file.write(str(key)+':'+str(local_cache[index][key]['time'])+','+str(local_cache[index][key]['fresh'])+'\n')#,local_cache[index][key]['fresh_metric'])
            file.write('----------------------------------------------------------------------'+'\n')
            file.write('Average test acc:'+str(np.average(acc_global,axis=0)[-1])+'\n')
            file.write('Variance test acc:'+str(np.var(acc_global,axis=0)[-1])+'\n')
        with open(model_dir+'/average_acc_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir+'.txt','a') as file:
            file.write(str(i)+':'+str(np.average(acc_global,axis=0)[-1])+'\t'+str(np.var(acc_global,axis=0)[-1])+'\n')
        #write a code to early stop, if test acc remain unchanged for 10 rounds, then stop and return
        if i>early_stop_round and (abs(np.average(acc_global,axis=0)[-early_stop_round:]-np.average(acc_global,axis=0)[-1])<1e-10).all():
            print('early stop at round:',i)
            with open(model_dir+'/log.txt','a') as file:
                file.write('early stop at round:'+str(i)+'\n')
            break
    return loss,acc_global,class_acc_list, acc_local , model_dir

def Decentralized_fresh_count_Cache_process(suffix_dir, train_loader,test_loader,num_round,local_ep):
    model = []
    local_cache = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = {}
    learning_rate = lr
    fresh_class_time_table = np.zeros([num_car,num_car])
    cache_statistic_table_by_version = np.zeros([num_car,num_round])
    cache_statistic_table_by_car_history = np.zeros([num_car,num_round])
    # current_model_time_table = np.zeros(num_car)
    # current_model_combination_table = np.eye(num_car)#statistic_data#
    current_class_test = np.zeros([num_car,10])
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir
    pair = write_info(model_dir)
    for i in range(1,size):
        comm.send(model_dir, dest=i, tag=7)
        
    # Distribute model to all clients 
    for i in range(num_car):
        local_cache.append({})
        model.append(copy.deepcopy(global_model))
        # optimizer.append([])
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    for j in client_rank_mapping[0]:
        optimizer[j] = optim.SGD(params=model[j].parameters(), lr=learning_rate)
    # mpi_send_model_host(model)
    for i in range(num_round):
        # old_model = copy.deepcopy(model)
        print('######################################################################')
        print('This is the round:',i)
        learning_rate, flag_lr = update_learning_rate(i, learning_rate)
        if flag_lr:
            clean_up_local_cache(local_cache)
            for j in client_rank_mapping[0]:
                change_learning_rate(optimizer[j], learning_rate)
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
        #carry out mpi training
        mpi_train_host(model,optimizer,train_loader,local_ep,loss,model_dir)
        for index in range(num_car):
            fresh_class_time_table[index][index] = i
        # model_before_aggregation = copy.deepcopy(model)
        
        # mpi_test_host(model_before_aggregation, acc_global_before_aggregation, class_acc_list_before_aggregation,False,model_dir)
        
        for a,b in pair[i]: 
            update_model_cache_fresh_count(local_cache, model[a],model[b],a,b,i, cache_size, fresh_class_time_table, cache_statistic_table_by_version, kick_out)
            
        
        # do model aggregation
        print('Updated/aggregated model time/combination:')
        for index in range(num_car):
            # model[index], fresh_class_time_table[index] = cache_average_process_fresh(model[index],local_cache[index], fresh_class_time_table[index],aggregation_metric)
            model[index] = cache_average_process(model[index],index,local_cache[index],weights)
        
        
        
        
        mpi_test_host(model, acc_global, class_acc_list, True,model_dir)

        end_time = time.time()

        for index in range(num_car):
            current_class_test[index] = class_acc_list[index][-1]
        print(fresh_class_time_table)
        for index in range(num_car):
            print(class_acc_list[index][-1])
        print(f'{end_time-start_time} [sec] for this epoch')
        # print('Before/After aggregation acc:')
        # for index in range(num_car):
        #     print('car:',index,'---------------------------------------------------------------')
        #     print(acc_global_before_aggregation[index][-1],acc_global[index][-1])
        #     print(class_acc_list_before_aggregation[index][-1])
        #     print(class_acc_list[index][-1])
        #     print('Local Cache model version:')
        #     for key in local_cache[index]:
        #         print(key,local_cache[index][key]['time'],local_cache[index][key]['cache_score'])
        #         cache_statistic_table_by_car_history[index][i] += 1
        #         cache_statistic_table_by_version[key][i] += 1
        print('current cached update time')
        print(cache_statistic_table_by_version[:,i])
        print('----------------------------------------------------------------------')
        print('Average test acc:',np.average(acc_global,axis=0)[-1])
        print('Variance test acc:',np.var(acc_global,axis=0)[-1])
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
                # file.write(str(acc_global_before_aggregation[index][-1])+','+str(acc_global[index][-1])+'\n')
                # file.write(str(class_acc_list_before_aggregation[index][-1])+'\n')
                file.write(str(class_acc_list[index][-1])+'\n')
                file.write('Local Cache model version:'+'\n')
                for key in local_cache[index]:
                    file.write(str(key)+':'+str(local_cache[index][key]['time'])+','+str(local_cache[index][key]['cache_score'])+'\n')#,local_cache[index][key]['fresh_metric'])
            file.write('current cached update time:\n')
            file.write(str(cache_statistic_table_by_version[:,i])+'\n')
            file.write('----------------------------------------------------------------------'+'\n')
            file.write('Average test acc:'+str(np.average(acc_global,axis=0)[-1])+'\n')
            file.write('Variance test acc:'+str(np.var(acc_global,axis=0)[-1])+'\n')
        with open(model_dir+'/average_acc_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir+'.txt','a') as file:
            file.write(str(i)+':'+str(np.average(acc_global,axis=0)[-1])+'\t'+str(np.var(acc_global,axis=0)[-1])+'\n')
        #write a code to early stop, if test acc remain unchanged for 10 rounds, then stop and return
        if i>early_stop_round and (abs(np.average(acc_global,axis=0)[-early_stop_round:]-np.average(acc_global,axis=0)[-1])<1e-10).all():
            print('early stop at round:',i)
            with open(model_dir+'/log.txt','a') as file:
                file.write('early stop at round:'+str(i)+'\n')
            break
    return loss,acc_global,class_acc_list, acc_local, model_dir



def Decentralized_fresh_history_Cache_process(suffix_dir, train_loader,test_loader,num_round,local_ep):
    model = []
    local_cache = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = {}
    learning_rate = lr
    fresh_time_table = np.zeros(num_car)
    cache_statistic_table_by_version = np.zeros([num_car,num_round])
    cache_statistic_table_by_car_history = np.zeros([num_car,num_round])
    # current_model_time_table = np.zeros(num_car)
    # current_model_combination_table = np.eye(num_car)#statistic_data#
    current_class_test = np.zeros([num_car,10])
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir
    pair = write_info(model_dir)
    for i in range(1,size):
        comm.send(model_dir, dest=i, tag=7)
        
    # Distribute model to all clients 
    for i in range(num_car):
        local_cache.append({})
        model.append(copy.deepcopy(global_model))
        # optimizer.append([])
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    for j in client_rank_mapping[0]:
        optimizer[j] = optim.SGD(params=model[j].parameters(), lr=learning_rate)
    # mpi_send_model_host(model)
    for i in range(num_round):
        # old_model = copy.deepcopy(model)
        print('######################################################################')
        print('This is the round:',i)
        learning_rate, flag_lr = update_learning_rate(i, learning_rate)
        if flag_lr:
            clean_up_local_cache(local_cache)
            for j in client_rank_mapping[0]:
                change_learning_rate(optimizer[j], learning_rate)
        for param_group in optimizer[j].param_groups:
            print(param_group['lr'])
        # if i %300== 0:
        #     learning_rate  = learning_rate/5
        #     for index in range(num_car):
        #         for param_group in optimizer[index].param_groups:
        #             param_group['lr'] = learning_rate
        with open(model_dir+'/log.txt','a') as file:
            file.write('This is the round:'+str(i)+'\n')
            for param_group in optimizer[j].param_groups:
                file.write('current lr is: '+str(param_group['lr'])+'\n')
        start_time = time.time()
        #carry out mpi training
        mpi_train_host(model,optimizer,train_loader,local_ep,loss,model_dir)
        for index in range(num_car):
            fresh_time_table[index] += 1*alpha_time
        # model_before_aggregation = copy.deepcopy(model)
        
        # mpi_test_host(model_before_aggregation, acc_global_before_aggregation, class_acc_list_before_aggregation,False,model_dir)
            
        for a,b in pair[i]: 
            update_model_cache_fresh_v3(local_cache, model[a],model[b],a,b,i, cache_size, fresh_time_table,  kick_out)
            # update_model_cache(local_cache, model,a,b,i, cache_size)
            # update_model_cache_only_one(local_cache,model, a,b,i,cache_size)
            
        
        # do model aggregation
        print('Updated/aggregated model time/combination:')
      
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
        
        mpi_test_host(model, acc_global, class_acc_list,model_dir)
        
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
        print('Variance test acc:',np.var(acc_global,axis=0)[-1])
        print('pair:',pair[i])
        # print('Duration:')
        # print(duration)
        #write a code to early stop, if test acc remain unchanged for 10 rounds, then stop and return
        if i>early_stop_round and (abs(np.average(acc_global,axis=0)[-early_stop_round:]-np.average(acc_global,axis=0)[-1])<1e-10).all():
            print('early stop at round:',i)
            with open(model_dir+'/log.txt','a') as file:
                file.write('early stop at round:'+str(i)+'\n')
            break
    return loss,acc_global,class_acc_list, acc_local, model_dir

def Decentralized_best_Cache_process(suffix_dir, train_loader,test_loader,num_round,local_ep):
    model = []
    local_cache = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = {}
    fresh_class_time_table = np.zeros([num_car,num_car])
    test_score = []
    learning_rate = lr
    # current_model_time_table = np.zeros(num_car)
    # current_model_combination_table = np.eye(num_car)#statistic_data#
    current_class_test = np.zeros([num_car,10])
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir
    pair = write_info(model_dir)
    for i in range(1,size):
        comm.send(model_dir, dest=i, tag=7)
        
    # Distribute model to all clients 
    for i in range(num_car):
        local_cache.append({})
        model.append(copy.deepcopy(global_model))
        # optimizer.append([])
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    for j in client_rank_mapping[0]:
        optimizer[j] = optim.SGD(params=model[j].parameters(), lr=learning_rate)
    # mpi_send_model_host(model)
    for i in range(num_round):
        # old_model = copy.deepcopy(model)
        print('######################################################################')
        print('This is the round:',i)
        learning_rate, flag_lr = update_learning_rate(i, learning_rate)
        if flag_lr:
            clean_up_local_cache(local_cache)
            for j in client_rank_mapping[0]:
                change_learning_rate(optimizer[j], learning_rate)
        for param_group in optimizer[j].param_groups:
            print(param_group['lr'])
        # if i %300== 0:
        #     learning_rate  = learning_rate/5
        #     for index in range(num_car):
        #         for param_group in optimizer[index].param_groups:
        #             param_group['lr'] = learning_rate
        with open(model_dir+'/log.txt','a') as file:
            file.write('This is the round:'+str(i)+'\n')
            for param_group in optimizer[j].param_groups:
                file.write('current lr is: '+str(param_group['lr'])+'\n')
        start_time = time.time()
        #carry out mpi training
        mpi_train_host(model,optimizer,train_loader,local_ep,loss,model_dir)
        for index in range(num_car):
            fresh_class_time_table[index][index] = i
        # model_before_aggregation = copy.deepcopy(model)
        
        # mpi_test_host(model_before_aggregation, acc_global_before_aggregation, class_acc_list_before_aggregation,False,model_dir)
        
        for a,b in pair[i]: 
            # update_model_cache_fresh(local_cache, model,a,b,i, cache_size, fresh_class_time_table, cache_update_metric, kick_out)
            # update_model_cache(local_cache, model,a,b,i, cache_size)
            # update_model_cache_only_one(local_cache,model, a,b,i,cache_size)
            update_best_model_cache(local_cache, model[a],model[b],a,b,i, cache_size, test_score)
        
        # do model aggregation
        print('Updated/aggregated model time/combination:')
        for index in range(num_car):
            model[index] = cache_average_process(model[index],index,local_cache[index],weights)
        
        mpi_test_host(model, acc_global, class_acc_list, True,model_dir)

        end_time = time.time()

        for index in range(num_car):
            test_score[index] = copy.deepcopy(class_acc_list_before_aggregation[index][-1])
        for index in range(num_car):
            current_class_test[index] = class_acc_list[index][-1]

        for index in range(num_car):
            print(class_acc_list[index][-1])
        print(f'{end_time-start_time} [sec] for this epoch')
        # print('Before/After aggregation acc:')
        # for index in range(num_car):
        #     print('car:',index,'---------------------------------------------------------------')
        #     print(acc_global_before_aggregation[index][-1],acc_global[index][-1])
        #     print(class_acc_list_before_aggregation[index][-1])
        #     print(class_acc_list[index][-1])
        #     print('Local Cache model version:')
        #     for key in local_cache[index]:
        #         print(key,local_cache[index][key]['time'])#,local_cache[index][key]['fresh_metric'])
      
        print('----------------------------------------------------------------------')
        print('Average test acc:',np.average(acc_global,axis=0)[-1])
        print('Variance test acc:',np.var(acc_global,axis=0)[-1])
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
                # file.write(str(acc_global_before_aggregation[index][-1])+','+str(acc_global[index][-1])+'\n')
                # file.write(str(class_acc_list_before_aggregation[index][-1])+'\n')
                file.write(str(class_acc_list[index][-1])+'\n')
                file.write('Local Cache model version:'+'\n')
                for key in local_cache[index]:
                    file.write(str(key)+':'+str(local_cache[index][key]['time'])+','+str(local_cache[index][key]['cache_score'])+'\n')#,local_cache[index][key]['fresh_metric'])
            file.write('----------------------------------------------------------------------'+'\n')
            file.write('Average test acc:'+str(np.average(acc_global,axis=0)[-1])+'\n')
            file.write('Variance test acc:'+str(np.var(acc_global,axis=0)[-1])+'\n')
        with open(model_dir+'/average_acc_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+suffix_dir+'.txt','a') as file:
            file.write(str(i)+':'+str(np.average(acc_global,axis=0)[-1])+'\t'+str(np.var(acc_global,axis=0)[-1])+'\n')
        #write a code to early stop, if test acc remain unchanged for 10 rounds, then stop and return
        if i>early_stop_round and (abs(np.average(acc_global,axis=0)[-early_stop_round:]-np.average(acc_global,axis=0)[-1])<1e-10).all():
            print('early stop at round:',i)
            with open(model_dir+'/log.txt','a') as file:
                file.write('early stop at round:'+str(i)+'\n')
            break
    return loss,acc_global,class_acc_list, acc_local 
    
def ml_process(suffix_dir,num_round,local_ep):
    model = copy.deepcopy(global_model)
    acc = []
    loss = []
    optimizer = optim.SGD(params=model.parameters(), lr=lr)
    learning_rate = lr
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+str(Randomseed)+'_'+args.algorithm+suffix_dir
    
    for i in range(num_round):
        print('This is the round:',i)
        learning_rate, flag_lr = update_learning_rate(i, learning_rate)
        if flag_lr:
            change_learning_rate(optimizer, learning_rate)
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
        current_acc, class_acc = test(model, test_loader)
        acc.append(current_acc) 
        end_time = time.time()
        print(f'{end_time-start_time} [sec] for this epoch')
        print('Acc:',current_acc)
        print('Class acc:',class_acc)
        with open(model_dir+'/log.txt','a') as file:
            file.write(f'{end_time-start_time} [sec] for this epoch'+'\n')
            file.write('Acc:'+str(current_acc)+'\n')
            file.write('Class acc:'+str(class_acc)+'\n')
        with open(model_dir+'/acc_'+task+'_'+str(Randomseed)+'_'+args.algorithm+suffix_dir+'.txt','a') as file:
            file.write(str(i)+':'+str(current_acc)+'\n')
        #write a code to early stop, if test acc remain unchanged for 10 rounds, then stop and return
        if i>early_stop_round and (abs(np.array(acc[-early_stop_round:])-acc[-1])<1e-10).all():
            print('early stop at round:',i)
            with open(model_dir+'/log.txt','a') as file:
                file.write('early stop at round:'+str(i)+'\n')
            break
    return loss,acc    




if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    # comm = pkl5.Intracomm(MPI.COMM_WORLD)
    rank = comm.Get_rank()
    size = comm.Get_size()
    status = MPI.Status()
    print('Hello from rank: '+str(rank))
    #rank==0 refers to the Master Process
    if rank ==0:
        data_distribution = distribution
        print('The size is: '+str(size))
        if task == 'mnist':
            hidden_size = 64
            global_model = CNNMnist(1,10)
            #MNIST
            if distribution == 'iid':
                train_loader, test_loader, full_loader =  get_mnist_iid(num_car,batch_size,batch_size)
            elif distribution == 'non-iid':
                train_loader, test_loader, full_loader =  get_mnist_imbalance(shards_allocation, num_car, batch_size,batch_size)
            elif distribution == 'dirichlet':
                train_loader, test_loader, full_loader =  get_mnist_dirichlet(alpha, num_car,batch_size,batch_size)
                data_distribution = data_distribution+'_'+str(alpha)
            else:
                raise ValueError('Error')
        elif task == 'cifar10':
            # global_model = Cifar10CnnModel()
            # global_model = AlexNet(10)
            global_model = ResNet18()
            # cifar10
            if distribution == 'iid':
                train_loader, test_loader, full_loader =  get_cifar10_iid(num_car,batch_size,batch_size)
            elif distribution == 'non-iid':
                train_loader, test_loader, full_loader =  get_cifar10_imbalance(shards_allocation, num_car, batch_size,batch_size)
            elif distribution == 'dirichlet':
                train_loader, test_loader, full_loader =  get_cifar10_dirichlet(alpha, num_car,batch_size,batch_size)
                data_distribution = data_distribution+'_'+str(alpha)
            else:
                raise ValueError('Error')
        elif task == 'fashionmnist':
            global_model = CNNFashion_Mnist()
            # FashionMNIST
            # train_loader, test_loader, full_loader =  get_fashionmnist_dataset_iid(shards_per_user, num_car)
            if distribution == 'iid':
                train_loader, test_loader, full_loader =  get_fashionmnist_iid(num_car,batch_size,batch_size)
            elif distribution == 'non-iid':
                train_loader, test_loader, full_loader =  get_fashionmnist_imbalance(shards_allocation, num_car, batch_size,batch_size)
            elif distribution == 'dirichlet':
                train_loader, test_loader, full_loader =  get_fashionmnist_dirichlet(alpha, num_car,batch_size,batch_size)
                data_distribution = data_distribution+'_'+str(alpha)
            else:
                raise ValueError('Error')
        else:
            raise ValueError('Error')
        # global_model = AlexNet(num_classes=10)
        
        
       
        
        global_model.to(device)
        
        #Allocate clients into different groups for rank, each rank for one group
        #try to split the model into groups
        client_rank_mapping = []
        for i in range(size):
            client_rank_mapping.append([])
        for i in range(num_car):
            client_rank_mapping[i%size].append(i)
        print('client_rank_mapping:')
        print(client_rank_mapping)
        # Distribute the mapping table to all ranks
        for i in range(1,size):
            comm.send(client_rank_mapping[i], dest=i, tag=8)
        for i in range(1,size):
            comm.send(serialize_model(global_model), dest=i, tag=6)
        # Distribute loader to all clients
        for i in range(1,size):
            train_loader_group = []
            for j in client_rank_mapping[i]:
                train_loader_group.append(train_loader[j])
            comm.send(train_loader_group, dest=i, tag=9)
        for i in range(1,size):
            comm.send(test_loader, dest=i, tag=10)
            
        #statistic data distribution
        statistic_data = np.zeros([num_car,10])
        for i in range(num_car):
            for input, target in train_loader[i]:
                for item in target:
                    statistic_data[i][item] += 1
        print('Data distribution among cars:')
        print(statistic_data)
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
        # mixing_pair = []
        # mixing_table = []
        # for i in range(cache_size):
        #     mixing_pair.append([])
        # for i in range(num_car):
        #     mixing_pair[i%cache_size].append(numbers[i])
        #     mixing_table.append([])
        # for slot in range(cache_size):
        #     for key in mixing_pair[slot]:
        #         mixing_table[key] = slot
        # print('mixing table',mixing_table)
        # print('mixing pair',mixing_pair)
        
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
            loss, acc = ml_process(suffix,num_round,local_ep)
            
        elif args.algorithm == 'cfl':
            fl_loss,  fl_test_acc, model_dir = Centralized_process(suffix,train_loader,test_loader,num_round,local_ep)
        elif args.algorithm == 'dfl':
            dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local, model_dir = Decentralized_process(suffix,train_loader,test_loader,num_round,local_ep)
        elif args.algorithm == 'cache_one':
            dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local, model_dir = Decentralized_Cache_one_process(suffix,train_loader,test_loader,num_round,local_ep)
        elif args.algorithm == 'cache_all':
            dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local, model_dir = Decentralized_Cache_all_process(suffix,train_loader,test_loader,num_round,local_ep)
        elif args.algorithm == 'cache_plus':
            dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local, model_dir = Decentralized_Cache_plus_process(suffix,train_loader,test_loader,num_round,local_ep)
        elif args.algorithm == 'fresh':
            dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_loca, model_dir  = Decentralized_fresh_Cache_process(suffix,train_loader,test_loader,num_round,local_ep)
        elif args.algorithm == 'fresh_update_fresh_by_dp':
            dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local, model_dir  = Decentralized_fresh_with_datapoints_Cache_process(suffix,train_loader,test_loader,num_round,local_ep)
        elif args.algorithm == 'fresh_count':
            dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local, model_dir  = Decentralized_fresh_count_Cache_process(suffix,train_loader,test_loader,num_round,local_ep)
        elif args.algorithm == 'fresh_history':
            dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local, model_dir  = Decentralized_fresh_history_Cache_process(train_loader,test_loader,num_round,local_ep)
        elif args.algorithm == 'best_test_cache':
            dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local, model_dir  = Decentralized_best_Cache_process(train_loader,test_loader,num_round,local_ep)
        else:
            raise ValueError('Error')
        if args.algorithm == 'ml':
            print(acc)
        elif args.algorithm == 'cfl':
            print(fl_test_acc)
        else:
            print(np.average(dfl_acc_global,axis=0))
            end_time = time.time()
            total_training_time = end_time-start_time
            hours, remainder = divmod(total_training_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            # with open(model_dir+'/summary.txt','w') as file:
            with open(model_dir+'/summary_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+suffix+'.txt','a') as file:
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
     
    else:
        # Worker processes
        # Initialization
        learning_rate = lr
        optimizer = []
        model_group = []
        
        #receive the mapping table
        allocated_clients = comm.recv(source = 0, tag = 8)
        num_client = len(allocated_clients)
        global_model = deserialize_model(comm.recv(source = 0, tag = 6))
        #initialize the model
        for item in allocated_clients:
            model_group.append(copy.deepcopy(global_model))

        for index in range(num_client):
            optimizer.append(optim.SGD(params=model_group[index].parameters(), lr=learning_rate))
        # receive the dataset
        train_loader_group = comm.recv(source = 0, tag = 9)
        test_loader = comm.recv(source = 0, tag = 10)
        model_dir = comm.recv(source = 0, tag = 7)
        with open(model_dir+'/rank_log/log_'+str(rank)+'.txt','a') as file:
            file.write('Hello from rank: '+str(rank)+'\n')
            file.write('This is the allocated clients: '+str(allocated_clients)+'\n')
                
        #begin training
        for i in range(num_round):
            with open(model_dir+'/rank_log/log_'+str(rank)+'.txt','a') as file:
                file.write('This is the round: '+str(i)+'\n')
                file.write('---------------------------------------------------------------------------\n')
            #change the learning rate
            learning_rate, flag_lr = update_learning_rate(i, learning_rate)
            if flag_lr:
                for index in range(num_client):
                    change_learning_rate(optimizer[index], learning_rate)
            #train current model
            #receive new model

            # Rank 1 probes for incoming message and then receives it
            comm.probe(source=0, tag=11, status=status)
            # Get the size of the incoming message
            count = status.Get_count(MPI.INT)
            for index in range(num_client):
                model_group[index].load_state_dict(deserialize_model(comm.recv(source=0, tag=11)))
                
            # recv_model_group = comm.recv(source=0, tag=11)
            # for index in range(num_client):
            #     model_group[index].load_state_dict(deserialize_model(recv_model_group[index]))
            #     model_group[index].to(device)
            #train model
            loss = []
            for index in range(num_client):
                loss.append([])
                with open(model_dir+'/rank_log/log_'+str(rank)+'.txt','a') as file:
                    file.write('index:'+str(index)+'\n')
                    file.write('before train:\n')
                    # file.write(str(model_group[index].state_dict()['conv1.weight'][0][0][0])+'\n')
                    model_group[index].to(device)
                    normal_training_process(model_group[index],optimizer[index],train_loader_group[index],local_ep,loss[index])
                    file.write('after train:\n')
                    # file.write(str(model_group[index].state_dict()['conv1.weight'][0][0][0])+'\n')
            # send back model and loss
            # serialized_models = [serialize_model(model.state_dict()) for model in model_group]
            # comm.send(serialized_models, dest=0, tag=12)
            for index in range(num_client):
                comm.send(serialize_model(model_group[index].state_dict()), dest=0, tag=12)
            comm.send(loss, dest=0, tag=15)
            if args.algorithm == 'cfl':
                continue;
            # test the model before aggregation
            # mpi_test_rank(model_group,False)
            # test the model after aggregation
            mpi_test_rank(model_group,True)
            
    MPI.Finalize()
