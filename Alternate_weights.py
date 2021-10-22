


import scipy.io as sio
import numpy as np

from Memory import MemoryDNN
from actionGeneration import decomposition
from demo_train import plot_rate, save_to_txt

import time


def alternate_weights(case_id=0):

    # set alternated weights
    weights=[[1,1.5,1,1.5,1,1.5,1,1.5,1,1.5],[1.5,1,1.5,1,1.5,1,1.5,1,1.5,1]]
    
    # load the input data
    if case_id == 0:
        # by defaulst, case_id = 0
        rate = sio.loadmat('./data/N_10')['output_obj']
    else:
        # alternate weights for all agents
        rate = sio.loadmat('./data/N_10_Weights')['output_obj']
    return weights[case_id], rate

if __name__ == "__main__":

    
    N = 10                     # number of agents
    n = 20000                # time slot, <= 10,000
    K = N                   # initialize K
    decoder_mode = 'AG'    # the quantization mode
    Memory = 1000          # memory
    Delta = 32             # Update interval

    # Load input data
    channel = sio.loadmat('./data/N_%d' %N)['input_h']
    rate = sio.loadmat('./data/N_%d' %N)['output_obj']
    
    # increase channel gain to close to 1 for better training; it is a trick widely adopted in deep learning
    channel = channel * 1000000

    # generate the train and test data sample index
    # data are splitted as 8:2

    split_idx = int(.8* len(channel))
    num_test = min(len(channel) - split_idx, n - int(.8* n)) # training data size
    
    
    Agent_net = MemoryDNN(net = [N, 200, 100, N],
                          lr= 0.01,
                          T_i=10,
                          M=128,
                          D=Memory
                          )

    start_time=time.time()
    
    rate_his = []
    rate_his_ratio = []
    mode_his = []
    k_i_h = []
    K_h = []
    h = channel[0,:]
    
    # initilize the weights
    weight, rate = alternate_weights(0)
    
    
    for i in range(n):
        # for dynamic number of agents
        if i ==0.6*n:
            weight, rate = alternate_weights(1)
        if i ==0.8*n:
            weight, rate = alternate_weights(0)

        if i % (n//10) == 0:
           print("%0.1f"%(i/n))
        if i> 0 and i % Delta == 0:
            if Delta > 1:
                max_k = max(k_i_h[-Delta:-1]) + 1;
            else:
                max_k = k_i_h[-1] + 1;
            K = min(max_k +1, N)

        i_idx = i
        h = channel[i_idx,:]
        
        # the action selection
        mode_list = Agent_net.decode(h, K, decoder_mode)
        
        r_list = []
        for m in mode_list:
            # only acitve agents are used to compute
            r_list.append(decomposition(h / 1000000, m, weight)[0])

        rate_his.append(np.max(r_list))
        rate_his_ratio.append(rate_his[-1] / rate[i_idx][0])
        k_i_h.append(np.argmax(r_list))
        K_h.append(K)
        mode_his.append(mode_list[np.argmax(r_list)])
        Agent_net.encode(h, mode_list[np.argmax(r_list)])
        

    total_time=time.time()-start_time
    Agent_net.plot_cost()
    plot_rate(rate_his_ratio)
 
    print("Normalized computation rate:", sum(rate_his_ratio[-num_test: -1])/num_test)
    print('Total time consumed:%s'%total_time)
    print('Average time per channel:%s'%(total_time/n))
    
    # save data into txt
    save_to_txt(k_i_h, "k_i_h.txt")
    save_to_txt(K_h, "K_h.txt")
    save_to_txt(Agent_net.cost_his, "cost_his.txt")
    save_to_txt(rate_his_ratio, "rate_h_r.txt")
    save_to_txt(mode_his, "mode_h.txt")


    
