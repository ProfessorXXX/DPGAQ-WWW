


import scipy.io as sio
import numpy as np
from Memory import MemoryDNN
from actionGeneration import decomposition
from demo_train import plot_rate, save_to_txt

import time


def agent_off(channel, N_active, N):
    # turn off one agent
    if N_active > 5: # Support half of agents are off
        N_active = N_active - 1
        channel[:,N_active] = channel[:, N_active] / 1000000
        print("    The %dth agent is turned on."%(N_active +1))
            
    # update the expected maximum reward (computing rate)
    rate = sio.loadmat('./data/N_%d' %N_active)['output_obj']
    return channel, rate, N_active

def agent_on(channel, N_active, N):
    # turn on one agent
    if N_active < N:
        N_active = N_active + 1
        channel[:,N_active-1] = channel[:, N_active-1] * 1000000 

    rate = sio.loadmat('./data/N_%d' %N_active)['output_obj']
    return channel, rate, N_active


    

if __name__ == "__main__":

    
    N = 30                     # agents
    N_active = N               # effective agents
    N_off = 0                  # off-agents
    n = 10000                     # slot
    K = N
    decoder_mode = 'AG'    # the quantization mode
    Memory = 1024
    Delta = 32
    

    # Load data
    channel = sio.loadmat('./data/N_%d' %N)['input_h']
    rate = sio.loadmat('./data/N_%d' %N)['output_obj']

    channel = channel * 1000000
    channel_bak = channel.copy()

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
    k_idx_his = []
    K_his = []
    h = channel[0,:]

    
    for i in range(n):
        # for dynamic number of agents
        if i ==0.6*n:
            print("At slot %d:"%(i))
            channel, rate, N_active = agent_off(channel, N_active, N)
        if i ==0.65*n:
            print("At slot %d:"%(i))
            channel, rate, N_active = agent_off(channel, N_active, N)
        if i ==0.7*n:
            print("At slot%d:"%(i))
            channel, rate, N_active = agent_off(channel, N_active, N)
        if i ==0.75*n:
            print("At slot %d:"%(i))
            channel, rate, N_active = agent_off(channel, N_active, N)
        if i ==0.8*n:
            print("At slot %d:"%(i))
            channel, rate, N_active = agent_on(channel, N_active, N)
        if i ==0.85*n:
            print("At slot %d:"%(i))
            channel, rate, N_active = agent_on(channel, N_active, N)
        if i ==0.9*n:
            print("At slot %d:"%(i))
            channel, rate, N_active = agent_on(channel, N_active, N)
            channel, rate, N_active = agent_on(channel, N_active, N)
        if i == 0.95*n:
            print("At slot %d:"%(i))
            channel, rate, N_active = agent_off(channel, N_active, N)
            channel, rate, N_active = agent_off(channel, N_active, N)
                
        if i % (n//10) == 0:
           print("%0.1f"%(i/n))
        if i> 0 and i % Delta == 0:
            # index counts from 0
            if Delta > 1:
                max_k = max(k_idx_his[-Delta:-1]) +1; 
            else:
                max_k = k_idx_his[-1] +1; 
            K = min(max_k +1, N)
        
        i_idx = i
        h = channel[i_idx,:]
        
        # the action selection
        mode_list = Agent_net.decode(h, K, decoder_mode)
        
        r_list = []
        for m in mode_list:
            r_list.append(decomposition(h[0:N_active] / 1000000, m[0:N_active])[0])

        # store largest reward
        rate_his.append(np.max(r_list))
        rate_his_ratio.append(rate_his[-1] / rate[i_idx][0])
        k_idx_his.append(np.argmax(r_list))
        K_his.append(K)
        mode_his.append(mode_list[np.argmax(r_list)])
        Agent_net.encode(h, mode_list[np.argmax(r_list)])
        

    total_time=time.time()-start_time
    Agent_net.plot_cost()
    plot_rate(rate_his_ratio)
 
    print("Normalized computation rate:", sum(rate_his_ratio[-num_test: -1])/num_test)
    print('Total time consumed:%s'%total_time)
    print('Average time per channel:%s'%(total_time/n))
    
    # save data into txt
    save_to_txt(k_idx_his, "k_i_h.txt")
    save_to_txt(K_his, "K_h.txt")
    save_to_txt(Agent_net.cost_his, "cost_his.txt")
    save_to_txt(rate_his_ratio, "rate_h_r.txt")
    save_to_txt(mode_his, "mode_h.txt")


    
