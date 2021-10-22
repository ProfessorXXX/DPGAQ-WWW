
import scipy.io as sio
import numpy as np
from Memory import MemoryDNN
from actionGeneration import decomposition
import os
import time


def plot_rate(rate_his, roll_n = 50):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    rate_array = np.asarray(rate_his)
    df = pd.DataFrame(rate_his)


    mpl.style.use('ggplot')
    mean=df.rolling(roll_n, min_periods=1).mean()
    mean = mean.values
    min = df.rolling(roll_n, min_periods=1).min()[0]
    min = min.values
    max = df.rolling(roll_n, min_periods=1).max()[0]
    max = max.values
    print('b:',max)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    Q_path = os.path.join(current_dir, "Q/" + 'reward.mat')
    sio.savemat(Q_path, {'Q_mean': mean,  'max': max, 'min': min})


    plt.plot(np.arange(len(rate_array)) + 1, df.rolling(roll_n, min_periods=1).mean())
    plt.fill_between(np.arange(len(rate_array)) + 1, df.rolling(roll_n, min_periods=1).min()[0], df.rolling(roll_n, min_periods=1).max()[0], alpha=0.2)
    plt.ylabel('Normalized data transmission rate')
    plt.xlabel('Time slot t')
    plt.savefig("data/temp.png",dpi=500)
    plt.show()


def save_to_txt(rate_his, file_path):
    with open(file_path, 'w') as f:
        for rate in rate_his:
            f.write("%s \n" % rate)

if __name__ == "__main__":


    N = 30                 # agents
    n = 10000                    # slot
    K = N
    decison_mode = 'AG'    # the quantization action mode
    Memory = 1000        # Menory
    Delta = 120


    channel = sio.loadmat('./data/N_%d' %N)['input_h']
    rate = sio.loadmat('./data/N_%d' %N)['output_obj']
    channel = channel * 1000000 #6


    split_idx = int(.8* len(channel))
    num_test = min(len(channel) - split_idx, n - int(.8* n)) # training data size


    Agent_net = MemoryDNN(net = [N, 200, 100, N],
                          lr= 0.01,
                          T_i=10,
                          M=128,
                          D=Memory
                          )

    start_time=time.time()

    rate_h = []
    rate_h_r = []
    mode_h = []
    k_idx_his = []
    K_his = []
    for i in range(n):
        if i % (n//10) == 0:
           print("%0.1f"%(i/n))
        if i> 0 and i % Delta == 0:
            if Delta > 1:
                max_k = max(k_idx_his[-Delta:-1]) +1;
            else:
                max_k = k_idx_his[-1] +1;
            K = min(max_k +1, N)

        if i < n - num_test:
            i_idx = i % split_idx
        else:
            i_idx = i - n + num_test + split_idx

        h = channel[i_idx,:]


        mode_list = Agent_net.decode(h, K, decison_mode)

        r_list = []
        for m in mode_list:
            r_list.append(decomposition(h / 1000000, m)[0])

        Agent_net.encode(h, mode_list[np.argmax(r_list)])


        rate_h.append(np.max(r_list))
        rate_h_r.append(rate_h[-1] / rate[i_idx][0])
        k_idx_his.append(np.argmax(r_list))
        K_his.append(K)
        mode_h.append(mode_list[np.argmax(r_list)])


    total_time=time.time()-start_time
    Agent_net.plot_cost()
    plot_rate(rate_h_r)

    print("Averaged normalized computation rate:", sum(rate_h_r[-num_test: -1]) / num_test)
    print('Total time consumed:%s'%total_time)
    print('Average time per channel:%s'%(total_time/n))

    # save data into txt
    save_to_txt(k_idx_his, "k_i_h.txt")
    save_to_txt(K_his, "K_h.txt")
    save_to_txt(Agent_net.cost_his, "cost_his.txt")
    save_to_txt(rate_h_r, "rate_h_r.txt")
    save_to_txt(mode_h, "mode_h.txt")
