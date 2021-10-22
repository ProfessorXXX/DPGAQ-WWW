
import numpy as np
from scipy import optimize
from scipy.special import lambertw
import scipy.io as sio
import time


def plot_h(gain_h):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl
    
    gain_array = np.asarray(gain_h)
    df = pd.DataFrame(gain_h)
    
    
    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15,8))
    rolling_intv = 20

    plt.plot(np.arange(len(gain_array))+1, df.rolling(rolling_intv, min_periods=1).mean(), 'b')
    plt.fill_between(np.arange(len(gain_array))+1, df.rolling(rolling_intv, min_periods=1).min()[0], df.rolling(rolling_intv, min_periods=1).max()[0], color = 'b', alpha = 0.2)
    plt.ylabel('Gain')
    plt.xlabel('t')
    plt.show()
    
def decomposition(h, M, weights=[]):

    o=100
    p=3
    u=0.7
    eta1=((u*p)**(1.0/3))/o
    ki=10**-26   
    eta2=u*p/10**-10
    B=2*10**6
    Vu=1.1
    epsilon=B/(Vu*np.log(2))
    x = [] # a =x[0], and tau_j = a[1:]
    
    M0=np.where(M==0)[0]
    M1=np.where(M==1)[0]
    
    hi=np.array([h[i] for i in M0])
    hj=np.array([h[i] for i in M1])
    

    if len(weights) == 0:
        weights = [1.5 if i%2==1 else 1 for i in range(len(M))]
        
    wi=np.array([weights[M0[i]] for i in range(len(M0))])
    wj=np.array([weights[M1[i]] for i in range(len(M1))])
    
    
    def sum_rate(x):
        sum1=sum(wi*eta1*(hi/ki)**(1.0/3)*x[0]**(1.0/3))
        sum2=0
        for i in range(len(M1)):
            sum2+=wj[i]*epsilon*x[i+1]*np.log(1+eta2*hj[i]**2*x[0]/x[i+1])
        return sum1+sum2

    def phi(v, j):
        return 1/(-1-1/(lambertw(-1/(np.exp( 1 + v/wj[j]/epsilon))).real))

    def p1(v):
        p1 = 0
        for j in range(len(M1)):
            p1 += hj[j]**2 * phi(v, j)

        return 1/(1 + p1 * eta2)

    def Q(v):
        sum1 = sum(wi*eta1*(hi/ki)**(1.0/3))*p1(v)**(-2/3)/3
        sum2 = 0
        for j in range(len(M1)):
            sum2 += wj[j]*hj[j]**2/(1 + 1/phi(v,j))
        return sum1 + sum2*epsilon*eta2 - v

    def tau(v, j):
        return eta2*hj[j]**2*p1(v)*phi(v,j)


    delta = 0.005
    UB = 999999999
    LB = 0
    while UB - LB > delta:
        v = (float(UB) + LB)/2
        if Q(v) > 0:
            LB = v
        else:
            UB = v

    x.append(p1(v))
    for j in range(len(M1)):
        x.append(tau(v, j))

    return sum_rate(x), x[0], x[1:]



def cd_method(h):
    N = len(h)
    M0 = np.random.randint(2,size = N)
    gain0,a,Tj= decomposition(h, M0)
    g_list = []
    M_list = []
    while True:
        for j in range(0,N):
            M = np.copy(M0)
            M[j] = (M[j]+1)%2
            gain,a,Tj= decomposition(h, M)
            g_list.append(gain)
            M_list.append(M)
        g_max = max(g_list)
        if g_max > gain0:
            gain0 = g_max
            M0 = M_list[g_list.index(g_max)]
        else:
            break
    return gain0, M0


if __name__ == "__main__":
                
    h=np.array([6.06020304235508*10**-6,1.10331933767028*10**-5,1.00213540309998*10**-7,1.21610610942759*10**-6,1.96138838395145*10**-6,1.71456339592966*10**-6,5.24563569673585*10**-6,5.89530717142197*10**-7,4.07769429231962*10**-6,2.88333185798682*10**-6])
    M=np.array([1,0,0,0,1,0,0,0,0,0])

    
    gain,a,Tj= decomposition(h, M)
    print('y:%s'%gain)
    print('a:%s'%a)
    print('Tj:%s'%Tj)
    
    # test CD method. Given h, generate the max mode
    gain0, M0 = cd_method(h)
    print('max y:%s'%gain0)
    print(M0)
    
    # test all data
    K = [10, 20, 30]                     # number of agents
    N = 1000                     # channels
    
    
    for k in K:
            # Load data
        channel = sio.loadmat('./data/data_%d' %int(k))['input_h']
        gain = sio.loadmat('./data/data_%d' %int(k))['output_obj']
    
        start_time=time.time()
        gain_his = []
        gain_his_ratio = []
        mode_his = []
        for i in range(N):
            if i % (N//10) == 0:
               print("%0.1f"%(i/N))
               
            i_idx = i 
                
            h = channel[i_idx,:]
            
            # the CD method
            gain0, M0 = cd_method(h)
            
    
            # memorize the largest reward
            gain_his.append(gain0)
            gain_his_ratio.append(gain_his[-1] / gain[i_idx][0])
    
            mode_his.append(M0)
                
        
        total_time=time.time()-start_time
        print('time_cost:%s'%total_time)
        print('average time per channel:%s'%(total_time/N))
        plot_h(gain_his_ratio)
        print("gain/max ratio: ", sum(gain_his_ratio)/N)

    































