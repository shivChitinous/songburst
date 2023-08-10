import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

#for documentation read conductance.md

def sptr_con(VI,v,i):
    #converting to spike trains
    V = VI.iloc[:,v].copy()
    sptr = V.copy()
    sp = [None]*np.shape(V)[1]
    vth = 0
    for n in range(0,np.shape(V)[1]):
        t = np.arange(0,np.shape(V)[0])
        Vx = V.iloc[:,n].copy()
        V['next'] = Vx.shift(-1)
        sptr.iloc[:,n] = (Vx >= vth) & (V['next'] < vth)
        sp[n] = t[sptr.iloc[:,n]==1].astype('float64')
    return sp

def hvc_neuron(x,plotter = False,name='hvcneuron',csv_dir = 'data/hvc_'):
    VI = pd.read_csv(csv_dir+x+'_current_clamp_VI.csv')
    i = range(0,np.shape(VI)[1],2)
    VI.iloc[:,i] = VI.iloc[:,i]*1e-12
    v = range(1,np.shape(VI)[1],2)
    VI.iloc[:,v] = VI.iloc[:,v]*1e-3
    VI['t'] = np.arange(0,np.shape(VI)[0])*1e-4
    sptr = sptr_con(VI,v,i)
    V = VI.iloc[:,v]
    I = VI.iloc[:,i]
    
    if plotter == True:
        #plotting current and voltage trace
        plt.plot(VI['t'],I*1e12); plt.ylim(-250,250); plt.ylabel('$I_{ectopic}\ (pA)$')
        plt.xlabel('$t\ (s)$'); plt.title('$HVC_{'+x+'}$'); 
        plt.savefig(name+'_injected_currents',dpi=300); plt.show()
        plt.plot(VI['t'],V*1e3); plt.ylim(-110,35); plt.ylabel('$V\ (mV)$');plt.xlabel('$t\ (s)$');
        plt.title('$HVC_{'+x+'}$');
        plt.savefig(name+'_voltage_traces',dpi=300); plt.show()
    return V,I,sptr