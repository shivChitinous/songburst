# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

#for documentation read networks.md

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as ptc
import pickle
from skimage.transform import downscale_local_mean


def random_net(N,
               p_ra = 72, #percent of RA projection neurons
               p_x = 18, #percent of X projection neurons
               int_ra = -0.15, #strengths based on the microcircuit
               int_x = -0.5, 
               ra_int = 2,
               x_int = 5,
               ra_ra = 2.5,
               p_int_ra = 67, #density of connections from Mooney-Prather, 
                                #modified within 10% to prevent network from shutting down
               p_ra_int = 76,
               p_int_x = 8,
               p_x_int = 17,
               p_ra_ra = 11,
               starter = np.random.randint(1) #sets random seeds
              ):
    
    #generating random states
    number_rand = 5
    rn = np.empty(number_rand, dtype=object)
    for i,e in enumerate(np.arange(starter,starter+number_rand,1)):
        rn[i] = np.random.RandomState(seed=e)
               
    #calculating numbers of interneurons and X, RA projection neurons
    n_ra = int(np.round(p_ra*N/100))
    n_x = int(np.round(p_x*N/100))
    n_int = N-(n_ra+n_x)
    
    conn = np.array(np.zeros((N,N)))
    
    #range vectors
    neuron_ra = range(0,n_ra)
    neuron_x = range(n_ra,n_ra+n_x)
    neuron_int = range(n_ra+n_x,n_ra+n_x+n_int)
    
    #interneuron-ra connections
    synapses = np.array(np.meshgrid(neuron_int, neuron_ra)).T.reshape(-1,2)
    which = rn[0].choice(range(n_int*n_ra),size=int(np.round(p_int_ra*(n_int*n_ra)/100)),replace=False)
    conn[synapses[which,0],synapses[which,1]] = int_ra

    #ra-interneuron connections
    synapses = np.array(np.meshgrid(neuron_ra, neuron_int)).T.reshape(-1,2)
    which = rn[1].choice(range(n_int*n_ra),size=int(np.round(p_ra_int*(n_int*n_ra)/100)),replace=False)
    conn[synapses[which,0],synapses[which,1]] = ra_int

    #interneuron-x connections
    synapses = np.array(np.meshgrid(neuron_int, neuron_x)).T.reshape(-1,2)
    which = rn[2].choice(range(n_int*n_x),size=int(np.round(p_int_x*(n_int*n_x)/100)),replace=False)
    conn[synapses[which,0],synapses[which,1]] = int_x

    #x-interneuron connections
    synapses = np.array(np.meshgrid(neuron_x, neuron_int)).T.reshape(-1,2)
    which = rn[3].choice(range(n_x*n_int),size=int(np.round(p_x_int*(n_int*n_x)/100)),replace=False)
    conn[synapses[which,0],synapses[which,1]] = x_int

    #ra-ra connections
    synapses = np.array(np.meshgrid(neuron_ra, neuron_ra)).T.reshape(-1,2)
    which = rn[4].choice(range(n_ra*n_ra),size=int(np.round(p_ra_ra*(n_ra*n_ra)/100)),replace=False)
    conn[synapses[which,0],synapses[which,1]] = ra_ra

    conn[range(np.shape(conn)[0]), range(np.shape(conn)[0])] = 0 #no self connections
    conn = pd.DataFrame(conn)
    
    return conn,n_ra,n_x,n_int


def synfire_chain(N,
                  groups, #number of groups
                  p_ra = 87, #proportion chosen to match empirical measurements
                  int_ra = -0.15, #strengths chosen to give chain-like activity
                  ra_int = 2,
                  ra_ra = 2.5,
                  p_int_ra = 50, #density
                  p_ra_int = 50,
                  starter = np.random.randint(1), #random number seed
                  suppress=False
                 ):
    
    #calculating numbers of int and RA neurons
    p_int = 100-p_ra
    n_ra = int(np.round(p_ra*N/100))
    n_int = N-n_ra
    
    #calculating number of RA neurons per group
    pergroup = int(n_ra/groups)
    if not suppress:
        print("RA projection neurons per group: ", pergroup)
    
    #generating four random states
    number_rand = 2
    rn = np.empty(number_rand, dtype=object)
    for i,e in enumerate(np.arange(starter,starter+number_rand,1)):
        rn[i] = np.random.RandomState(seed=e)
    
    conn = np.array(np.zeros((N,N)))
    
    #range vectors
    neuron_ra = range(0,n_ra)
    neuron_int = range(n_ra,n_ra+n_int)
    
    if n_int>0:
        #interneuron-ra connections
        synapses = np.array(np.meshgrid(neuron_int, neuron_ra)).T.reshape(-1,2)
        which = rn[0].choice(range(n_int*n_ra),size=int(np.round(p_int_ra*(n_int*n_ra)/100)),replace=False)
        conn[synapses[which,0],synapses[which,1]] = int_ra

        #ra-interneuron connections
        synapses = np.array(np.meshgrid(neuron_ra, neuron_int)).T.reshape(-1,2)
        which = rn[1].choice(range(n_int*n_ra),size=int(np.round(p_ra_int*(n_int*n_ra)/100)),replace=False)
        conn[synapses[which,0],synapses[which,1]] = ra_int

    #ra-ra connections
    for i in range(pergroup,n_ra-pergroup,pergroup):
        conn[i-pergroup:i,i:i+pergroup] = ra_ra
    conn[i:n_ra-pergroup,i+pergroup:n_ra] = ra_ra

    conn[range(np.shape(conn)[0]), range(np.shape(conn)[0])] = 0 #no self connections
    
    return conn,n_ra,n_int,pergroup


def branched_chain(N,
                   groups, #number of groups
                   p_ra = 87, #proportion chosen to match empirical measurements
                   int_ra = -0.15, #strengths optimised to give chain-like 
                   ra_int = 1,     #activity for Jin's parameters; groups=40
                   ra_ra = 2, 
                   p_int_ra = 50, #density according to Jin
                   p_ra_int = 50, 
                   bias = 0.5, #bifurcation parameter optimised for repeating INs
                   starter = np.random.randint(1), #random number seed
                   branch_frac = 1/2, #breakpoint for the chain
                   suppress=False
                  ): 
    
    #calculating numbers of int and RA neurons
    p_int = 100-p_ra
    n_ra = int(np.round(p_ra*N/100))
    n_int = N-n_ra
    
    #calculating number of RA neurons per group
    pergroup = int(n_ra/groups)
    if not suppress:
        print("RA projection neurons per group: ", pergroup)
    
    #generating four random states
    number_rand = 2
    rn = np.empty(number_rand, dtype=object)
    for i,e in enumerate(np.arange(starter,starter+number_rand,1)):
        rn[i] = np.random.RandomState(seed=e)
    
    conn = np.array(np.zeros((N,N)))
    
    #range vectors
    neuron_ra = range(0,n_ra)
    neuron_int = range(n_ra,n_ra+n_int)
    
    if n_int>0:
        #interneuron-ra connections
        synapses = np.array(np.meshgrid(neuron_int, neuron_ra)).T.reshape(-1,2)
        which = rn[0].choice(range(n_int*n_ra),size=int(np.round(p_int_ra*(n_int*n_ra)/100)),replace=False)
        conn[synapses[which,0],synapses[which,1]] = int_ra

        #ra-interneuron connections
        synapses = np.array(np.meshgrid(neuron_ra, neuron_int)).T.reshape(-1,2)
        which = rn[1].choice(range(n_int*n_ra),size=int(np.round(p_ra_int*(n_int*n_ra)/100)),replace=False)
        conn[synapses[which,0],synapses[which,1]] = ra_int

    #ra-ra connections
    for i in range(0,n_ra-pergroup,pergroup):
        conn[i-pergroup:i,i:i+pergroup] = ra_ra
        if ((i>int(n_ra*branch_frac)) & (i<=(int(n_ra*branch_frac)+pergroup))):
            conn[i-pergroup:i,:pergroup] = ra_ra*2*(1-bias)
            conn[i-pergroup:i,i:i+pergroup] = ra_ra*2*bias
            branch_neuron=i

    conn[i:n_ra-pergroup,i+pergroup:n_ra] = ra_ra

    conn[range(np.shape(conn)[0]), range(np.shape(conn)[0])] = 0 #no self connections
    return conn,n_ra,n_int,branch_neuron,pergroup


def plot_neuron(df,X,Neu,tmin,tmax,Iec_df=None,mod_df=None,save=False,name=None,fig=None,ax=None):
    
    #checking for modulation
    if ((mod_df is None) & (Iec_df is None)): num = 2
    elif ((mod_df is None) | (Iec_df is None)): num = 3
    else: num = 4
    
    #create figure
    sns.set_style('white')
    #create figure
    if ax is None:
        canSave = True
        fig, ax = plt.subplots(num,1,figsize=(5,num*4)); 
    else:
        canSave = False
    
    #plot membrane voltage
    ax[0].plot(df['t'][(df['t']>tmin)&(df['t']<tmax)],X['N'][Neu].V[(df['t']>tmin)&(df['t']<tmax)]*1e3,
               linewidth=1);
    ax[0].set_ylim(-120,25); ax[0].set_ylabel("$V\ (mV)$")

    #plot synaptic input
    ax[1].plot(df['t'][(df['t']>tmin)&(df['t']<tmax)],X['N'][Neu].Iin[
        (df['t']>tmin)&(df['t']<tmax)],linewidth=1); ax[1].set_ylabel("$I_{synaptic}\ (pA/pF)$")

    #plot external input if specified
    if Iec_df is not None:
        ax[2].plot(df['t'][(df['t']>tmin)&(df['t']<tmax)],Iec_df[Neu][
            (df['t']>tmin)&(df['t']<tmax)],linewidth=1)
        ax[2].set_ylabel("$I_{ectopic}\ (pA/pF)$")
    
    #plot modulation if specified
    if mod_df is not None:
        ax[3-(num==3)].plot(df['t'][(df['t']>tmin)&(df['t']<tmax)],
                    mod_df[Neu][(df['t']>tmin)&(df['t']<tmax)],linewidth=1)
        ax[3-(num==3)].set_ylabel("$m$")
    
    #time on x axis at the bottom
    ax[num-1].set_xlabel("Time (s)")
    
    #save figure
    if canSave:
        sns.set()
        if save: plt.savefig(name, dpi = 300, bbox_inches='tight')
        plt.show();


def plot_raster(df,X,tmin,tmax,n_ra=0,branch_neuron=0,patches=[[0,0,0,0]],save=False,name=None,sort=False, ax=None, fig=None):
     
    sns.set_style('white')
    #create figure
    if ax is None:
        canSave = True
        fig,ax = plt.subplots(1,1,figsize=(4,8)); 
        ax.set_ylabel('$neuron$')
    else:
        canSave = False
    
    Y = X.copy()
    if sort is True:
        Y['firstburst'] = [np.append(df['t'][Y['N'].iloc[n].sptr==1],0)[0] for n in range(np.size(Y['N']))]
        Y.loc[Y['firstburst']==0.0,'firstburst']=float('NaN')
        Y = Y.sort_values(by=['firstburst'])

    #run over the neurons
    for n in range(np.shape(Y)[0]):
        x = (df['t'].loc[(Y['N'].iloc[n].sptr == 1)]-tmin)*1e3 #ms
        p1, = ax.eventplot(x,linewidth=1,lineoffsets=n,color='k')
    ax.set_xlabel('$t\ (ms)$'); 
    ax.set_xlim([0,(tmax-tmin)*1e3])

    sns.despine(fig=fig,top=True, right=True, left=False, bottom=False, ax=ax);

    #add lines separating neurons
    ax.axhline(branch_neuron, color='steelblue', alpha=branch_neuron!=0)
    if sort:
        ax.axhline(len(Y)-n_ra-0.5, color='crimson',alpha=bool(n_ra!=0))
    else:
        ax.axhline(n_ra, color='crimson',alpha=bool(n_ra!=0))

    #add patches to show inputs
    for patch in patches:
        ax.add_patch(ptc.Rectangle((patch[2], patch[0]),patch[3],patch[1],fill = True,
                               alpha = 0.1,color = 'mediumpurple',linewidth = 0))

    #save figure
    if canSave:
        sns.set()
        if save: plt.savefig(name, dpi = 800, bbox_inches='tight')
        plt.show();
    

def plot_interneurons(df,X,smoothed_activity,tmin,tmax,save=False,name=None):
    
    #create figure
    plt.figure(figsize = (12,2))
    
    #get plot mean activity
    mean_activity = np.mean(smoothed_activity,0)
    plt.plot(df['t'],mean_activity)
    plt.eventplot(np.array([df['t'][X['N'][20].sptr==1],
                            np.max(mean_activity)*np.ones(np.shape(df['t'][X['N'][20].sptr==1]))]),
                  linelengths=np.max(0.05), color = 'r', alpha =0.5)

    #plot standard deviation
    shade = np.std(smoothed_activity,0)

    lower_bound = mean_activity - shade
    upper_bound = mean_activity + shade

    plt.fill_between(df['t'], lower_bound, upper_bound, alpha=.2)

    #plot parameters
    plt.xlim([tmin,tmax]);
    plt.ylim([np.min(mean_activity)-(1*np.max(shade)),np.max(mean_activity)+(2*np.max(shade))]);
    plt.ylabel("mean activity")
    plt.xlabel("Time (s)")
    
    #save figure
    if save: plt.savefig(name, dpi=300, bbox_inches = "tight")
        
def bottle(X,filename,frame_name="X.csv"):
    #saves the objects in a file
    
    #save the objects
    f = open(filename,'wb')
    
    for n in range(X.shape[0]):
        pickle.dump(X['N'][n],f)
        
    f.close()
    
    #save the dataframe
    X.to_csv(frame_name,index=False)
    
def unbottle(filename,frame_name="X.csv"):
    #extracts the objects from a file
    
    #get dataframe
    X = pd.read_csv(frame_name)
        
    #get the objects
    f = open(filename, 'rb')
    
    #store objects in dataframe
    for n in range(X.shape[0]):
        X.iloc[n,0] = pickle.load(f)
    
    f.close()
        
    return X

def discretize(X,Iec_df,t,groups,n_ra,
               before = 3, after=1.05
              ):
    S,U,tmax,tmin,dt = Norm(X, Iec_df, n_ra, groups, t, before=before, after=after)
    t = np.arange(tmin,tmax,dt)
    if t.shape[0]==0:
        raise ValueError('Chain failed to propagate.')
    s = downscale_local_mean(S, (1,int(S.shape[1]/t.shape[0])))
    s = s*(S.shape[1]/s.shape[1])
    u = downscale_local_mean(U, (1,int(U.shape[1]/t.shape[0])),cval=float('NaN'))
    return s,u,tmax,tmin,dt

def Norm(X, Iec_df, N, groups, t, before=3, after=1.05):
    #groupsize
    k = int(np.round(N/groups))
    
    #get start of chain and end of chain
    tmin = t[np.where(X['N'].iloc[0].sptr)].min()
    tmax = after*([t[np.where(X['N'].iloc[n].sptr)].max() for n in np.arange(
    N-1,0,-1) if np.sum(X['N'].iloc[n].sptr)>0][0])
    dt = np.nanmean([t[np.where(X['N'].iloc[i+k].sptr)].min()-t[np.where(
    X['N'].iloc[i].sptr)].min() for i in range(N-k) if (
    np.sum(X['N'].iloc[i+k].sptr)>0) & (np.sum(X['N'].iloc[i].sptr)>0)])
    
    tmin = tmin-before*dt #recompute
    
    S = np.array([X['N'][i].sptr for i in range(0,N)])[:,(t>=tmin) & (t<tmax)]
    U = np.array([X['N'].iloc[i].Iin+Iec_df.iloc[:,i] for i in range(0,N)])[:,(t>=tmin) & (t<tmax)]
    U = np.array([(U[i]-np.nanmean(U[0,:]))*X['C'].iloc[i]*1e9 for i in range(U.shape[0])]) #nA
    return S,U,tmax,tmin,dt