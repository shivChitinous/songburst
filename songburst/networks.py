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
                  p_int_ra = 25, #density - about 1/4 the empirical value
                  p_ra_int = 25,
                  starter = np.random.randint(1) #random number seed
                 ):
    
    #calculating numbers of int and RA neurons
    p_int = 100-p_ra
    n_ra = int(np.round(p_ra*N/100))
    n_int = N-n_ra
    
    #calculating number of RA neurons per group
    pergroup = int(n_ra/groups)
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
                   ra_int = 2,     #activity for Jin's parameters; groups=40
                   ra_ra = 2.5, 
                   p_int_ra = 25, #density according to Jin
                   p_ra_int = 26, 
                   frac = 0.7, #bifurcation parameter optimised for repeating INs
                   starter = np.random.randint(1) #random number seed
                  ): 
    
    #calculating numbers of int and RA neurons
    p_int = 100-p_ra
    n_ra = int(np.round(p_ra*N/100))
    n_int = N-n_ra
    
    #calculating number of RA neurons per group
    pergroup = int(n_ra/groups)
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
        if ((i>int(n_ra/2)) & (i<=(int(n_ra/2)+pergroup))):
            conn[i-int(pergroup*frac):i,:pergroup] = ra_ra
            conn[i-int(pergroup*frac):i,i:i+pergroup] = 0
            branch_neuron=i

    conn[i:n_ra-pergroup,i+pergroup:n_ra] = ra_ra

    conn[range(np.shape(conn)[0]), range(np.shape(conn)[0])] = 0 #no self connections
    return conn,n_ra,n_int,branch_neuron,pergroup


def plot_neuron(df,X,Neu,tmin,tmax,Iec_df=None,mod_df=None,save=False,name=None):
    
    #checking for modulation
    if ((mod_df is None) & (Iec_df is None)): num = 2
    elif ((mod_df is None) | (Iec_df is None)): num = 3
    else: num = 4
    
    #create figure
    fig,ax = plt.subplots(num,1,figsize=(5,num*4))
    
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
    if save: plt.savefig(name,dpi=300,bbox_inches="tight")
    plt.show()


def plot_raster(df,X,tmin,tmax,n_ra=0,branch_neuron=0,patches=[[0,0,0,0]],save=False,name=None,sort=False):
    
    #create figure
    fig = plt.figure(figsize=(4,8)); sns.set_style('white'); 
    #set white background
    ax=fig.add_axes([0,0,0.8,0.8])
    
    sns.despine(fig=fig,top=True, right=True, left=False, bottom=False);
    
    Y = X.copy()
    if sort is True:
        Y['firstburst'] = [np.append(df['t'][Y['N'].iloc[n].sptr==1],0)[0] for n in range(np.size(Y['N']))]
        Y.loc[Y['firstburst']==0.0,'firstburst']=float('NaN')
        Y = Y.sort_values(by=['firstburst'])
    
    #run over the neurons
    for n in range(np.shape(Y)[0]):
        x = df['t'].loc[(Y['N'].iloc[n].sptr == 1)]
        p1, = plt.eventplot(x,linewidth=1,lineoffsets=n,color='k')
    plt.xlabel('$t\ (s)$'); plt.ylabel('$neuron$')
    plt.xlim([tmin,tmax])
    
    #add lines separating neurons
    plt.axhline(branch_neuron, color='steelblue', alpha=branch_neuron!=0)
    if sort:
        plt.axhline(len(Y)-n_ra-0.5, color='crimson',alpha=bool(n_ra!=0))
    else:
        plt.axhline(n_ra, color='crimson',alpha=bool(n_ra!=0))
    
    #add patches to show inputs
    for patch in patches:
        ax.add_patch(ptc.Rectangle((patch[2], patch[0]),patch[3],patch[1],fill = True,
                               alpha = 0.1,color = 'mediumpurple',linewidth = 0))
    
    #save figure
    if save: plt.savefig(name, dpi = 800, bbox_inches='tight')
    plt.show(); sns.set()


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
