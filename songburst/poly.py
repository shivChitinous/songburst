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

import numpy as np
import pandas as pd
import songburst.hvc as hvc
import songburst.izneurons as izn
import songburst.neurons as nrn
import matplotlib.pyplot as plt
import seaborn as sns

def poly_start(N_max,
            N = 100,
            n_start = None,
            p_ra = 87, #proportion chosen to match empirical measurements
            int_ra = -0.15, #strengths chosen to give chain-like activity
            ra_int = 0.5,
            ra_ra = 1,
            p_ra_ra = 11,
            p_int_ra = 15, #density - about 1/4th the empirical value
            p_ra_int = 30,
            frac = 1/2,
            del_mean_ra_ra = 3.5e-3, #in ms
            del_std_ra_ra = 2e-3,
            del_mean_ra_int = 3.5e-3,
            del_std_ra_int = 2e-3,
            del_mean_int_ra = 2e-3,
            del_std_int_ra = 1e-3,
            plot_dist = False,
            save_dist = False,
            figfile = None,
            starter = np.random.randint(1) #random number seed
            ):
    
    #calculating numbers of int and RA neurons
    p_int = 100-p_ra
    n_int = int((p_int*N)/100)
    n_ra = int(N-n_int)
    
    #calculating number of RA neurons per group
    if n_start is None:
        n_start = int(n_ra*frac)
    print("RA:", n_ra, "| int:", n_int, "| Starters:", n_start)
    
    #generating four random states
    number_rand = 6
    rn = np.empty(number_rand, dtype=object)
    for i,e in enumerate(np.arange(starter,starter+number_rand,1)):
        rn[i] = np.random.RandomState(seed=e)
    
    conn = np.array(np.zeros((N,N)))
    
    #range vectors
    neuron_ra = range(0,n_ra)
    neuron_int = range(n_ra,n_ra+n_int)
    neuron_start = range(0,n_start)
    neuron_nonstart = range(n_start,n_ra)
    
    #interneuron-ra connections
    synapses = np.array(np.meshgrid(neuron_int, neuron_ra)).T.reshape(-1,2)
    which = rn[0].choice(range(n_int*n_ra),size=int(np.round(p_int_ra*(n_int*n_ra)/100)),replace=False)
    conn[synapses[which,0],synapses[which,1]] = int_ra

    #ra-interneuron connections
    synapses = np.array(np.meshgrid(neuron_ra, neuron_int)).T.reshape(-1,2)
    which = rn[1].choice(range(n_int*n_ra),size=int(np.round(p_ra_int*(n_int*n_ra)/100)),replace=False)
    conn[synapses[which,0],synapses[which,1]] = ra_int

    #ra-ra connections
    n_nonstart = n_ra-n_start
    synapses = np.array(np.meshgrid(neuron_start, neuron_nonstart)).T.reshape(-1,2)
    which = rn[2].choice(range(n_start*n_nonstart),size=int(np.round(p_ra_ra*(n_start*n_nonstart)/100)),replace=False)
    conn[synapses[which,0],synapses[which,1]] = ra_ra
    
    conn[range(np.shape(conn)[0]), range(np.shape(conn)[0])] = 0 #no self connections
    
    dels_ra_ra = rn[3].lognormal(mu(del_mean_ra_ra,del_std_ra_ra),sig(del_mean_ra_ra,del_std_ra_ra),
                                 size=np.sum(np.abs(conn[:n_start,n_start:n_ra])>0))
    
    dels_ra_int = rn[4].lognormal(mu(del_mean_ra_int,del_std_ra_int),sig(del_mean_ra_int,del_std_ra_int), 
                                       size=np.sum(np.abs(conn[:n_ra,n_ra:])>0))
    
    dels_int_ra = rn[5].lognormal(mu(del_mean_int_ra,del_std_int_ra),sig(del_mean_int_ra,del_std_int_ra), 
                                       size=np.sum(np.abs(conn[n_ra:,:n_ra])>0))

    delays = np.zeros(np.shape(conn))
    delays[:n_start,n_start:n_ra][np.abs(conn[:n_start,n_start:n_ra])>0] = dels_ra_ra
    delays[:n_ra,n_ra:][np.abs(conn[:n_ra,n_ra:])>0] = dels_ra_int
    delays[n_ra:,:n_ra][np.abs(conn[n_ra:,:n_ra])>0] = dels_int_ra
    
    if plot_dist:
        plt.figure(figsize = (8,3)); sns.set_style('white'); 
        sns.histplot(1e3*rn[3].lognormal(mu(del_mean_ra_ra,del_std_ra_ra),sig(del_mean_ra_ra,del_std_ra_ra),
                                 size=1000),element = 'step',
                     stat='probability',color='black',binwidth=0.5,binrange=(0,10))
        sns.histplot(1e3*rn[4].lognormal(mu(del_mean_ra_int,del_std_ra_int),sig(del_mean_ra_int,del_std_ra_int),
                                 size=1000),element = 'step',stat='probability',
                     color='grey',binwidth=0.5,binrange=(0,12))
        sns.histplot(1e3*rn[5].lognormal(mu(del_mean_int_ra,del_std_int_ra),sig(del_mean_int_ra,del_std_int_ra),
                                 size=1000),element = 'step',stat='probability',
                     color='indianred',binwidth=0.5,binrange=(0,12));
        plt.legend(["$HVC_{RA} - HVC_{RA}$","$HVC_{RA} - HVC_{int}$","$HVC_{int} - HVC_{RA}$"])
        sns.despine();
        plt.xlabel("Delay (ms)"); plt.xlim(0.5,None);
        if save_dist:
            plt.savefig(figfile+"delay_distributions.pdf",bbox_inches='tight')
    
    return conn,delays,n_ra,n_int,n_start


def insert_new(neuron_typ,location,X,conn,delays,typ,n_ra,n_int,n_start,
               int_ra = -0.15, #strengths chosen to give chain-like activity
               ra_int = 0.5,
               p_int_ra = 15, #density - about half the empirical value
               p_ra_int = 30,
               hra=0,
               hint=0,
               hx=0,
               del_mean_ra_int = 3.5e-3,
               del_std_ra_int = 2e-3,
               del_mean_int_ra = 2e-3,
               del_std_int_ra = 1e-3
              ):
    
    Y = X.copy()
    if location<n_start:
        raise ValueError('location cannot be less than n_start')
    #which type
    elif neuron_typ == 'ra':
        if location>n_ra:
            raise ValueError('location for hvcra neuron cannot be less than n_ra')
        k1,k2,k3,a,b,c,d,E,vth,C = hvc.paramshvcra(hra)
    
        #modify conn
        outward = np.random.choice([0,ra_int], n_int, p=[1-(p_ra_int/100), p_ra_int/100])
        outward = np.append(np.zeros(n_ra),outward)
        inward = np.random.choice([0,int_ra], n_int, p=[1-(p_int_ra/100), p_int_ra/100])
        inward = np.append(np.zeros(n_ra),inward)
        inward = np.insert(inward,location,0)
        conn = np.insert(conn,location,outward,axis=0)
        conn = np.insert(conn,location,inward,axis=1)
        
        #modify delays
        delout = np.random.lognormal(mu(del_mean_ra_int,del_std_ra_int),sig(del_mean_ra_int,del_std_ra_int), 
                                       size=np.size(outward))*(np.abs(outward)>0)
        delin = np.random.lognormal(mu(del_mean_int_ra,del_std_int_ra),sig(del_mean_int_ra,del_std_int_ra), 
                                       size=np.size(inward))*(np.abs(inward)>0)
        delays = np.insert(delays,location,delout,axis=0)
        delays = np.insert(delays,location,delin,axis=1)
        
        #modify typ
        typ = np.insert(typ,location,'ra')
        
        #modify n_ra
        n_ra += 1
        
    elif neuron_typ == 'int': 
        if location<=n_ra:
            raise ValueError('location for interneuron cannot be less than n_ra')
        k1,k2,k3,a,b,c,d,E,vth,C = hvc.paramshvcint(hint)

        #modify conn
        outward = np.random.choice([0,int_ra], n_ra, p=[1-(p_int_ra/100), p_int_ra/100])
        outward = np.append(outward,np.zeros(n_int))
        inward = np.random.choice([0,ra_int], n_ra, p=[1-(p_ra_int/100), p_ra_int/100])
        inward = np.append(inward,np.zeros(n_int))
        inward = np.insert(inward,location,0) 
        conn = np.insert(conn,location,outward,axis=0)
        conn = np.insert(conn,location,inward,axis=1)
        
        #modify delays
        delout = np.random.lognormal(mu(del_mean_int_ra,del_std_int_ra),sig(del_mean_int_ra,del_std_int_ra), 
                                       size=np.size(outward))*(np.abs(outward)>0)
        delin = np.random.lognormal(mu(del_mean_ra_int,del_std_ra_int),sig(del_mean_ra_int,del_std_ra_int), 
                                       size=np.size(inward))*(np.abs(inward)>0)
        delays = np.insert(delays,location,delout,axis=0)
        delays = np.insert(delays,location,delin,axis=1)
        
        #modify typ
        typ = np.insert(typ,location,'int')
        
        #modify n_int
        n_int += 1
    
    #create neuron
    n = izn.IZN(k1,k2,k3,a,b,c,d,E,vth,C)
    
    #insert within dataframe at given location
    params = np.array([n,k1,k2,k3,a,b,c,d,E,vth,C])
    Y.loc[location-0.5] = np.append(params,np.array([None]*(Y.shape[1]-11)))
    Y = Y.sort_index().reset_index(drop=True)
    
    return Y,conn,delays,typ,n_ra,n_int,n_start

def prune(X,conn,delays,typ,n_ra,n_int,n_start):
    
    Y = X.copy()
    
    remover = (Y['firstburst']==0.0)
    remove = Y[remover].index
    
    conn = np.delete(conn,remove,axis=1)
    delays = np.delete(delays,remove,axis=1)
    remove = remove[(remove!=1)|(remove!=X.shape[0])]
    conn = np.delete(conn,remove,axis=0)
    delays = np.delete(delays,remove,axis=0)
    
    n_int -= np.sum(remover & (typ=='int'))
    n_ra -= np.sum(remover & (typ=='ra'))
    n_start -= np.sum(remover & (Y['typ']=='starter'))
    
    typ = np.delete(typ,remove,axis=0)
    
    Y = Y.drop(remove)
    Y = Y.reset_index(drop=True)
    
    return Y,conn,delays,typ,n_ra,n_int,n_start

def mu(mean,std):
    mu = np.log((mean**2)/np.sqrt((std**2)+(mean**2)))
    return mu


def sig(mean,std):
    sig = np.sqrt(np.log(1+(std/mean)**2))
    return sig



