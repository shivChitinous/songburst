import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def U(W,u,s,alpha,j=0.1,noise=0.1):
    n = np.random.normal(loc=np.zeros(len(u)),scale=noise)
    return np.matmul(W.T,s) - alpha*np.sum(s) + j*u + n

def activation(x,step=True,K=5,O=5,B=10):
    if step:
        A = B if x>O else 0
    else:
        A = np.max(np.array([K*(x-O),0]))
    return A

def A(u,K=2,O=5,B=10,step=True):
    a = np.array([activation(x,K=K,O=O,B=B,step=step) for x in u])
    return a

def dB(u,s,alpha,j=0.1):
    #delta bias
    return j*u - alpha*np.sum(s)

def evolve(N,W,k,b,Exc,Inh,u0,B,K,O,T,step=True,j=0.1,noise=0.1,dist='N',checkpoint=0.01,percent_chain=0.5,sylls=['i','a']):
    #initialize
    s = np.zeros([N,T])
    u = np.zeros([N,T])
    I = np.zeros([N,T])
    
    if np.isscalar(Exc):
        E = Exc*np.ones(T)
    else:
        E = Exc
    alpha = Inh/k

    I[:k,1] = u0-E[0] #initial stimulus pulse
    
    #evolve activity
    for t in np.arange(1,T):
        #update membrane potential
        u[:,t] = U(W,u[:,t-1],s[:,t-1],alpha,j=j,noise=noise) + E[t] + I[:,t]
        #update activity
        s[:,t] = A(u[:,t],K=K,O=O,B=B,step=step)
        #external input interpretation of internal currents
        I[:,t] = u[:,t]-(dB(u[:,t-1],s[:,t-1],alpha,j=j)+E[t])
        
    #sequence
    seq, syllables = getSequence(s,N,k,b=b,checkpoint=checkpoint,percent_chain=percent_chain,sylls=sylls)
    
    return u,s,seq,syllables,I

def countTransitions(seq):
    seq = seq[1:].str[:1]
    c = pd.crosstab(seq, seq.shift(-1)).rename_axis(index=None, columns=None)
    return c

def probmap(Probability, excitation, inhibition, cmap='Reds', center=None, vmin=None, vmax=None, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(data=Probability, 
                xticklabels = np.round(excitation,2), 
                yticklabels = np.round(inhibition,3),
                ax=ax, cmap=cmap, center=center, vmin=vmin, vmax = vmax, square=True, **kwargs)
    plt.ylim(0,len(inhibition))
    plt.xlabel('$E$')
    plt.ylabel('$I$')
    return ax

def Ppropagation(N_trials,N,W,k,b,Exc,Inh,I0,B,K,O,T,step=True,j=0.1,noise=0.1,dist='N',checkpoint=0.1):
    p = 0
    for t in range(N_trials):
        _,s,_,_,_ = evolve(N,W,k,b,Exc,Inh,I0,B,K,O,T,step=step,j=j,noise=noise,dist=dist)
        p += np.sum(s[-int(s.shape[0]*checkpoint):,:])>0
    return p/N_trials

def getSequence(s,N,k,b=None,sylls = ['i','a'],checkpoint=0.1,percent_chain=0.5):
    
    #logical operations
    if b is not None:
        ichain = np.any(s[int(N*checkpoint):b-int(N*checkpoint),:]>0,axis=0)
        achain = np.any(s[b+int(N*checkpoint):N-int(N*checkpoint),:]>0,axis=0)
        minN = b
    else:
        achain = np.any(s[int(N*checkpoint):N-int(N*checkpoint),:]>0,axis=0)
        ichain = np.zeros(np.shape(achain)).astype('bool')
        minN = N
        
    silence = np.all(s[:,:]==0,axis=0)
    collision = np.sum(s[:,:]>0,axis=0)>k
    song = (~(silence) & ~(collision))
    iclean = ichain & song
    aclean = achain & song
    syll_labels,_ = sp.ndimage.label(iclean|aclean)
    syllables = np.zeros(np.shape(syll_labels)).astype('object')
    count = 0
    for l in np.unique(syll_labels)[1:]:
        if np.sum(syll_labels==l)>=(minN*percent_chain/k):
            if np.all(aclean[syll_labels==l]):
                syllables[syll_labels==l] = sylls[1]
            elif np.all(iclean[syll_labels==l]):
                syllables[syll_labels==l] = sylls[0]+str(count)
                count += 1
    
    coll_labels,_ = sp.ndimage.label(collision)
    for c in np.unique(coll_labels)[1:]:
        if np.sum(coll_labels==c)>=(minN*(1-5*checkpoint)/k):
            if np.all(collision[coll_labels==c]):
                syllables[coll_labels==c] = '#'
    
    
    syllables[silence] = '_'
    syllables[syllables==0.0] = ''
    uniq, index = np.unique(syllables, return_index=True)
    seq = pd.Series(uniq[index.argsort()])
    seq = seq.loc[seq!=''].reset_index(drop=True)
    return seq, syllables

def classifyBout(seq,maxi):
    seq = list(seq)
    if '#' in seq:
        bout = 'abnormal'
    elif 'a' in seq:
        bout = 'song'
    elif maxi in seq:
        bout = 'repeat'
    else:
        bout = 'abort'
    return bout

def rastermap(ax,x,T,N,cmap='Greys',xtickn=5,ytickn=5,**kwargs):
    ax = sns.heatmap(x,cmap=cmap,ax=ax,**kwargs)
    ax.set_xlim(0,T)
    ax.set_ylim(0,N);
    ax.set_xticks(np.linspace(0,T,xtickn))
    ax.set_xticklabels(np.round(np.linspace(0,T,xtickn)).astype('int'),rotation = 0)
    ax.set_yticks(np.linspace(0,N,ytickn))
    ax.set_yticklabels(np.round(np.linspace(0,N,ytickn)).astype('int'),rotation = 0)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    return ax

def weightmap(W, N, g, ax, cmap='viridis', tickn=5, tickc = 2, **kwargs):
    ax = sns.heatmap(W,cmap=cmap, square=True, 
                     cbar_kws = {'shrink':0.5,'ticks':np.round(np.unique(W),tickc),'aspect':40},ax=ax,**kwargs)
    ax.set_ylabel("presynaptic neuron");
    ax.set_xlabel("postsynaptic neuron");
    ax.set_xticks(np.linspace(0,N,tickn))
    ax.set_xticklabels(np.round(np.linspace(0,N,tickn)).astype('int'),rotation = 0)
    ax.set_yticks(np.linspace(0,N,tickn))
    ax.set_yticklabels(np.round(np.linspace(0,N,tickn)).astype('int'),rotation = 0)
    ax.set_title(r"$N="+str(N)+", g="+str(g)+"$", fontsize=10);
    return ax

def lesion(W,l=0,g=50,homo=False,seed=None):
    Wl = W.copy()
    N = np.shape(W)[0]
    np.random.seed(seed)
    if homo:
        k = int(N/g)
        gindex = np.random.choice(range(g), size = int(l*N))
        for g in np.unique(gindex):
            Wl[(g*k):np.sum(gindex==g)+(g*k),:] = 0
            Wl[:,(g*k):np.sum(gindex==g)+(g*k)] = 0
    else:
        lindex = np.random.choice(range(N), size = int(l*N), replace=False)
        Wl[lindex,:] = 0
        Wl[:,lindex] = 0
    return Wl

def TransProb(cDf, l, n_iterations, excitation, N, W, k, b, Inh, u0, B, K, O, T, step, j, noise, sylls=['g','h'], seed=None, homo=False):
    print('*',end='')
    count = len(cDf) #counter to populate Df
    g = int(N/k)
    Wl = lesion(W,l,g=g,homo=homo,seed=seed)
    for ei, exc in enumerate(excitation):
            Cl = pd.DataFrame(np.zeros([2,2]),columns=sylls, index= sylls)
            hashcount = 0
            for n in range(n_iterations):
                _,_,seql,_,_ = evolve(N,Wl,k,b,exc,Inh*(1-l),u0,B,K,O,T,step,j,noise,sylls=sylls)
                if '#' not in list(seql):
                    cl = countTransitions(seql)
                    Cl = Cl.add(cl/np.sum(cl.values),fill_value=0)
                else:
                    hashcount += 1
            ClNorm = Cl.values/(n_iterations-hashcount) if (n_iterations-hashcount)>0 else np.ones(np.shape(Cl.values))*float('NaN')
            cDf.loc[count,sylls[0]+'-'+sylls[0]] = ClNorm[0,0]
            cDf.loc[count,sylls[0]+'-'+sylls[1]] = ClNorm[0,1]
            cDf.loc[count,'lesion'] = l*100
            cDf.loc[count,'excitation'] = exc
            cDf.loc[count,'seed'] = seed
            count+=1
            if len(excitation)>10:
                if count%int(len(excitation)/10)==0:
                    print('.',end='')
            else:
                if count%(len(excitation))==0:
                    print('.',end='')
    return cDf

def biasI(biasDf,I,b,k,syllables,bias=0.38,link=0,E=0):
    if biasDf is None:
        biasDf = pd.DataFrame(columns = ['bi','ba','sylls','bout'])
    count = len(biasDf)
    boutcount = biasDf['bout'].max()+1 if ~np.isnan(biasDf['bout'].max()) else 1
    bi = I[link:(link+1)*k,link+1::int(np.round(b/k))].mean(axis=0)
    ba = I[link+b:b+(link+1)*k,link+1::int(np.round(b/k))].mean(axis=0)
    sylls = syllables[int(np.round(b/k))::int(np.round(b/k))]
    for i,s in enumerate(sylls):
        biasDf.loc[count,'bi'] = bi[i]
        biasDf.loc[count,'ba'] = ba[i]
        biasDf.loc[count,'sylls'] = sylls[i]
        biasDf.loc[count,'bout'] = boutcount
        biasDf.loc[count,'bias'] = bias
        biasDf.loc[count,'link'] = link
        biasDf.loc[count,'E'] = E
        count+=1
    biasDf['ba'] = biasDf['ba'].astype('float')
    biasDf['bi'] = biasDf['bi'].astype('float')
    biasDf['bias'] = np.round(biasDf['bias'].astype('float'),2)
    biasDf['E'] = np.round(biasDf['E'].astype('float'),2)
    biasDf['note'] = biasDf['sylls'].str[:1]
    biasDf['biasI'] = biasDf['bi']-biasDf['ba']
    return biasDf

def INno(seq):
    return int(seq.str[1:].max()) if (('a' in list(seq)) & ('#' not in list(seq))) else float('NaN')

def aborted(seq,maxi='i6'):
    if (('a' not in list(seq)) & (maxi not in list(seq)) & ('#' not in list(seq))):
        ab = 1
    elif (('a' in list(seq)) & ('#' not in list(seq))):
        ab = 0
    else:
        ab = float('NaN')
    return ab

def Et(Tst,T,A=5,s=0.1,b=0):
    time = np.arange(0,T)
    I = ((1/(1+np.exp(-s*(time-Tst)))) - (1/(1+np.exp(-s*(-Tst)))))/(1 - (1/(1+np.exp(-s*(-Tst)))))
    return (A-b)*I+b
