import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from stable_trunc_gaussian import TruncatedGaussian as TG
from torch import tensor as t
from scipy.optimize import brentq
from scipy.optimize import newton

def U(W,u,s,epsilon,iota,j=0.25,noise=0.25):
    n = np.random.normal(loc=np.zeros(len(u)),scale=noise)
    return epsilon*np.matmul(W.T,s) - iota*np.sum(s) + j*u + n

def activation(x,step=True,ramp=True,K=10,O=2,B=5):
    if ramp:
        A = np.max(np.array([K*(x-O),0]))
    else:
        A = np.inf if x>O else 0
    if step:
        A = np.min(np.array([A,B]))
    return A

def Factivation(x,step=True,ramp=True,K=10,O=2,B=5,noise=0.25):
    LD = O
    HD = np.inf if not step else ((O + B/K) if ramp else O)
    if LD == HD:
        S1 = 0
    else:
        aZ = (LD-x)/(np.sqrt(2)*noise)
        bZ = (HD-x)/(np.sqrt(2)*noise)
        S1 = K*((noise/np.sqrt(2*np.pi))*(np.exp(-aZ**2)-np.exp(-bZ**2)) + (x-O)/2*(sp.special.erf(bZ)-sp.special.erf(aZ)))
    
    S2 = (B/2)*(1 - (sp.special.erf((HD-x)/(np.sqrt(2)*noise))))
    return S1+S2

# Derivative of F(mu) with respect to mu for Newton-Raphson method #TODO: modify for ramp and step
def dFactivation(x,K=10,O=2,B=5,noise=0.25):
    S1 = K*(noise/np.sqrt(2 * np.pi))*(
        ((O - x) / (noise ** 2)) * np.exp(-((O - x) ** 2) / (2 * noise ** 2)) -
        ((O + K**-1 * B - x) / (noise ** 2)) * np.exp(-((O + K**-1 * B - x) ** 2) / (2 * noise ** 2)))
    
    S2 = K / 2 * (
        sp.special.erf((O + K**-1 * B - x) / (np.sqrt(2) * noise)) -
        sp.special.erf((O - x) / (np.sqrt(2) * noise)))
    
    S3 = (x - O) / 2 * (
        -2 / (np.sqrt(np.pi) * noise) * np.exp(-((O + K**-1 * B - x) ** 2) / (2 * noise ** 2)) +
        2 / (np.sqrt(np.pi) * noise) * np.exp(-((O - x) ** 2) / (2 * noise ** 2)))
    
    S4 = -(B / 2) * (2 / (np.sqrt(np.pi) * noise)) * np.exp(-((O + K**-1 * B - x) ** 2) / (2 * noise ** 2))
    
    return S1 + S2 + S3 + S4

# Function to find root (inverse) of F(mu) - F_target = 0 using Newton-Raphson
def Finverse(S,step=True,ramp=True,K=10,O=2,B=5,noise=0.25,ansatz=1.5):
    if noise>0:
        if (S>0)&(S<B):
            func = lambda x: Factivation(x,step=step,ramp=ramp,K=K,O=O,B=B,noise=noise) - S
            dfunc = lambda x: dFactivation(x,step=step,ramp=ramp,K=K,O=O,B=B,noise=noise)
            #ansatz is improved initial guess based on previous solution
            try:
                mu = newton(func, ansatz, fprime=dfunc)
            except RuntimeError:
                # If Newton-Raphson fails, fallback to Brent's method
                mu = brentq(func, -10, 10)  # Adjust bounds as necessary
            return mu
        elif S<=0.0:
            return 0.0
        elif S>=float(B):
            return float(B)
    else:
        raise ValueError('F is not invertible')

def A(u,step=True,ramp=True,K=10,O=2,B=5,noise=0.25):
    if noise>0:
        a = Factivation(u,step=step,ramp=ramp,K=K,O=O,B=B,noise=noise)
    else:
        a = np.array([activation(x,step=step,ramp=ramp,K=K,O=O,B=B) for x in u])
    return a

def At(u0,T,Exc,Inh,D=0,step=True,ramp=True,K=10,O=2,B=5,j=0.25,noise=0.25):
    u = np.zeros(T+1)
    u[0] = u0
    u[1] = (Exc-Inh)*Factivation(u0,step=step,ramp=ramp,K=K,O=O,B=B,noise=noise) + D
    for t in np.arange(1,T):
        c1 = -np.sum([(j**i)*Inh*Factivation(u[t-1],step=step,ramp=ramp,K=K,O=O,B=B,noise=noise) for i in np.arange(1,t+1)])
        c2 = D*(1-(j**(t+1)))/(1-j)
        c3 = (Exc-Inh)*Factivation(u[t],step=step,ramp=ramp,K=K,O=O,B=B,noise=noise)
        u[t+1] = c1+c2+c3
    return u

def dB(u,s,iota,j=0.25):
    #delta bias
    return j*u - iota*np.sum(s)

def evolve(N,W,k,b,Exc,Inh,u0,D,step=True,ramp=True,K=10,O=1.5,B=5,T=50,j=0.25,noise=0.25,checkpoint=0.12,percent_chain=0.3,sylls=['i','a'],seed=None,excitability=False,excitability_factor=0.1,receptor_lesion=False,**kwargs):
    np.random.seed(seed)
    #initialize
    s = np.zeros([N,T])
    u = np.zeros([N,T])
    I = np.zeros([N,T])

    #convert into time series
    Bdrive = B*np.ones(T).copy()
    Kdrive = K*np.ones(T).copy()
    Ddrive = D*np.ones(T).copy()

    if excitability:
        if receptor_lesion:
            _, _, Ddrive = modulateActivation(Exc,Inh,Ddrive,Kdrive,Bdrive,j=j,excitability_factor=excitability_factor)
        else:
            Kdrive, Bdrive, Ddrive = modulateActivation(Exc,Inh,Ddrive,Kdrive,Bdrive,j=j,excitability_factor=excitability_factor)

    I[:k,1] = u0-Ddrive[0] #initial stimulus pulse

    iota = Inh/k #normalize by k
    epsilon = Exc/k
    
    #evolve activity
    for t in np.arange(1,T):
        #update membrane potential
        u[:,t] = U(W,u[:,t-1],s[:,t-1],epsilon,iota,j=j,noise=noise) + Ddrive[t] + I[:,t]
        #update activity
        s[:,t] = A(u[:,t],K=Kdrive[t],O=O,B=Bdrive[t],step=step,ramp=ramp)
        #external input interpretation of internal exitatory currents
        I[:,t] = u[:,t]-(dB(u[:,t-1],s[:,t-1],iota,j=j)+Ddrive[t])
        
    #sequence
    seq, syllables = getSequence(s,N,k,b=b,checkpoint=checkpoint,percent_chain=percent_chain,sylls=sylls,B=B,**kwargs)
    
    return u,s,seq,syllables,I

def countTransitions(seq,nsyls=2):
    upto = (nsyls + 1) if nsyls is not None else None
    seq = seq[1:upto].str[:1]
    c = pd.crosstab(seq, seq.shift(-1)).rename_axis(index=None, columns=None)
    return c

def probmap(Probability, excitation, inhibition, cmap='Reds', center=None, vmin=None, vmax=None, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(data=Probability, 
                xticklabels = np.round(excitation,2), 
                yticklabels = np.round(inhibition,3),
                ax=ax, cmap=cmap, center=center, vmin=vmin, vmax = vmax, square=True, **kwargs)
    ax.set_ylim(0,len(inhibition))
    ax.set_xlabel('$E$')
    ax.set_ylabel('$I$')
    return ax

def Ppropagation(N_trials,N,W,k,b,Exc,Inh,u0,Drive,step=True,ramp=True,K=10,O=1.5,B=5,T=50,j=0.25,noise=0.25,**kwargs):
    p = 0
    for t in range(N_trials):
        _,_,seq,_,_ = evolve(N,W,k,b,Exc,Inh,u0,Drive,step=step,ramp=ramp,K=K,O=O,B=B,T=T,j=j,noise=noise,**kwargs)
        p += seq.str.contains('a').sum()
    return p/N_trials

def getSequence(s,N,k,b=None,checkpoint=0.12,percent_chain=0.3,sylls = ['i','a'],B=5,O=2):
    
    #clean up sequence for K!=\infty
    s[s<=O] = 0
    s[s>O] = B

    #logical operations
    if b is not None:
        ichain = np.any(s[int(N*checkpoint):b-int(N*checkpoint),:]>0,axis=0)
        achain = np.any(s[b+int(N*checkpoint):N-int(N*checkpoint),:]>0,axis=0)
        minN = b
    else:
        achain = np.any(s[int(N*checkpoint):N-int(N*checkpoint),:]==B,axis=0)
        ichain = np.zeros(np.shape(achain)).astype('bool')
        minN = N

    silence = np.all(s[:,:]==0,axis=0)
    collision = np.sum(s[:,:]==B,axis=0)>k
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

def weightmap(W, N, k, ax, cmap='viridis', tickn=5, tickc = 2, inset=False, **kwargs):
    ax = sns.heatmap(W,cmap=cmap, square=True, 
                     cbar_kws = {'shrink':0.5,'ticks':np.round(np.unique(W),tickc),'aspect':40},ax=ax,**kwargs)
    ax.set_xticks(np.linspace(0,N,tickn))
    ax.set_xticklabels(np.round(np.linspace(0,N,tickn)).astype('int'),rotation = 0)
    ax.set_yticks(np.linspace(0,N,tickn))
    ax.set_yticklabels(np.round(np.linspace(0,N,tickn)).astype('int'),rotation = 0)
    if not inset:
        ax.set_ylabel("presynaptic neuron");
        ax.set_xlabel("postsynaptic neuron");
        ax.set_title(r"$N="+str(N)+", k="+str(k)+"$", fontsize=10);
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

def TransProb(cDf, l, n_iterations, N, W, k, b, Exc, Inh, u0, drive, step, ramp, K, O, B, T, j, noise, sylls=['i','a'], seed=None, homo=False, conditional=True, nsylsCount=None, **kwargs):
    print('*',end='')
    count = len(cDf) #counter to populate Df
    g = int(N/k)
    Wl = lesion(W,l,g=g,homo=homo,seed=seed)
    for di, D in enumerate(drive):
            Cl = pd.DataFrame(np.zeros([2,2]),columns=sylls, index= sylls)
            hashcount = 0
            for n in range(n_iterations):
                _,_,seql,_,_ = evolve(N,Wl,k,b,Exc,Inh*(1-l),u0,D,step=step,ramp=ramp,K=K,O=O,B=B,T=T,j=j,noise=noise,sylls=sylls,**kwargs)
                if conditional:
                    if '#' not in list(seql):
                        cl = countTransitions(seql,nsyls=nsylsCount)
                        Cl = Cl.add(cl/np.sum(cl.values),fill_value=0)
                    else:
                        hashcount += 1
                else:
                    cl = countTransitions(seql,nsyls=nsylsCount)
                    Cl = Cl.add(cl/np.sum(cl.values),fill_value=0)
            ClNorm = Cl/(n_iterations-hashcount) if (n_iterations-hashcount)>0 else pd.DataFrame(np.ones([2,2]),columns=sylls, index= sylls)*float('NaN')
            cDf.loc[count,sylls[0]+'-'+sylls[0]] = ClNorm.loc[sylls[0],sylls[0]]
            cDf.loc[count,sylls[0]+'-'+sylls[1]] = ClNorm.loc[sylls[0],sylls[1]]
            cDf.loc[count,'lesion'] = l*100
            cDf.loc[count,'drive'] = D
            cDf.loc[count,'seed'] = seed
            count+=1
            if len(drive)>10:
                if count%int(len(drive)/10)==0:
                    print('.',end='')
            else:
                print('.',end='')
    return cDf

def biasI(biasDf,I,b,k,syllables,bias=0.35,link=0,D=0,rounding=2):
    if biasDf is None:
        biasDf = pd.DataFrame(columns = ['bi','ba','sylls','bout','bias','link','D'])
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
        biasDf.loc[count,'bias'] = np.round(bias,rounding)
        biasDf.loc[count,'link'] = link
        biasDf.loc[count,'D'] = np.round(D,rounding)
        count+=1
    biasDf['ba'] = biasDf['ba'].astype('float')
    biasDf['bi'] = biasDf['bi'].astype('float')
    biasDf['bias'] = np.round(biasDf['bias'].astype('float'),2)
    biasDf['D'] = np.round(biasDf['D'].astype('float'),2)
    biasDf['note'] = biasDf['sylls'].str[:1]
    biasDf['biasI'] = biasDf['bi']-biasDf['ba']
    return biasDf

def INno(seq):
    return int(seq.str[1:].max()) if (('a' in list(seq)) & ('#' not in list(seq)))+1 else float('NaN')

def aborted(seq,maxi='i6'):
    if (('a' not in list(seq)) & (maxi not in list(seq)) & ('#' not in list(seq))):
        ab = 1
    elif (('a' in list(seq)) & ('#' not in list(seq))):
        ab = 0
    else:
        ab = float('NaN')
    return ab

def modulateActivation(Exc,Inh,D,K,B,j=0.25,excitability_factor=0.1):
    maxF = (Exc-Inh/(1-j))
    scaleA = 1 + D/((1-j)*B*maxF)*excitability_factor
    return K*scaleA, B*scaleA, D*(1-excitability_factor)

def uStar(u,Exc,Inh,D=0,step=True,ramp=True,K=10,O=1.5,B=5,j=0.25,noise=0.25,drive='extrinsic',excitability_factor = 0.1):
    if drive=='extrinsic':
        return (Exc-Inh-(Inh*j)/(1-j))*Factivation(u,step=step,ramp=ramp,K=K,O=O,B=B,noise=noise) + D/(1-j)
    elif drive=='intrinsic':
        return ((Exc+D/((1-j)*B))-Inh-(Inh*j)/(1-j))*Factivation(u,step=step,ramp=ramp,K=K,O=O,B=B,noise=noise)
    elif drive=='excitability':
        Kstar,Bstar,Dstar = modulateActivation(Exc,Inh,D,K,B,j=j,excitability_factor=excitability_factor)
        return (Exc-Inh-(Inh*j)/(1-j))*Factivation(u,step=step,ramp=ramp,K=Kstar,O=O,B=Bstar,noise=noise) + Dstar/(1-j)

def Dt(Tst,T,A=5,s=0.1,b=0):
    time = np.arange(0,T)
    I = ((1/(1+np.exp(-s*(time-Tst)))) - (1/(1+np.exp(-s*(-Tst)))))/(1 - (1/(1+np.exp(-s*(-Tst)))))
    return (A-b)*I+b

def Flow(l1,lb,Exc,Inh,D=0,step=True,ramp=True,K=10,O=1.5,B=5,j=0.25,noise=0.25):
    l1p = D/(1-j) + (Exc-Inh/(1-j))*Factivation(l1,step=step,ramp=ramp,K=K,O=O,B=B,noise=noise) - (Inh/(1-j))*Factivation(lb,step=step,ramp=ramp,K=K,O=O,B=B,noise=noise)
    lbp = D/(1-j) + (Exc-Inh/(1-j))*Factivation(lb,step=step,ramp=ramp,K=K,O=O,B=B,noise=noise) - (Inh/(1-j))*Factivation(l1,step=step,ramp=ramp,K=K,O=O,B=B,noise=noise)
    return l1p, lbp

def forwardPass(s0,order,Exc=1.5,Inh=0.25,D=0,step=True,ramp=True,K=10,O=2,B=5,j=0.25,noise=0.25,ansatz=1.5,drive='extrinsic'):
    S = np.zeros([order,order]) ##cast into 32 bit float
    S[-1,0] = float(s0)
    for t in range(1,order):
        Inh_term = Inh*np.sum(S[:,t-1])
        Exc_term = Exc*sp.ndimage.shift(S[:,t-1],-1)
        if drive=='extrinsic':
            Mem_term = np.array([j*Finverse(st,step=step,ramp=ramp,K=K,O=O,B=B,noise=noise) for st in S[:,t-1]])
            S[:,t] = Factivation(Mem_term + Exc_term - Inh_term + D,step=step,ramp=ramp,K=K,O=O,B=B,noise=noise)
        elif drive=='excitability':
            Mem_term = np.array([j*Finverse(st,step=step,ramp=ramp,K=K,O=O-D,B=B,noise=noise) for st in S[:,t-1]])
            S[:,t] = Factivation(Mem_term + Exc_term - Inh_term,step=step,ramp=ramp,K=K,O=O-D,B=B,noise=noise)
    return S

def Et(t, a0, a1, tau, slope):

    # Define the error function with the specified parameters
    scale_factor = (a1 - a0) / 2
    shift_factor = (a1 + a0) / 2

    # Adjust the error function to fit the desired parameters
    return scale_factor * sp.special.erf((t - tau) * slope) + shift_factor

def riseToSong(myDrive,myDriveParams,cDf,dvalSub,time,b,k,samples=100):
    PaT = np.interp(myDrive[int(b/k)::int(b/k)],dvalSub,cDf['i-a'].values)
    PiT = np.interp(myDrive[int(b/k)::int(b/k)],dvalSub,cDf['i-i'].values)
    timeKicks = time[int(b/k)::int(b/k)]
    probabilities = np.zeros(np.shape(PaT))
    for t in range(len(PaT)):
        Iprob = 1
        for i in range(t-1):
            if i>0:
                Iprob *= PiT[i]
        probabilities[t] = PaT[t]*Iprob
    riseToSong = np.array([Et(np.arange(timeKicks[-n]-np.max(timeKicks),timeKicks[-n],1), a0= 0, a1=myDriveParams[0], 
        tau = myDriveParams[1], slope=myDriveParams[2]) for n in np.arange(1,len(timeKicks)+1)])
    return np.repeat(riseToSong, np.round(np.flip(probabilities)*samples).astype(int), axis=0)

'''
def sStar(s,Exc,Inh,j,D=0,B=5,K=2,step=True,ramp=False,O=1.5,noise=0,drive='extrinsic',ansatz=2.5,order=1):
    if drive=='extrinsic':
        return Factivation(np.array([j*Finverse(st,step=step,ramp=ramp,K=K,O=O,B=B,noise=noise,ansatz=ansatz) +Exc*st - Inh*st*order + D for st in s]),step=step,ramp=ramp,K=K,O=O,B=B,noise=noise)
    elif drive=='excitability':
        return Factivation(np.array([j*Finverse(st,step=step,ramp=ramp,K=K,O=O-D,B=B,noise=noise,ansatz=ansatz) +Exc*st - Inh*st*order for st in s]),step=step,ramp=ramp,K=K,O=O-D,B=B,noise=noise)
'''
                


