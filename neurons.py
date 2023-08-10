import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# for documentation read neurons.md

# functions to implement neurons

def lif_eu(I,time,R,E,Vth,V_r,tm,dt):
    V = np.ones(np.size(time))*E
    spiketrain = np.zeros(np.size(time))
    for i,t in enumerate(time):
        i=i+(1*(i<1))
        V[i] = V[i-1]+dt/tm*(E-V[i-1]+(R*I[i-1])) #euler's method
        if V[i]>Vth:
            V[i] = V_r
            spiketrain[i] = 1
    return (V*1e3),spiketrain

def AP(dur,spikeM,spikem,Vth):
    t_rise = np.rint(dur*10/2)
    rise = sp.stats.norm.pdf(np.arange(-1,0,1/t_rise))
    rise = (rise-np.min(rise))/(np.max(rise)-np.min(rise))
    rise = rise*(spikeM-Vth)+Vth
    t_fall = np.rint(dur*9/2)
    fall = np.sin(np.arange(np.pi/2,3*np.pi/2,np.pi/t_fall))
    fall = (fall-np.min(fall))/(np.max(fall)-np.min(fall))
    fall = fall*(spikeM-spikem)+spikem-1e-4
    ap = np.hstack((rise,fall))
    ap = np.delete(ap,[0])
    ap = np.append(ap,[ap[-1]+1e-4]); ap = np.append(ap,[ap[-1]+1e-4])
    return ap

def spi_lif_eu(I,time,R,E,Vth,V_r,tm,ref_per,apDur,spM,spm):
    V = np.ones(np.size(time))*E
    spiketrain = np.zeros(np.size(time))
    ap = AP(apDur,spM,spm,Vth)
    i=1
    while i<np.size(V)-1:
        V[i] = V[i-1]+dt/tm*(E-V[i-1]+(R*I[i-1])) #euler's method
        if V[i]>Vth:
            V[i:i+np.size(ap)] = ap
            V[i+np.size(ap):i+np.size(ap)+int(np.rint(ref_per/dt))] = np.min(ap)
            spiketrain[i] = 1
            i = i+np.size(ap)+int(np.rint(ref_per/dt))
        i+=1
    return (V*1e3),spiketrain

def PSC(t_rise, t_fall, mag, dt):
    rise = np.sin(np.linspace(-np.pi/2,np.pi/2,int(np.rint(t_rise/dt))))
    rise = (rise-np.min(rise))/(np.max(rise)-np.min(rise))
    
    fall = 2**(-(np.arange(0,7*t_fall/dt))*dt/t_fall)
    fall = (fall-np.min(fall))/(np.max(fall)-np.min(fall))
    
    psc = np.hstack((rise,fall))
    psc = psc*23.05/np.sum(psc)
    psc = psc*mag/np.max(psc)
    return psc

def VI_lif_eu(I,time,R,E,Vth,V_r,tm,ref_per,apDur,spM,spm,t_rise_PSC,t_fall_PSC,mag,w,dt):
    V = np.ones(np.size(time))*E
    spiketrain = np.zeros(np.size(time))
    Io = np.zeros(np.size(time))
    ap = AP(apDur,spM,spm,Vth)
    psc = PSC(t_rise_PSC,t_fall_PSC,mag,dt)
    i=1
    while i<np.size(V)-1:
        V[i] = V[i-1]+dt/tm*(E-V[i-1]+(R*I[i-1])) #euler's method
        if V[i]>Vth:
            V[i:i+np.size(ap)] = ap
            V[i+np.size(ap):i+np.size(ap)+int(np.rint(ref_per/dt))] = np.min(ap)
            spiketrain[i] = 1
            Io[i:i+np.size(psc)] = Io[i:i+np.size(psc)]+psc
            i = i+np.size(ap)+int(np.rint(ref_per/dt))
        i+=1
    return (V*1e3),(w*Io),spiketrain

def my_lif_eu(Ie_df,t,R,E,Vth,V_r,tm,ref_per,apDur,spM,spm,t_rise_PSC,t_fall_PSC,mag,C,dt):
    ap = AP(apDur,spM,spm,Vth)
    psc = PSC(t_rise_PSC,t_fall_PSC,mag,dt)
    Io_df = pd.DataFrame(np.zeros((np.size(t), np.shape(C)[0])))
    V_df = pd.DataFrame(np.zeros((np.size(t), np.shape(C)[0]))); V_df.iloc[0,:] = E
    spiketrain_df = Io_df.copy()
    mask = pd.DataFrame(np.ones((np.size(t), np.shape(C)[0])))
    i=1
    while i<np.size(t):
        I_t = Io_df.iloc[i-1,:]
        I_net = pd.Series(np.array(I_t.dot(C))+np.array(Ie_df.iloc[i-1,:]))
        V_df.iloc[i,:] = (V_df.iloc[i,:]*(
            -~-mask.iloc[i,:].astype(int)))+(V_df.iloc[i-1,:]+dt/tm*(E-V_df.iloc[i-1,:]+(R*I_net)))*mask.iloc[i,:]
        #euler's method
        for n in np.arange(0,np.shape(C)[0]):
            if ((V_df.iloc[i,n]>Vth)&(mask.iloc[i,n]==1)):
                V_df.iloc[i:i+np.size(ap),n] = ap
                V_df.iloc[i+np.size(ap):i+np.size(ap)+int(np.rint(ref_per/dt)),n] = np.min(ap)
                spiketrain_df.iloc[i,n] = 1
                Io_df.iloc[i:i+np.size(psc),n] = Io_df.iloc[i:i+np.size(psc),n]+psc[:np.size(Io_df.iloc[i:i+np.size(psc),n])]
                mask.iloc[i:i+np.size(ap)+int(np.rint(ref_per/dt)),n] = 0
        i+=1
    return (V_df*1e3), Io_df, spiketrain_df

# functions to generate stimuli

def stimuli(df, mag, stim='pulse', dur=3, st=1, pwidth=0.05, gap=2, base=0, noise=0.2, ramp=1, rampup_t=None, rampdown_t=None, psp_dur=0.04, freq=1/(0.02), synaptize=False, noisy=False):   
    T = df['t'].max() #maximum time
    #step current
    if stim == 'step':
        df['step'] = np.ones(np.size(df['t']))*base
        step = (df['t']>st) & (df['t']<st+dur)
        df['step'][step] = mag
    #sine current
    if stim == 'sin':
        df['sin'] = np.ones(np.size(df['t']))*base
        step = (df['t']>st) & (df['t']<st+dur)
        df['sin'][step] = mag*0.1*np.sin(1e2*df['t'][step]+5)+mag
    #linear increase
    if stim == 'lin':
        df['lin'] = np.ones(np.size(df['t']))*base
        step = (df['t']>st) & (df['t']<st+dur)
        t_step = df['t'][step]; l_t_step = np.max(t_step)-np.min(t_step);
        df['lin'][step] = (mag/l_t_step)*(t_step)-(mag/l_t_step)*(t_step.iloc[0])
    #pulsatile
    if stim == 'pulse':
        df['pulse'] = np.ones(np.size(df['t']))*base
        step = (df['t']<0)
        for i in np.arange(st, st+dur, pwidth+gap):
            step = step|((df['t']>=i)&(df['t']<=i+pwidth))
        df['pulse'][step] = mag    
    #bump
    if stim == 'bump':
        df['bump'] = np.ones(np.size(df['t']))*base
        bump = (df['t']>st) & (df['t']<st+dur)
        df['bump'][bump] = mag
        rampup_t = ramp/2 if (rampup_t is None) else rampup_t; rampdown_t = ramp/2 if (rampdown_t is None) else rampdown_t
        rampup = (df['t']>st) & (df['t']<st+rampup_t); rampdown = (df['t']>st+dur-rampdown_t) & (df['t']<st+dur)
        t_step = df['t'][rampup]; l_t_step = np.max(t_step)-np.min(t_step);
        df['bump'][rampup] = (mag/l_t_step)*(t_step)-(mag/l_t_step)*(t_step.iloc[0])
        t_step = df['t'][rampdown]; l_t_step = np.max(t_step)-np.min(t_step);
        df['bump'][rampdown] = (-mag/l_t_step)*(t_step)+(mag/l_t_step)*(t_step.iloc[-1])
    
    #synaptize the stimulus - makes the stimulus high frequency pulse like (realistic)
    if synaptize==True:
        step = (df['t']<0)
        for i in np.arange(0, T, psp_dur+(1/freq)):
            step = step|((df['t']>=i)&(df['t']<=i+(1/freq)))
        df[stim][step] = 0    
    #make the stimulus noisy                     
    if noisy==True:
        df[stim] = df[stim]+(np.ones(np.size(df['t']))*noise*np.random.uniform(-1,1,np.size(df['t'])))        
    return df[stim] #return the stimulus asked for

def inject(neurons, X, mag, df, stim='step', dur=3, pwidth=0.05, gap=0.25, base=0, noise=1, st=1, ramp=1, rampup_t=None, rampdown_t=None, psp_dur=0.04, freq=1/(0.02), synaptize=False, noisy=False, overlay=False, Iec_df=None):                         
    if overlay == False:
        Iec_df = pd.DataFrame(np.zeros((np.size(df['t']), np.shape(X)[0]))) #clear the dataframe
    else: Iec_df = Iec_df                         
    for i,j in X.iterrows():
        if i in neurons:
                Iec_df.iloc[:,i] = stimuli(df, mag/X['N'][i].C, stim=stim, dur=dur, pwidth=pwidth, gap=gap, base=base/X['N'][i].C, noise=noise, st=st, ramp=ramp, rampup_t=rampup_t, rampdown_t=rampdown_t, psp_dur=psp_dur, freq=freq, synaptize=synaptize, noisy=noisy)      
    return Iec_df

def release(neurons, X, mag, df, stim='step', dur=3, pwidth=0.05, gap=0.25, base=0, noise=1, st=1, ramp=1, rampup_t=None, rampdown_t=None, psp_dur=0.04, freq=1/(0.02), synaptize=False, noisy=False, overlay=False, mod_df=None):    
    if overlay == False:
        mod_df = pd.DataFrame(np.zeros((np.size(df['t']), np.shape(X)[0]))) #clear the dataframe
    else: mod_df = mod_df                         
    for i,j in X.iterrows():
        if i in neurons:
                mod_df.iloc[:,i] = stimuli(df, mag, stim=stim, dur=dur, pwidth=pwidth, gap=gap, base=base, noise=noise, st=st, ramp=ramp, rampup_t=rampup_t, rampdown_t=rampdown_t, psp_dur=psp_dur, freq=freq, synaptize=synaptize, noisy=noisy)                         
    return mod_df

def inject_single(neuron, mag, df, stim='step', dur=3, pwidth=0.05, gap=0.25, base=0, noise=1, st=1, ramp=1, rampup_t=None, rampdown_t=None, psp_dur=0.04, freq=1/(0.02), synaptize=False, noisy=False, plotter=False):                         
    df['ectopic'] = stimuli(df, mag/neuron.C, stim=stim, dur=dur, pwidth=pwidth, gap=gap, base=base/neuron.C, noise=noise, st=st, ramp=ramp, rampup_t=rampup_t, rampdown_t=rampdown_t, psp_dur=psp_dur, freq=freq, synaptize=synaptize, noisy=noisy)
    neuron.Iec = np.array(df['ectopic'])
    if plotter == True:
        plt.plot(df['t'],neuron.Iec,'r')
        plt.ylabel('$I_{ectopic}\ (pA/pF)$'); plt.xlabel('$t\ (s)$');
        plt.show();
    return df, neuron

def release_single(neuron, mag, df, stim='step', dur=3, pwidth=0.05, gap=0.25, base=0, noise=1, st=1, ramp=1, rampup_t=None, rampdown_t=None, psp_dur=0.04, freq=1/(0.02), synaptize=False, noisy=False, plotter=False):
    df['modulator'] = stimuli(df, mag, stim=stim, dur=dur, pwidth=pwidth, gap=gap, base=base, noise=noise, st=st, ramp=ramp, rampup_t=rampup_t, rampdown_t=rampdown_t, psp_dur=psp_dur, freq=freq, synaptize=synaptize, noisy=noisy)
    neuron.mod = np.array(df['modulator'])
    if plotter == True:
        plt.plot(df['t'],neuron.mod,'r')
        plt.ylabel('$Modulation$'); plt.xlabel('$t\ (s)$');
        plt.show();
    return df, neuron
