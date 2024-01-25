import numpy as np
import pandas as pd
import songburst.izneurons as izn

# for documentation read hvc.md

def paramshvc():#define parameters with homogeneity for a generic excitatory HVC neuron
    k1 = 0.04*1e6
    k2 = 5*1e3
    k3 = 140
    a = 0.02*1e3
    b = 0.2*1e3
    c = (-47)*1e-3
    d = 1.7
    E = -70*1e-3
    vth = 30*1e-3
    C = 167.5e-12 #F
    return k1,k2,k3,a,b,c,d,E,vth,C

def paramshvcra(h=0):#define parameters with homogeneity for a generic excitatory HVC neuron
    h = h*1e-3 #to ensure required dynamics
    k1 = np.random.normal(loc = 0.06849370077544009*1e6, scale = h*0.06849370077544009*1e6)
    k2 = np.random.normal(loc = 8.561712596930011*1e3, scale = h*8.561712596930011*1e3)
    k3 = np.random.normal(loc = 239.7279527140403, scale = h*239.7279527140403)
    a = np.random.normal(loc = 0.03141656545046908*1e3, scale = h*0.03141656545046908*1e3)
    b = np.random.normal(loc = 0.04186796270086881*1e3, scale = h*0.04186796270086881*1e3)
    c = np.random.normal(loc = (-51.59979175324334)*1e-3, scale = abs(h*(-51.59979175324334)*1e-3))
    d = np.random.normal(loc = 2.9362081518193364, scale = h*2.9362081518193364)
    E = np.random.normal(loc = (-81.38132030112037)*1e-3, scale = abs(h*(-81.38132030112037)*1e-3))
    vth = np.random.normal(loc = 20*1e-3, scale = h*20*1e-3)
    C = np.random.normal(loc = 111.15888637642897*1e-12, scale = h*111.15888637642897*1e-12) #F (Normalisation Constant)
    return k1,k2,k3,a,b,c,d,E,vth,C

def paramshvcx(h=1):#define parameters with homogeneity for a generic excitatory HVC neuron
    h = h*1e-3 #to ensure required dynamics
    k1 = np.random.normal(loc = 0.06849370077544009*1e6, scale = h*0.06849370077544009*1e6)
    k2 = np.random.normal(loc = 8.561712596930011*1e3, scale = h*8.561712596930011*1e3)
    k3 = np.random.normal(loc = 239.7279527140403, scale = h*239.7279527140403)
    a = np.random.normal(loc = 0.03141656545046908*1e3, scale = h*0.03141656545046908*1e3)
    b = np.random.normal(loc = 0.04186796270086881*1e3, scale = h*0.04186796270086881*1e3)
    c = np.random.normal(loc = (-51.59979175324334)*1e-3, scale = abs(h*(-51.59979175324334)*1e-3))
    d = np.random.normal(loc = 2.9362081518193364, scale = h*2.9362081518193364)
    E = np.random.normal(loc = (-81.38132030112037)*1e-3, scale = abs(h*(-81.38132030112037)*1e-3))
    vth = np.random.normal(loc = 20*1e-3, scale = h*20*1e-3)
    C = np.random.normal(loc = 111.15888637642897*1e-12, scale = h*111.15888637642897*1e-12) #F (Normalisation Constant)
    return k1,k2,k3,a,b,c,d,E,vth,C

def paramshvcint(h=0):#define parameters with homogeneity for a generic excitatory HVC neuron
    h = h*1e-3
    k1 = np.random.normal(loc = 0.30887914093147095*1e6, scale = h*0.30887914093147095*1e6)
    k2 = np.random.normal(loc = 38.60989261643387*1e3, scale = h*38.60989261643387*1e3)
    k3 = np.random.normal(loc = 1081.0769932601484, scale = h*1081.0769932601484)
    a = np.random.normal(loc = 0.1640855439624411*1e3, scale = h*0.1640855439624411*1e3)
    b = np.random.normal(loc = 1.805229494637493*1e3, scale = h*1.805229494637493*1e3)
    c = np.random.normal(loc = -61.45313597405551*1e-3, scale = abs(h*(-61.45313597405551)*1e-3))
    d = np.random.normal(loc = 62.5925531735772, scale = h*62.5925531735772)
    E = np.random.normal(loc = -66.61418754789635*1e-3, scale = abs(h*(-66.61418754789635)*1e-3))
    vth = np.random.normal(loc = 10*1e-3, scale = h*10*1e-3)
    C = np.random.normal(loc = 36.138071507338694*1e-12, scale = h*36.138071507338694*1e-12) #F
    return k1,k2,k3,a,b,c,d,E,vth,C

def paramshvcraRoss(h=0):#define parameters with homogeneity for a generic excitatory HVC neuron
    h = h*1e-3 #to ensure required dynamics
    k1 = np.random.normal(loc = 0.034*1e6, scale = h*0.034*1e6)
    k2 = np.random.normal(loc = 4.7*1e3, scale = h*4.7*1e3)
    k3 = np.random.normal(loc = 144.54, scale = h*144.54)
    a = np.random.normal(loc = 0.021*1e3, scale = h*0.021*1e3)
    b = np.random.normal(loc = 0.198*1e3, scale = h*0.198*1e3)
    c = np.random.normal(loc = (-61)*1e-3, scale = abs(h*(-61)*1e-3))
    d = np.random.normal(loc = 4.17, scale = h*4.17)
    E = np.random.normal(loc = (-77.46)*1e-3, scale = abs(h*(-77.46)*1e-3))
    vth = np.random.normal(loc = 20*1e-3, scale = h*20*1e-3)
    C = np.random.normal(loc = 50*1e-12, scale = h*50*1e-12) #F (Normalisation Constant)
    return k1,k2,k3,a,b,c,d,E,vth,C

def paramshvcxRoss(h=0):#define parameters with homogeneity for a generic excitatory HVC neuron
    h = h*1e-3
    k1 = np.random.normal(loc = 0.042*1e6, scale = h*0.042*1e6)
    k2 = np.random.normal(loc = 4.96*1e3, scale = h*4.96*1e3)
    k3 = np.random.normal(loc = 133.20, scale = h*133.20)
    a = np.random.normal(loc = 0.019*1e3, scale = h*0.019*1e3)
    b = np.random.normal(loc = 0.198*1e3, scale = h*0.198*1e3)
    c = np.random.normal(loc = (-53.29)*1e-3, scale = abs(h*(-53.29)*1e-3))
    d = np.random.normal(loc = 2.02, scale = h*2.02)
    E = np.random.normal(loc = (-64.13)*1e-3, scale = abs(h*(-64.13)*1e-3))
    vth = np.random.normal(loc = 30*1e-3, scale = h*30*1e-3)
    C = np.random.normal(loc = 46*1e-12, scale = h*46*1e-12) #F
    return k1,k2,k3,a,b,c,d,E,vth,C

def paramshvcintRoss(h=0):#define parameters with homogeneity for a generic excitatory HVC neuron
    h = h*1e-3
    k1 = np.random.normal(loc = 0.047*1e6, scale = h*0.047*1e6)
    k2 = np.random.normal(loc = 5.24*1e3, scale = h*5.24*1e3)
    k3 = np.random.normal(loc = 131.54, scale = h*131.54)
    a = np.random.normal(loc = 0.02*1e3, scale = h*0.02*1e3)
    b = np.random.normal(loc = 0.249*1e3, scale = h*0.249*1e3)
    c = np.random.normal(loc = -58.95*1e-3, scale = abs(h*(-58.95)*1e-3))
    d = np.random.normal(loc = 2.05, scale = h*2.05)
    E = np.random.normal(loc = -58.5*1e-3, scale = abs(h*(-58.5)*1e-3))
    vth = np.random.normal(loc = 10*1e-3, scale = h*10*1e-3)
    C = np.random.normal(loc = 9.75*1e-12, scale = h*9.75*1e-12) #F
    return k1,k2,k3,a,b,c,d,E,vth,C

def neuron_gen(conn, typ = None, hra = 0, hx = 0, hint = 0):
    #let us create object names to use
    X = pd.DataFrame({'N':np.arange(0,np.shape(conn)[0])})
    X = X.astype('str')
    X = 'n'+X[['N']]

    #generating object parameters names
    p = ['k1','k2','k3','a','b','c','d','E','vth','C']
    for i in p:
        X[i] = np.zeros(np.shape(X['N']))

    #creating N neurons within the dataframe
    for i,j in X.iterrows():
        if typ is None: k1,k2,k3,a,b,c,d,E,vth,C = paramshvc()
        elif typ[i] == 'ra': k1,k2,k3,a,b,c,d,E,vth,C = paramshvcra(hra)
        elif typ[i] == 'x' : k1,k2,k3,a,b,c,d,E,vth,C = paramshvcx(hx)
        elif typ[i] == 'int': k1,k2,k3,a,b,c,d,E,vth,C = paramshvcint(hint)
        j[1:] = k1,k2,k3,a,b,c,d,E,vth,C
        X.iloc[i,:] = j
        n = j['N']
        n = izn.IZN(j['k1'],j['k2'],j['k3'],j['a'],j['b'],j['c'],j['d'],j['E'],j['vth'],j['C'])
        X.at[i, 'N'] = n
    return X

def parametrize(n1,t_r,df):
    dt = df['t'].iloc[1]-df['t'].iloc[0]
    rmp = np.mean(n1.V[t_r]) #here t_r is that duration in which neuron is at resting membrane potential
    rmp = rmp*1e3
    mean_fr = np.sum(n1.sptr[df['ectopic']>0])/(np.shape(df['t'][df['ectopic']>0])[0]*dt)
    return rmp, mean_fr

def params_int_HH(T=298):
    # Parameters
    Cm = 60*pF

    El = -57*mV
    gl = 4.2*nS

    Ek = -90*mV
    gk = 220*nS
    theta_n = -30*mV
    sigma_n = -4*mV
    tau_n_bar = 7*ms

    Ena = 50*mV
    gna = 400*nS
    theta_m = -40*mV
    sigma_m = -3*mV

    gnap = 1*nS
    theta_mp = -40*mV
    sigma_mp = -6*mV
    theta_hp = -48*mV
    sigma_hp = 6*mV
    tau_hp_bar = 1000*ms

    ga = 1*nS
    theta_a = -20*mV
    sigma_a = -10*mV
    theta_e = -60*mV
    sigma_e = 5*mV
    tau_e = 20*ms

    ca_ex = 2.5*mM
    F = 96485*coulomb/mole
    T = T*kelvin
    R = 8.314*joule/mole/kelvin

    gcal = 1*nS
    theta_s = -13*mV
    sigma_s = -8.6*mV

    gcat = 2*nS
    theta_at = -59*mV
    sigma_at = -6*mV
    theta_bt = 0.4*mV
    sigma_bt = -0.1*mV
    theta_rt = -67*mV
    sigma_rt = 2*mV
    theta_rrt = 68*mV
    sigma_rrt = 2.2*mV
    tau_r0 = 200*ms
    tau_r1 = 87.5*ms

    gsk = 2.2*nS
    ks = 0.4*umolar
    f = 0.01
    eps = 0.0015*umolar/pA/ms
    kca = 0.3/ms
    bca = 0.05*umolar

    gm = 7*nS
    theta_z = -39*mV
    sigma_z = 5*mV
    tau_z = 75*ms

    gh = 8*nS
    Eh = -43*mV
    kr = 0.01
    prf = 100
    theta_rf = -87.7*mV
    sigma_rf = 6.4*mV
    theta_rs = -84*mV
    sigma_rs = 6.4*mV

    ##additional parameter
    w_tau_h = 1.2

    return Cm, El, gl, Ek, gk, theta_n, sigma_n, tau_n_bar, Ena, gna, theta_m, sigma_m, gnap, theta_mp, sigma_mp, theta_hp, sigma_hp, tau_hp_bar, ga, theta_a, sigma_a, theta_e, sigma_e, tau_e, ca_ex, R,T,F, gcal, theta_s, sigma_s, gcat, theta_at, sigma_at, theta_bt, sigma_bt, theta_rt, sigma_rt, theta_rrt, sigma_rrt, tau_r0, tau_r1, gsk, ks, f, eps, kca, bca, gm, theta_z, sigma_z, tau_z, gh, Eh, kr, prf, theta_rf, sigma_rf, theta_rs, sigma_rs, w_tau_h

def params_x_HH(T=298):
    # Parameters
    Cm = 260*pF

    El = -63*mV
    gl = 5*nS

    Ek = -90*mV
    gk = 120*nS
    theta_n = -30*mV
    sigma_n = -6*mV
    tau_n_bar = 10*ms

    Ena = 50*mV
    gna = 2300*nS
    theta_m = -38*mV
    sigma_m = -4*mV

    gnap = 1*nS
    theta_mp = -40*mV
    sigma_mp = -6*mV
    theta_hp = -48*mV
    sigma_hp = 6*mV
    tau_hp_bar = 1000*ms

    ga = 5*nS
    theta_a = -20*mV
    sigma_a = -10*mV
    theta_e = -60*mV
    sigma_e = 5*mV
    tau_e = 20*ms

    ca_ex = 2.5*mM
    F = 96485*coulomb/mole
    T = T*kelvin
    R = 8.314*joule/mole/kelvin

    gcal = 1*nS
    theta_s = -13*mV
    sigma_s = -8.6*mV

    gcat = 3.8*nS
    theta_at = -59*mV
    sigma_at = -6*mV
    theta_bt = 0.4*mV
    sigma_bt = -0.1*mV
    theta_rt = -67*mV
    sigma_rt = 2*mV
    theta_rrt = 68*mV
    sigma_rrt = 2.2*mV
    tau_r0 = 200*ms
    tau_r1 = 87.5*ms

    gsk = 2.1*nS
    ks = 0.4*umolar
    f = 0.01
    eps = 0.0015*umolar/pA/ms
    kca = 0.3/ms
    bca = 0.05*umolar

    gm = 15.4*nS
    theta_z = -39*mV
    sigma_z = 5*mV
    tau_z = 75*ms

    gh = 2.25*nS
    Eh = -43*mV
    kr = 0.17
    prf = 100
    theta_rf = -87.7*mV
    sigma_rf = 6.4*mV
    theta_rs = -87.7*mV
    sigma_rs = 6.4*mV

    ##additional parameter
    w_tau_h = 1

    return Cm, El, gl, Ek, gk, theta_n, sigma_n, tau_n_bar, Ena, gna, theta_m, sigma_m, gnap, theta_mp, sigma_mp, theta_hp, sigma_hp, tau_hp_bar, ga, theta_a, sigma_a, theta_e, sigma_e, tau_e, ca_ex, R,T,F, gcal, theta_s, sigma_s, gcat, theta_at, sigma_at, theta_bt, sigma_bt, theta_rt, sigma_rt, theta_rrt, sigma_rrt, tau_r0, tau_r1, gsk, ks, f, eps, kca, bca, gm, theta_z, sigma_z, tau_z, gh, Eh, kr, prf, theta_rf, sigma_rf, theta_rs, sigma_rs, w_tau_h

def params_ra_HH(T=298):
    # Parameters
    Cm = 58*pfarad

    El = -77*mV
    gl = 7*nsiemens

    Ek = -90*mV
    gk = 500*nsiemens
    theta_n = -30*mV
    sigma_n = -7*mV
    tau_n_bar = 15*ms

    Ena = 50*mV
    gna = 300*nsiemens
    theta_m = -35*mV
    sigma_m = -10*mV

    gnap = 1*nsiemens
    theta_mp = -40*mV
    sigma_mp = -6*mV
    theta_hp = -48*mV
    sigma_hp = 6*mV
    tau_hp_bar = 1000*ms

    ga = 5*nsiemens
    theta_a = -20*mV
    sigma_a = -10*mV
    theta_e = -60*mV
    sigma_e = 5*mV
    tau_e = 20*ms

    ca_ex = 2.5*mM
    F = 96485*coulomb/mole
    T = T*kelvin
    R = 8.314*joule/mole/kelvin

    gcal = 1*nsiemens
    theta_s = -13*mV
    sigma_s = -8.6*mV

    gcat = 1*nsiemens
    theta_at = -59*mV
    sigma_at = -6*mV
    theta_bt = 0.4*mV
    sigma_bt = -0.1*mV
    theta_rt = -67*mV
    sigma_rt = 2*mV
    theta_rrt = 68*mV
    sigma_rrt = 2.2*mV
    tau_r0 = 200*ms
    tau_r1 = 87.5*ms

    gsk = 32*nsiemens
    ks = 0.4*umolar
    f = 0.01
    eps = 0.0015*umolar/pA/ms
    kca = 0.3/ms
    bca = 0.05*umolar

    gm = 32*nsiemens
    theta_z = -45*mV
    sigma_z = 5*mV
    tau_z = 75*ms

    gh = 1.6*nsiemens
    Eh = -43*mV
    kr = 0.95
    prf = 100
    theta_rf = -87.7*mV
    sigma_rf = 6.4*mV
    theta_rs = -87.7*mV
    sigma_rs = 6.4*mV

    ##additional parameter
    w_tau_h = 1

    return Cm, El, gl, Ek, gk, theta_n, sigma_n, tau_n_bar, Ena, gna, theta_m, sigma_m, gnap, theta_mp, sigma_mp, theta_hp, sigma_hp, tau_hp_bar, ga, theta_a, sigma_a, theta_e, sigma_e, tau_e, ca_ex, R,T,F, gcal, theta_s, sigma_s, gcat, theta_at, sigma_at, theta_bt, sigma_bt, theta_rt, sigma_rt, theta_rrt, sigma_rrt, tau_r0, tau_r1, gsk, ks, f, eps, kca, bca, gm, theta_z, sigma_z, tau_z, gh, Eh, kr, prf, theta_rf, sigma_rf, theta_rs, sigma_rs, w_tau_h
