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

def hvc_neuron(x,plotter = False,name='hvcneuron',csv_dir = 'data/hvc_', model='long_2010', sim_time=1):
    VI = pd.read_csv(csv_dir+x+'_'+model+'_VI.csv')
    i = range(0,np.shape(VI)[1],2)
    VI.iloc[:,i] = VI.iloc[:,i]
    v = range(1,np.shape(VI)[1],2)
    VI.iloc[:,v] = VI.iloc[:,v]
    VI['t'] = np.linspace(0,sim_time,np.shape(VI)[0])
    sptr = sptr_con(VI,v,i)
    V = VI.iloc[:,v]
    I = VI.iloc[:,i]
    
    if plotter == True:
        #plotting current and voltage trace
        plt.plot(VI['t'],I*1e12); plt.ylabel('$I_{ectopic}\ (pA)$')
        plt.xlabel('$t\ (s)$'); plt.title('$HVC_{'+x+'}$'); 
        plt.savefig(name+'_injected_currents',dpi=300); plt.show()
        plt.plot(VI['t'],V*1e3); plt.ylim(-110,35); plt.ylabel('$V\ (mV)$');plt.xlabel('$t\ (s)$');
        plt.title('$HVC_{'+x+'}$');
        plt.savefig(name+'_voltage_traces',dpi=300); plt.show()
    return V,I,sptr

##Long 2010 2-compartment and 1-compartment conductance models
def Vt(v, currents, Iext, Cm, dt):
    return v + (dt/Cm)*(np.sum(currents) + Iext)

def Il(v, gl, El):
    return -gl*(v-El)

def INa(v, M, H, gNa, ENa):
    return -gNa*(M**3)*H*(v-ENa)

def IKdr(v, N, gKdr, EKdr):
    return -gKdr*(N**4)*(v-EKdr)

def IKHT(v, W, gKHT, EKHT):
    return -gKHT*W*(v-EKHT)

def G1(g, ginf, taug, dt):
    return g + (dt/taug)*(ginf - g)

def G2(g, alpha, beta, dt):
    return g + dt*((alpha*(1-g)) - (beta*g))

def Alpham(v):
    return (v+22)/(1-np.exp(-(v+22)/10))

def Betam(v):
    return 40*np.exp(-(v+47)/18)

def Alphah(v):
    return 0.7*np.exp(-(v+34)/20)

def Betah(v):
    return 10/(1+np.exp(-(v+4)/10))

def Alphan(v):
    return 0.15*(v+15)/(1-np.exp(-(v+15)/10))

def Betan(v):
    return 0.2*np.exp(-(v+25)/80)

def winf(v):
    return 1/(1+np.exp(-v/5))

def evolve1C(t, I, dt,
                V0 = -66, #mV
                m0 = 0,
                h0 = 1,
                n0 = 0,
                w0 = 0,
                gl = 0.1, #mS/cm^2
                gNa = 100, #mS/cm^2
                gKdr = 20, #mS/cm^2
                gKHT = 500, #mS/cm^2
                El = -65, #mV
                ENa = 55, #mS/cm^2
                EK = -80, #mV
                Cm = 1, #uF/cm^2
                A = 6000/1e5 #cm^2/100
            ):
    steps = int(t/dt)
    V = np.zeros(steps); V[0] = V0;
    m = np.zeros(steps); m[0] = m0;
    h = np.zeros(steps); h[0] = h0;
    n = np.zeros(steps); n[0] = n0;
    w = np.zeros(steps); w[0] = w0;
    for i in range(1,steps):
        m[i] = G2(m[i-1], Alpham(V[i-1]), Betam(V[i-1]), dt)
        h[i] = G2(h[i-1], Alphah(V[i-1]), Betah(V[i-1]), dt)
        n[i] = G2(n[i-1], Alphan(V[i-1]), Betan(V[i-1]), dt)
        w[i] = G1(w[i-1], winf(V[i-1]), 1, dt)
        V[i] = Vt(V[i-1], 
                    [Il(V[i-1], gl, El), 
                    INa(V[i-1], m[i-1], h[i-1], gNa, ENa),
                    IKdr(V[i-1], n[i-1], gKdr, EK),
                    IKHT(V[i-1], w[i-1], gKHT, EK)],
                    I[i-1]/A,
                    Cm,
                    dt)
    
    return V

def ICa(v, R, gCa, ECa):
    return -gCa*(R**2)*(v-ECa)

def ICaK(v, c, Ca, gCaK, ECaK):
    return -gCaK*c/(1+6/Ca)*(v-ECaK)

def taun(v):
    return 0.1 + 0.5/(1 + np.exp((v+27)/15))

def tauh(v):
    return 0.1 + 0.75/(1 + np.exp((v+40.5)/6))

def ninf(v):
    return 1/(1+ np.exp(-(v+35)/10))

def hinf(v):
    return 1/(1+ np.exp((v+45)/7))

def minf(v):
    return 1/(1+ np.exp(-(v+30)/9.5))

def rinf(v):
    return 1/(1+ np.exp(-(v+5)/10))

def cinf(v):
    return 1/(1+ np.exp(-(v-10)/7))

def calcium(ica, ca, dt):
    return ca + dt*(0.1*ica - 0.02*ca)

def evolve2C(t, I, dt,
                Vd0 = -80, #mV
                Vs0 = -80, #mV
                h0 = 1,
                n0 = 0,
                r0 = 0,
                c0 = 0,
                Ca0 = 70,
                gl = 0.1, #mS/cm^2
                gNa = 60, #mS/cm^2
                gK = 8, #mS/cm^2
                gCa = 55, #mS/cm^2
                gCaK = 150, #mS/cm^2
                El = -80, #mV
                ENa = 50, #mV
                EK = -90, #mV
                ECa = 120, #mV
                Cm = 1, #uF/cm^2
                Ad = 10000/1e5, #cm^2/100
                As = 5000/1e5, #cm^2/100
                Rc = 55 #MOhm
            ):
    steps = int(t/dt)
    Vd = np.zeros(steps); Vd[0] = Vd0;
    Vs = np.zeros(steps); Vs[0] = Vs0;
    h = np.zeros(steps); h[0] = h0;
    n = np.zeros(steps); n[0] = n0;
    r = np.zeros(steps); r[0] = r0;
    c = np.zeros(steps); c[0] = c0;
    Ca = np.zeros(steps); Ca[0] = Ca0;
    
    for i in range(1,steps):
        h[i] = G1(h[i-1], hinf(Vs[i-1]), tauh(Vs[i-1]), dt)
        n[i] = G1(n[i-1], ninf(Vs[i-1]), taun(Vs[i-1]), dt)
        r[i] = G1(r[i-1], rinf(Vd[i-1]), 1, dt)
        c[i] = G1(c[i-1], cinf(Vd[i-1]), 10, dt)
        Ca[i] = calcium(ICa(Vd[i-1], r[i-1], gCa, ECa), Ca[i-1], dt)
        
        Vd[i] = Vt(Vd[i-1], 
                    [Il(Vd[i-1], gl, El), 
                    ICa(Vd[i-1], r[i-1], gCa, ECa),
                    ICaK(Vd[i-1], c[i-1], Ca[i-1], gCaK, EK)],
                    (I[i-1]+(Vs[i-1]-Vd[i-1])/Rc)/Ad,
                    Cm,
                    dt)
        
        Vs[i] = Vt(Vs[i-1], 
                    [Il(Vs[i-1], gl, El), 
                    INa(Vs[i-1], minf(Vs[i-1]), h[i-1], gNa, ENa),
                    IKdr(Vs[i-1], n[i-1], gK, EK)],
                    ((Vd[i-1]-Vs[i-1])/Rc)/As,
                    Cm,
                    dt)
        
    return Vs

