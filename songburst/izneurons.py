import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import songburst.neurons as nrn

# for documentation read izneurons.md

class IZN:
    def __init__(neuron,k1,k2,k3,a,b,c,d,E,vth,C):
        x = np.random.rand()
        neuron.k1 = k1
        neuron.k2 = k2
        neuron.k3 = k3
        neuron.a = a
        neuron.b = b
        neuron.c = c
        neuron.d = d
        neuron.E = E
        neuron.vth = vth
        neuron.C = C

def modulate(n1, m, reset = False, params=None):
    if reset == True: n1 = IZN(*params)
    m = m*(1e-3*4)
    n1.k1 = n1.k1*(1+m)
    n1.k2 = n1.k2*(1-m)
    n1.k3 = n1.k3*(1+m)
    n1.a = n1.a*(1+m)
    n1.b = n1.b*(1+m)
    n1.c = n1.c*(1+m) #the reset value of voltage is an important parameter for realistic HVCra activity
    n1.d = n1.d*(1+m)
    n1.vth = n1.vth*(1-m)
    n1.C = n1.C*(1-m)
    return n1

def modulate_net(k1,k2,k3,a,b,c,d,vth,C,m):
    m = m*(1e-3*4)
    k1 = k1*(1+m)
    k2 = k2*(1-m)
    k3 = k3*(1+m)
    a = a*(1+m)
    b = b*(1+m)
    c = c*(1+m) #the reset value of voltage is an important parameter for realistic HVCra activity
    d = d*(1+m)
    vth = vth*(1-m)
    C = C*(1-m)
    return k1,k2,k3,a,b,c,d,vth,C

def evolve(n1,df,psc = nrn.PSC(0.5e-3,1e-3,0.05e-9,1e-5)):
    time = np.array(df['t'])
    dt = time[1]-time[0]
    n1.V = np.ones(np.size(df['t']))*n1.E
    n1.U = n1.V*n1.b
    n1.Ips = np.zeros(np.size(df['t']))
    n1.sptr = np.zeros(np.size(df['t']))
    Iin = n1.Iin if hasattr(n1,'Iin') else n1.Ips.copy() #if no synaptic input, then set it as 0
    v = n1.V.copy()
    u = n1.U.copy()
    sptr = n1.sptr.copy()
    Sps = n1.Ips.copy() #Sps is non-normalised current
    mod = n1.mod if hasattr(n1,'mod') else n1.Ips.copy() #if no modulatory input, then set it as 0
    
    #convert to integers for speed
    k1 = n1.k1; k2 = n1.k2; k3 = n1.k3; a = n1.a; b = n1.b; c = n1.c; d = n1.d
    vth = n1.vth; C = n1.C; Iec = n1.Iec
    
    for i,t in enumerate(time):
        i=i+(1*(i<1))
        shift = mod[i]-mod[i-1]
        n1 = modulate(n1,shift)
        f = n1.k1*(v[i-1]**2) + n1.k2*(v[i-1]) + n1.k3 - u[i-1]
        g = n1.a*(n1.b*v[i-1]-u[i-1])
        v[i] = v[i-1]+dt*(f+Iec[i-1]+Iin[i-1]) #forward-euler method
        u[i] = u[i-1]+g*dt
        if v[i]>vth:
            v[i] = c
            u[i] = u[i]+d
            sptr[i] = 1
            Sps[i:i+np.size(psc)] = Sps[i:i+np.size(psc)]+psc[:np.size(Sps[i:i+np.size(psc)])]
    n1.V = v; n1.U = u; n1.sptr = sptr; n1.Ips = Sps/C #normalised by membrane capacitance
    return n1

def evolve_net(X,conn,df,Iec_df,mod_df = None,psc = nrn.PSC(0.5e-3,1e-3,0.05e-9,1e-5),delays=None,upto=None):
    
    #get time vector and step size
    time = np.array(df['t'])
    dt = time[1]-time[0]
    
    #specify evolution time
    if upto is not None:
        upto = np.size(time[time<=upto])
    else:
        upto = np.size(time)
    
    #creating numpy arrays to increase speed:
    Iec = np.array(Iec_df)
    conn = np.array(conn)
    E = np.array(X['E']); b = np.array(X['b']); k1 = np.array(X['k1']);
    k2 = np.array(X['k2']); k3 = np.array(X['k3']); a = np.array(X['a']); 
    c = np.array(X['c']); d = np.array(X['d']); C = np.array(X['C']); vth = np.array(X['vth'])
    
    if delays is not None: #then axonal delays have to be incorporated
        delays = np.array(delays) #create array for speed
        Ips_shifted = np.zeros((np.shape(X)[0], np.shape(X)[0]))
        if np.max(delays)>=np.max(time):
            #check for any delays interfering with simulation
            raise ValueError("Delays too large for simulation")
    
    Ips = np.zeros((np.size(time), np.shape(X)[0]))
    Iin = np.zeros((np.size(time), np.shape(X)[0]))
    mod = np.zeros((np.size(time), np.shape(X)[0])) if (mod_df is None) else np.array(mod_df)
    V = np.zeros((np.size(time), np.shape(X)[0])); V[0,:] = E
    U = V.copy(); U[0,:] = np.array(E*b)
    sptr = Ips.copy()
    Sps = Ips.copy() #Sps is non-normalised current
    
    for i in range(upto):
        i=i+(1*(i<1))
        
        #axonal delays
        if delays is not None:
            Del = [list(np.round(i-1-delays[:,x]/dt).astype('int')) for x in range(np.shape(X)[0])] #get delays
            Ips_shifted = Ips[Del,range(np.shape(X)[0])] #shift PSC for each neuron
            
            #get recurrent inputs
            Iin[i-1,:] =  (Ips_shifted * conn.T).sum(-1) #diagonal of matrix product
        else:
            Iin[i-1,:] = Ips[i-1,:].dot(conn) #matrix product
        
        #modulation
        shift = mod[i,:]-mod[i-1,:]
        k1,k2,k3,a,b,c,d,vth,C = modulate_net(k1,k2,k3,a,b,c,d,vth,C,shift)
        
        #izhikevich equations
        f = k1*(V[i-1,:]**2) + k2*(V[i-1,:]) + k3 - U[i-1,:]
        g = a*((b*V[i-1,:])-U[i-1,:])
        
        V[i,:] = V[i-1,:]+(dt*(f+Iec[i-1,:]+Iin[i-1,:])) #euler's method
        U[i,:] = U[i-1,:]+(g*dt)
        
        for n in range(np.shape(X)[0]): #n is the neuron number, X contains all the params for that neuron
            if (V[i,n]>vth[n]):
                V[i,n] = c[n]
                U[i,n] = U[i,n]+d[n]
                sptr[i,n] = 1
                Sps[i:i+np.size(psc),n] = Sps[i:i+np.size(psc),n]+psc[:np.size(Sps[i:i+np.size(psc),n])]
                Ips[i:i+np.size(psc),n] = Sps[i:i+np.size(psc),n]/C[n]
            if i==(upto-1):
                X['N'][n].V = V[:,n]; X['N'][n].U = U[:,n]; 
                X['N'][n].sptr = sptr[:,n]; X['N'][n].Iin = Iin[:,n]
    return X
