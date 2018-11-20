# simple script to use odeint
# here we integrate the Schrodinger equation for a free particle, U(x)=0
import numpy as np 
import matplotlib 
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.integrate import odeint
print("lol")

def schrodinger(state,x,k):
    hbar=1.0546e-34    # Plancks constant
    m=9.1094e-31       # electron mass
    e=1.6022e-19       # electron charge=-e
    c=2.0*m/hbar**2    # constant in Schrödinger equation

    EeV = 0.3          # input energy in eV: test 0.3 , 0.4 , 0.3760 , 1.5
    E = EeV*e          # input energy in J
    U0 = 0.6
    if 0.0 < x < 10.0:
        Ux = -(U0/100)*(4*x*(x-10))
    else:
        Ux = 0
    if EeV > Ux:
         k2 = EeV-Ux
    else:
        k2 = -(Ux-EeV)
    ''' system of 1st order differential equations to integrate.
        The Schrodinger equation for a free particle with wave vector,
        k**2 = 2m(E-U)/hbar**2
        
       args:
       state - array: psi[0] - Psi(x),  psi[1]= d/dx Psi(x).  Wave function and it's derivative.   
    '''
    return np.array([state[1], -k2*state[0]])

def calculateAmplitude(function, maxValue,x, Is_Stop_value):
    stopIndex = x.index(maxValue)
    print(stopIndex)
    if Is_Stop_value == True:
        limitedFunction =  function[:stopIndex]
    else:
        limitedFunction =  function[stopIndex:]
    amplitude = (np.absolute(np.amax(limitedFunction))-np.absolute(np.amin(limitedFunction)))/2
    print('Amplitude: ', amplitude)
    return amplitude
    

    
init=[1,0] # the initial condition, [ Psi, d/dx Psi] at x=0.

x= np.arange(-10., 20, 1) # vector of x values for output

#k =np.sqrt(c*(E-U)) # wave vector, suitably normalized
k = 1.
out=odeint(schrodinger,init,x,(k,)) # last argument, tupel of additional arguments to function schrodinger)
lista= out[:,0].tolist()
#amplitude_before_wall = calculateAmplitude(lista, 0,x.tolist(), True)
#amplitude_after_wall = calculateAmplitude(lista, 10,x.tolist(), False)
# make figure and plot results

fig = plt.figure(num=1, figsize=(10,6))
fig.clf()
ax = fig.add_subplot(111)

plt.plot(x,out[:,0])
plt.xticks(np.arange(0, 11, step=10), (0, 'L'))

plt.xlabel('$x$')
plt.ylabel('$\Psi (x)$') #note: matplotlib will interpret LateX 

#fig.canvas.draw()
#print(out)
plt.show()
