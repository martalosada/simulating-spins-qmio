from qmio import QmioRuntimeService
      
import numpy as np
from matplotlib import pyplot as plt
from qiskit import *
import logging

from qiskit.circuit import Parameter
# tau =  Parameter('tau')
pi = 3.141592653589793
Nt = 75
qr = QuantumRegister(3,'q')
cr = ClassicalRegister(3,'c')

from qmiotools.integrations.qiskitqmio import QmioBackend

backend=QmioBackend(logging_level=logging.DEBUG)
nshots = 8192
tau_range = np.linspace(0, 2*pi, Nt)
counts = []
for tau in tau_range:
    print(f"creating circuit for t={tau}...")
    timecirc = QuantumCircuit(qr,cr) 
    timecirc.u(2*tau,3.141592653589793,3.141592653589793,qr) #apply exp(-iHt/ħ)
    timecirc.barrier(qr)
    timecirc.ry(-3.141592653589793/2,2) #rotation to measure <Sx>

    timecirc.rz(-3.141592653589793/2,1)
    timecirc.ry(-3.141592653589793/2,1) #rotation to measure <Sy>
    timecirc.barrier(qr)
    #no rotation needed to measure <Sz>

    timecirc.measure(qr,cr)

    c=transpile(timecirc,backend,optimization_level=2)
    print("circuit transpiled.")
    print(c)
    #Execute the circuit with 1000 shots. Must be executed from a node with a QPU.
    job=backend.run(c,shots=1000)
    print(f"job for t={tau} sent...")
    print()
    #Return the results
    counts.append(job.result().get_counts())



Sx = np.zeros(Nt)
Sy = np.zeros(Nt)
Sz = np.zeros(Nt)

for n,count in zip(range(0,Nt),counts):
    simcounts = count
    states = list(simcounts.keys())
    for j in range(0,len(states)):
        state = states[j]
        if (state[0]=='0'):
            Sx[n] = Sx[n] + 0.5*simcounts[state]/nshots
        else:
            Sx[n] = Sx[n] - 0.5*simcounts[state]/nshots

        if (state[1]=='0'):
            Sy[n] = Sy[n] + 0.5*simcounts[state]/nshots
        else:
            Sy[n] = Sy[n] - 0.5*simcounts[state]/nshots

        if (state[2]=='0'):
            Sz[n] = Sz[n] + 0.5*simcounts[state]/nshots
        else:
            Sz[n] = Sz[n] - 0.5*simcounts[state]/nshots


plt.clf()
plt.plot(tau_range,Sx,'.-',label='<Sx>')
plt.plot(tau_range,Sy,'.-',label='<Sy>')
plt.plot(tau_range,Sz,'.-',label='<Sz>')
plt.plot(tau_range,-0.5*np.sin(2*tau_range),'b-')
plt.plot(tau_range,0*tau_range,'r-')
plt.plot(tau_range,0.5*np.cos(2*tau_range),'k-')
plt.xlabel('$\omega t$'); plt.ylabel('<S>'); plt.grid(True)
plt.legend()
plt.show()
plt.savefig("../results/single-spin-observables.png", dpi=400)