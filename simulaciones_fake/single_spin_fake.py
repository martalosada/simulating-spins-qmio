from qmio import QmioRuntimeService
      
import numpy as np
from matplotlib import pyplot as plt
from qiskit import *
import logging

from qiskit.circuit import Parameter
# tau =  Parameter('tau')
Nt = 75
qr = QuantumRegister(3,'q')
cr = ClassicalRegister(3,'c')

from qmiotools.integrations.qiskitqmio import FakeQmio

backend=FakeQmio("/opt/cesga/qmio/hpc/calibrations/2025_04_10__12_00_02.json",gate_error=True, readout_error=True, logging_level=logging.DEBUG)
nshots = 8192
tau_range = np.linspace(0, 2*np.pi, Nt)
counts = []
for tau in tau_range:
    print(f"creating circuit for t={tau}...")
    timecirc = QuantumCircuit(qr,cr) 
    timecirc.u(2*tau,np.pi,np.pi,qr) #apply exp(-iHt/ħ)
    timecirc.barrier(qr)
    timecirc.ry(-np.pi/2,2) #rotation to measure <Sx>

    timecirc.rz(-np.pi/2,1)
    timecirc.ry(-np.pi/2,1) #rotation to measure <Sy>
    timecirc.barrier(qr)
    #no rotation needed to measure <Sz>

    timecirc.measure(qr,cr)

    c=transpile(timecirc,backend,optimization_level=2)
    print("circuit transpiled.")
    print(c)
    #Execute the circuit with 1000 shots. Must be executed from a node with a QPU.
    job=backend.run(c,shots=nshots)
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
plt.plot(tau_range,Sx,'o',alpha = 0.7,label=r'$\langle \hat{S}_x\rangle/\hbar $')
plt.plot(tau_range,Sy,'o',alpha = 0.7,label=r'$\langle \hat{S}_y\rangle/\hbar $')
plt.plot(tau_range,Sz,'o',alpha = 0.7,label=r'$\langle \hat{S}_z\rangle/\hbar $')
plt.plot(tau_range,-0.5*np.sin(2*tau_range),'k-', linewidth = 0.8)
plt.plot(tau_range,0*tau_range,'k-', linewidth = 0.8)
plt.plot(tau_range,0.5*np.cos(2*tau_range),'k-', linewidth = 0.8)
plt.xlabel(r'$\omega_0t = \mu B_0t$'); plt.ylabel(r'$\langle \hat{S}^i\rangle /\hbar$'); plt.grid(True)
plt.legend(); plt.title(r"$\hat{U}(t) = \hat{U}_3(\omega_0t,\pi,\pi)$")
plt.show()
plt.savefig("./results/single-spin-observables-fake.png", dpi=400)