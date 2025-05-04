from qmio import QmioRuntimeService
      
import numpy as np
from matplotlib import pyplot as plt
from qiskit import *
import logging

'expectation value of an operator in a state'
exp_val = lambda A, state: np.dot(state.conj().T,np.dot(A, state))

#two-gate operator for exp[i(α σx σx + β σy σy + γ σz σz)]
def N(α,β,γ,circ,q1,q2):
    circ.rz(-0.5*np.pi,q2)
    circ.cx(q2,q1)
    circ.rz(0.5*np.pi-γ,q1)
    circ.ry(α-0.5*np.pi,q2)
    circ.cx(q1,q2)
    circ.ry(0.5*np.pi-β,q2)
    circ.cx(q2,q1)
    circ.rz(0.5*np.pi,q1)

from qiskit.circuit import Parameter
tau = Parameter('τ')
Nt = 75
tau_range = np.linspace(0,15,Nt)

Jx = 0.5
Jy = -0.45
Jz = 0.25

ntrot = 10

qr = QuantumRegister(3,'q')
cr = ClassicalRegister(3,'c')

timecirc = QuantumCircuit(qr,cr) 

#initial states
timecirc.u(np.pi/6,np.pi/3,0,0)
timecirc.u(3*np.pi/5,4*np.pi/3,0,1)
timecirc.u(-np.pi/5,2*np.pi/3,0,2)

for i in range(0,ntrot):
    N(Jx*(tau/ntrot)/4.0,Jy*(tau/ntrot)/4.0,Jz*(tau/ntrot)/4.0,timecirc,0,1)
    N(Jx*(tau/ntrot)/4.0,Jy*(tau/ntrot)/4.0,Jz*(tau/ntrot)/4.0,timecirc,1,2)
    
timecirc.measure(qr,cr)
timecirc.draw()

from qmiotools.integrations.qiskitqmio import FakeQmio
from qiskit_aer import AerSimulator
backend=FakeQmio("/opt/cesga/qmio/hpc/calibrations/2025_04_10__12_00_02.json",gate_error=True, readout_error=True, logging_level=logging.DEBUG)
# backend=AerSimulator()
nshots = 81920

counts=[]
for t in tau_range:
    qc = timecirc.assign_parameters([t])
    qct = transpile(qc,backend, optimization_level=2)

    counts.append(backend.run(qct, shots=nshots).result().get_counts())



#array to store time-dependent expectation values for all three spins

Szt = np.zeros((3,Nt))

for n,c in zip(range(0,Nt),counts):
    simcounts = c
    states = list(simcounts.keys())
    for j in range(0,len(states)):
        state = states[j]
        if (state[0]=='0'):
            Szt[0,n] = Szt[0,n] + 0.5*simcounts[state]/nshots
        else:
            Szt[0,n] = Szt[0,n] - 0.5*simcounts[state]/nshots

        if (state[1]=='0'):
            Szt[1,n] = Szt[1,n] + 0.5*simcounts[state]/nshots
        else:
            Szt[1,n] = Szt[1,n] - 0.5*simcounts[state]/nshots

        if (state[2]=='0'):
            Szt[2,n] = Szt[2,n] + 0.5*simcounts[state]/nshots
        else:
            Szt[2,n] = Szt[2,n] - 0.5*simcounts[state]/nshots


# theoretical

tau = Parameter('τ')
Nt = 75
tau_range = np.linspace(0,15,Nt)

Jx = 0.5
Jy = -0.45
Jz = 0.25

ntrot = 10

qr = QuantumRegister(3,'q')
cr = ClassicalRegister(3,'c')

timecirc = QuantumCircuit(qr,cr) 

#initial states
timecirc.u(np.pi/6,np.pi/3,0,0)
timecirc.u(3*np.pi/5,4*np.pi/3,0,1)
timecirc.u(-np.pi/5,2*np.pi/3,0,2)

for i in range(0,ntrot):
    N(Jx*(tau/ntrot)/4.0,Jy*(tau/ntrot)/4.0,Jz*(tau/ntrot)/4.0,timecirc,0,1)
    N(Jx*(tau/ntrot)/4.0,Jy*(tau/ntrot)/4.0,Jz*(tau/ntrot)/4.0,timecirc,1,2)
    
timecirc.draw()

from qiskit.quantum_info import SparsePauliOp

Sx1 = SparsePauliOp.from_list([("IIX", 0.5)])
Sy1 = SparsePauliOp.from_list([("IIY", 0.5)])
Sz1 = SparsePauliOp.from_list([("IIZ", 0.5)])

Sx2 = SparsePauliOp.from_list([("IXI", 0.5)])
Sy2 = SparsePauliOp.from_list([("IYI", 0.5)])
Sz2 = SparsePauliOp.from_list([("IZI", 0.5)])

Sx3 = SparsePauliOp.from_list([("XII", 0.5)])
Sy3 = SparsePauliOp.from_list([("YII", 0.5)])
Sz3 = SparsePauliOp.from_list([("ZII", 0.5)])

pub = (timecirc, [[Sx1], [Sy1], [Sz1], [Sx2], [Sy2], [Sz2], [Sx3], [Sy3], [Sz3]], tau_range)

nshots = 8192

from qiskit.primitives import StatevectorEstimator

estimator = StatevectorEstimator()

job_result = estimator.run(pubs=[pub]).result()


Sx1 = job_result[0].data.evs[0]
Sy1 = job_result[0].data.evs[1]
Sz1 = job_result[0].data.evs[2]

Sx2 = job_result[0].data.evs[3]
Sy2 = job_result[0].data.evs[4]
Sz2 = job_result[0].data.evs[5]

Sx3 = job_result[0].data.evs[6]
Sy3 = job_result[0].data.evs[7]
Sz3 = job_result[0].data.evs[8]


plt.clf()
# th
plt.plot(tau_range,Sz1,"g--")
plt.plot(tau_range,Sz2,"--", color="orange")
plt.plot(tau_range,Sz3,"b--")

# sim
plt.plot(tau_range,Szt[0,:], "o", alpha=0.7,label=r"$\langle S^zII\rangle/\hbar$")
plt.plot(tau_range,Szt[1,:], "o", alpha=0.7,label=r"$\langle IS^zI\rangle/\hbar$")
plt.plot(tau_range,Szt[2,:], "o", alpha=0.7,label=r"$\langle IIS^z\rangle/\hbar$")
plt.xlabel('Jt')
plt.ylabel('$<S_{i}^{z}>/\hbar$'); plt.title(r"n=10 Trotter steps"); plt.grid(True); plt.legend(loc="lower left")
plt.show()
plt.savefig("./results/three-spin-fake_10.png", dpi=400)