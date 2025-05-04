from qiskit.circuit import Parameter
import numpy as np
from matplotlib import pyplot as plt
from qiskit import *
import logging

from qiskit_aer import AerSimulator

def spin_chain(N, J,Jz, ideal=False):
    Nhalf = int(0.5*N)
    t = Parameter('t')
    Nt = 20 #number of time samples
    tmax = float(Nhalf)
    tau_range = np.linspace(0,tmax,Nt)

    ntrot = 5 #Trotter steps

    Ngate_qr = QuantumRegister(2)
    Ngate_qc = QuantumCircuit(Ngate_qr,name='N3')

    Ngate_qc.rz(-0.5*np.pi,Ngate_qr[1])
    Ngate_qc.cx(Ngate_qr[1],Ngate_qr[0])
    Ngate_qc.rz(0.5*np.pi-2*Jz*t/(4.0*ntrot),Ngate_qr[0])
    Ngate_qc.ry(2.0*J*t/(4.0*ntrot)-0.5*np.pi,Ngate_qr[1])
    Ngate_qc.cx(Ngate_qr[0],Ngate_qr[1])
    Ngate_qc.ry(0.5*np.pi-2.0*J*t/(4.0*ntrot),Ngate_qr[1])
    Ngate_qc.cx(Ngate_qr[1],Ngate_qr[0])
    Ngate_qc.rz(0.5*np.pi,Ngate_qr[0])

    N_gate = Ngate_qc.to_instruction()


    qr = QuantumRegister(N,'q')
    cr = ClassicalRegister(N,'c')

    timecirc = QuantumCircuit(qr,cr)

    #domain wall initial state |ψ> = |++...++--...-->
    timecirc.x(qr[0:Nhalf])

    for _ in range(0,ntrot):
        for i in range(0,N-1):
            timecirc.append(N_gate, [qr[i], qr[i+1]])

    timecirc.measure_all()

    from qiskit.quantum_info import SparsePauliOp

    pad = "I"
    for j in range(0,N-2):
        pad = pad + "I"
        
    Sop = [[SparsePauliOp.from_list([("Z"+pad, 0.5)])]]

    for i in range(0,N):
        if (i==N-1):
            pad = "I"
            for j in range(0,N-2):
                pad = pad + "I"
                
            Sop.append([SparsePauliOp.from_list([(pad+"Z", 0.5)])])
        
        else:
            if (i!=0):
                l = "I"
                for j in range(0,i-1):
                    l = l + "I"
        
                r = "I"
                for j in range(0,N-i-2):
                    r = r + "I"
                
                Sop.append([SparsePauliOp.from_list([(l + "Z" + r, 0.5)])])
                    
    from qiskit.primitives import StatevectorEstimator
    from qmiotools.integrations.qiskitqmio import FakeQmio
    from qiskit_aer import AerSimulator
    backend=FakeQmio("/opt/cesga/qmio/hpc/calibrations/2025_04_10__12_00_02.json",gate_error=True, readout_error=True, logging_level=logging.DEBUG)
    if ideal:
        backend = AerSimulator()

    counts=[]
    for t in tau_range:
        qc = timecirc.assign_parameters({'t':t})
        qc = transpile(qc, backend, optimization_level = 2)
        counts.append(backend.run(qc, shots = 1e6).result().get_counts())


    Szt = np.zeros((N,Nt))


    for i,c in zip(range(0,Nt), counts):
        simcounts = c
        keylist = list(simcounts.keys())
        for j in range(0,len(keylist)):
            state = keylist[j]
            
            for k in range(0,N):
                if (state[k]=='0'):
                    Szt[k,i] = Szt[k,i] + simcounts[state]
                else:
                    Szt[k,i] = Szt[k,i] - simcounts[state]

    return 0.5*Szt/(1e6) 

N = 8 # number of spins; scale up *slowly*, as computational load increases exponentially with N

Szt_1 = spin_chain(N, J=1,Jz=-0.1)
Szt_2 = spin_chain(N, J=1,Jz=-1)
Szt_3 = spin_chain(N, J=1,Jz=-5)

Szt_1i = spin_chain(N, J=1,Jz=-0.1, ideal=True)
Szt_2i = spin_chain(N, J=1,Jz=-1, ideal=True)
Szt_3i = spin_chain(N, J=1,Jz=-5, ideal=True)

plt.clf()

fig, axes = plt.subplots(2, 3, figsize=(11, 5), sharex=True, sharey=True, constrained_layout=True)

# Primera fila
ax = axes[0][0]; im1 = ax.imshow(Szt_1, aspect='equal', cmap='RdPu', vmin=-0.5, vmax=0.5); ax.set_title(r"$|J_z| = 0.1 J_{xy}$"); ax.set_ylabel("site j")
ax = axes[0][1]; im2 = ax.imshow(Szt_2, aspect='equal', cmap='RdPu', vmin=-0.5, vmax=0.5); ax.set_title(r"$|J_z| = J_{xy}$")
ax = axes[0][2]; im3 = ax.imshow(Szt_3, aspect='equal', cmap='RdPu', vmin=-0.5, vmax=0.5); ax.set_title(r"$|J_z| = 5 J_{xy}$")

# Segunda fila
ax = axes[1][0]; im4 = ax.imshow(Szt_1i, aspect='equal', cmap='RdPu', vmin=-0.5, vmax=0.5); ax.set_xlabel(r"$J_{xy} t$"); ax.set_ylabel("site j")
ax = axes[1][1]; im5 = ax.imshow(Szt_2i, aspect='equal', cmap='RdPu', vmin=-0.5, vmax=0.5); ax.set_xlabel(r"$J_{xy} t$")
ax = axes[1][2]; im6 = ax.imshow(Szt_3i, aspect='equal', cmap='RdPu', vmin=-0.5, vmax=0.5); ax.set_xlabel(r"$J_{xy} t$")

# Barra de color común horizontal

cbar = fig.colorbar(im6, ax=axes.ravel().tolist(), shrink=0.9, orientation="horizontal", pad=0.05)
cbar.set_label(r'$\langle \hat{S}_j^z \rangle / \hbar$')

# Guardar o mostrar
plt.show()
# plt.savefig("results/spin-chain_com.png", dpi=400)
plt.savefig(f"results/spin-chain_{N}_com.png", dpi=400)
