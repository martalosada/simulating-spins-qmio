from qmio import QmioRuntimeService
      
import numpy as np
from matplotlib import pyplot as plt
from qiskit import *
import logging
from qiskit.circuit import Parameter

phi = Parameter('ϕ')
qr = QuantumRegister(2,'q')

nshots = 1e3

# circuit for <SxSx>
circx = QuantumCircuit(qr) 
circx.u(np.pi/2,phi,0,0)
circx.x(1)
circx.cx(0,1)
circx.ry(-np.pi/2,qr) # measure in x-axis
circx.measure_all()


# circuit for <SySy>
circy = QuantumCircuit(qr) 
circy.u(np.pi/2,phi,0,0)
circy.x(1)
circy.cx(0,1)
circy.rz(-np.pi/2,qr)
circy.ry(-np.pi/2,qr) # measure in y-axis
circy.measure_all()


# circuit for <SzSz>
circz = QuantumCircuit(qr) 
circz.u(np.pi/2,phi,0,0)
circz.x(1)
circz.cx(0,1)
# measure in z-axis
circz.measure_all()


circuits = [circx,circy,circz]


Nphi = 50

phi_vals = np.linspace(0,np.pi,Nphi)

from qiskit.quantum_info import SparsePauliOp

XX = SparsePauliOp.from_list([("XX", 0.25)])
YY = SparsePauliOp.from_list([("YY", 0.25)])
ZZ = SparsePauliOp.from_list([("ZZ", 0.25)])

observables = [XX,YY,ZZ]

def expectation(counts, pauli, n_shots):
    """Calcula el valor esperado de un observable tipo ZZ, XX o YY."""
    print("Abou to calc for pauli: ", pauli)
    expval = 0
    for bitstring, count in counts.items():
        # bitstring: tipo '00', '01', etc.
        value = 1
        for i, p in enumerate(pauli):
            if p == 'Z':
                if bitstring[-1-i] == '1':
                    value *= -1
            elif p == 'X':
                # Suponemos medición previa en base-X (medido como en Z)
                if bitstring[-1-i] == '1':
                    value *= -1
            elif p == 'Y':
                # Suponemos medición previa en base-Y (medido como en Z)
                if bitstring[-1-i] == '1':
                    value *= -1
        expval += value * count
    return expval / n_shots

from qmiotools.integrations.qiskitqmio import FakeQmio

backend=FakeQmio("/opt/cesga/qmio/hpc/calibrations/2025_04_10__12_00_02.json",gate_error=True, readout_error=True, logging_level=logging.DEBUG)

circuits = [transpile(circ,backend, optimization_level=2) for circ in circuits]


SxSx = np.zeros(Nphi)
SySy = np.zeros(Nphi)
SzSz = np.zeros(Nphi)

results = [SxSx, SySy, SzSz]

for i,phi in enumerate(phi_vals):

    for circ,obs, res in zip(circuits, observables,results):

        circ = circ.assign_parameters([phi])

        counts = backend.run(circ, shots=nshots).result().get_counts()

        res[i] = expectation(counts, obs.paulis.to_labels()[0], nshots) * obs.coeffs[0].real


# now theoretical #######################################################

print("calculando teorico...")

phi = Parameter('ϕ')
Nphi = 100

phi_vals_ = np.linspace(0,np.pi,Nphi)

qr = QuantumRegister(2,'q')

circ = QuantumCircuit(qr) 
circ.u(np.pi/2,phi,0,0)
circ.x(1)
circ.cx(0,1)

from qiskit.quantum_info import SparsePauliOp

XX = SparsePauliOp.from_list([("XX", 0.25)])
YY = SparsePauliOp.from_list([("YY", 0.25)])
ZZ = SparsePauliOp.from_list([("ZZ", 0.25)])

from qiskit.primitives import StatevectorEstimator

estimator = StatevectorEstimator()

pub1 = (
    circ,  # circuit
    [[XX], [YY], [ZZ]],  # Observables
    phi_vals_,
)

job_result = estimator.run(pubs=[pub1]).result()

xx = job_result[0].data.evs[0]
yy = job_result[0].data.evs[1]
zz = job_result[0].data.evs[2]



plt.clf()

plt.plot(phi_vals,results[0],"o", alpha=0.7, label = r"$\langle \hat{S}^x\hat{S}^x\rangle/\hbar^2$")
plt.plot(phi_vals,results[1],"o", alpha=0.7, label = r"$\langle \hat{S}^y\hat{S}^y\rangle/\hbar^2$")
plt.plot(phi_vals,results[2],"o", alpha=0.7, label = r"$\langle \hat{S}^z\hat{S}^z\rangle/\hbar^2$")

plt.plot(phi_vals_,xx,'k-', linewidth = 1)
plt.plot(phi_vals_,yy,'k-', linewidth = 1)
plt.plot(phi_vals_,zz,'k-', linewidth = 1)

plt.xticks(ticks=np.linspace(0,np.pi,5), labels=["0", "π/4", "π/2", "3π/4", "π"])
plt.grid(True); plt.xlabel("ϕ"); plt.ylabel(r"$\langle \hat{S}^i\hat{S}^i\rangle/\hbar^2$")
plt.title(r"$|\psi\rangle = |01\rangle + e^{i\phi}|10\rangle $")
plt.legend()
plt.show()
plt.savefig("./results/entangled-state-fake.png", dpi=400)


hbar = 1

def totalspin(sxx,syy,szz):
    return 1.5*hbar + 2*(sxx+syy+szz)

theoretical_S_total = [totalspin(sxx,syy,szz) for sxx,syy,szz in zip(xx,yy,zz)]

exp_S_total = [totalspin(sxx,syy,szz) for sxx,syy,szz in zip(results[0], results[1], results[2])]

plt.clf()

plt.plot(phi_vals_,theoretical_S_total,"k--", label="Theoretical")
plt.plot(phi_vals,exp_S_total,"o",alpha = 0.7,label="Estimated")

plt.xticks(ticks=np.linspace(0,np.pi,5), labels=["0", "π/4", "π/2", "3π/4", "π"])
plt.xlabel(r"$\phi$"); plt.ylabel(r"$\langle\hat{S}^2\rangle/\hbar^2$"); plt.legend()
plt.title("Total spin estimation")
plt.grid(True)
plt.show()
plt.savefig("./results/total-spin-2-fake.png", dpi=400)
