from qmio import QmioRuntimeService
      
import numpy as np
from matplotlib import pyplot as plt
from qiskit import *
import logging

#two-gate operator for exp[i(α σx σx + β σy σy + γ σz σz)]
def N(α,β,γ,circ,q1,q2):
    circ.rz(-0.5*np.pi,q2)
    circ.cx(q2,q1)
    circ.rz(0.5*np.pi-2*γ,q1)
    circ.ry(2.0*α-0.5*np.pi,q2)
    circ.cx(q1,q2)
    circ.ry(0.5*np.pi-2.0*β,q2)
    circ.cx(q2,q1)
    circ.rz(0.5*np.pi,q1)

# construimos los circuitos para la estimacion de la energía
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qmiotools.integrations.qiskitqmio import FakeQmio

def energy_estimation(Jx,Jy,Jz):
    tau = Parameter('τ')
    # circuit for estimation of <SxSx>
    circx = QuantumCircuit(2)
    N(Jx*tau/4.0,Jy*tau/4.0,Jz*tau/4.0,circx,0,1)
    circx.ry(-np.pi/2,range(2)) # measure in x-axis
    circx.measure_all()

    # circuit for estimation of <SySy>
    circy = QuantumCircuit(2)
    N(Jx*tau/4.0,Jy*tau/4.0,Jz*tau/4.0,circy,0,1)
    circy.rz(-np.pi/2,range(2))
    circy.ry(-np.pi/2,range(2)) # measure in y-axis
    circy.measure_all()

    # circuit for estimation of <SxSx>
    circz = QuantumCircuit(2)
    N(Jx*tau/4.0,Jy*tau/4.0,Jz*tau/4.0,circz,0,1)
    circz.measure_all()

    circs = [circx,circy,circz]

    XX = SparsePauliOp.from_list([("XX", 0.25)])
    YY = SparsePauliOp.from_list([("YY", 0.25)])
    ZZ = SparsePauliOp.from_list([("ZZ", 0.25)])

    IX = SparsePauliOp.from_list([("IX", 0.5)])
    XI = SparsePauliOp.from_list([("XI", 0.5)])
    IY = SparsePauliOp.from_list([("IY", 0.5)])
    YI = SparsePauliOp.from_list([("YI", 0.5)])
    ZI = SparsePauliOp.from_list([("ZI", 0.5)])
    IZ = SparsePauliOp.from_list([("IZ", 0.5)])
    observables = [XX,YY,ZZ]; single_observables = [(IX,XI), (IY,YI), (IZ,ZI)]

    def expectation(counts, pauli, nshots):
        """Calcula el valor esperado de un observable de varias qubits,
        p.ej. 'ZZ', 'XI', 'IY', 'IXZ', etc.
        counts: dict mapping bitstrings ('010', '111', ...) a conteos
        pauli:  string con misma longitud que cada bitstring, usando 'I','X','Y','Z'
        n_shots: total de mediciones"""
        if len(pauli) == 0:
            raise ValueError("El string de pauli no puede estar vacío")
        expval = 0
        for bitstring, count in counts.items():
            if len(bitstring) != len(pauli):
                raise ValueError(f"Longitud de bitstring ({len(bitstring)}) "
                                f"y pauli ({len(pauli)}) deben coincidir")
            value = 1
            # recorremos desde el bit menos significativo hacia el más
            for i, p in enumerate(pauli):
                bit = bitstring[-1 - i]
                if p == 'I':
                    # identidad: no cambia el producto
                    continue
                elif p in ('Z', 'X', 'Y'):
                    # asumimos que la medición ya se hizo en la base adecuada
                    if bit == '1':
                        value *= -1
                else:
                    raise ValueError(f"Operador no reconocido '{p}' en pauli")
            expval += value * count

        return expval / nshots


    backend=FakeQmio("/opt/cesga/qmio/hpc/calibrations/2025_04_10__12_00_02.json",gate_error=True, readout_error=True, logging_level=logging.DEBUG)

    circuits = [transpile(circ,backend, optimization_level=2) for circ in circs]

    Nt = np.linspace(0,10,50)

    SxSx = np.zeros(50)
    SySy = np.zeros(50)
    SzSz = np.zeros(50)
    ISx = np.zeros(50); SxI=np.zeros(50); ISy=np.zeros(50); SyI=np.zeros(50); ISz =np.zeros(50); SzI=np.zeros(50)

    results = [SxSx, SySy, SzSz]; results_single = [(ISx,SxI), (ISy,SyI), (ISz,SzI)]

    nshots = 1e5

    for i,phi in enumerate(Nt):

        for circ,obs,res, singleobs,singleres in zip(circuits, observables,results, single_observables, results_single):
            
            # print("For obs: ", obs, "single obs: ", singleobs)

            circ = circ.assign_parameters([phi])

            counts = backend.run(circ, shots=nshots).result().get_counts()

            # print("res: ", res)
            res[i] = expectation(counts, obs.paulis.to_labels()[0], nshots) * obs.coeffs[0].real

            singleres[0][i] = expectation(counts, singleobs[0].paulis.to_labels()[0], nshots) * singleobs[0].coeffs[0].real
            singleres[1][i] = expectation(counts, singleobs[1].paulis.to_labels()[0], nshots) * singleobs[1].coeffs[0].real

    energy = []
    for rx,ry,rz in zip(results[0],results[1],results[2]):
        energy.append( -Jx*rx-Jy*ry-Jz*rz )

    proy_Sx = []; proy_Sy = []; proy_Sz = []
    for rx0,rx1,ry0,ry1,rz0,rz1 in zip(results_single[0][0],results_single[0][1],results_single[1][0],results_single[1][1],results_single[2][0],results_single[2][1]):
        proy_Sx.append(rx0+rx1)
        proy_Sy.append(ry0+ry1)
        proy_Sz.append(rz0+rz1)

    return energy,proy_Sx,proy_Sy,proy_Sz

# now theoretical #######################################################

def exact(Jx,Jy,Jz):
    print("calculando teorico...")

    tau = Parameter('τ')
    # circuit for estimation of <SxSx>
    circ = QuantumCircuit(2)
    N(Jx*tau/4.0,Jy*tau/4.0,Jz*tau/4.0,circ,0,1)

    Nt = 50

    phi_vals_ = np.linspace(0,10,Nt)

    # energy estimation

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

    energy = []
    for x,y,z in zip(xx,yy,zz):
        energy.append(-Jx*x-Jy*y-Jz*z)

    # spin obs estimation

    IX = SparsePauliOp.from_list([("IX", 0.5)])
    XI = SparsePauliOp.from_list([("XI", 0.5)])
    IY = SparsePauliOp.from_list([("IY", 0.5)])
    YI = SparsePauliOp.from_list([("YI", 0.5)])
    ZI = SparsePauliOp.from_list([("ZI", 0.5)])
    IZ = SparsePauliOp.from_list([("IZ", 0.5)])

    pub1 = (
        circ,  # circuit
        [[IX],[XI], [IY],[YI], [ZI],[IZ]],  # Observables
        phi_vals_,
    )

    job_result = estimator.run(pubs=[pub1]).result()

    xi_ix = job_result[0].data.evs[0]+job_result[0].data.evs[1]
    yi_iy = job_result[0].data.evs[2]+job_result[0].data.evs[3]
    zi_iz = job_result[0].data.evs[4]+job_result[0].data.evs[5]

    return energy, xi_ix, yi_iy, zi_iz



JxJy = energy_estimation(0.5,0.5,0.25)
JxJy_exact = exact(0.5,0.5,0.25)

JxJyJz = energy_estimation(0.5,0.5,0.5)
JxJyJz_exact = exact(0.5,0.5,0.5)

different = energy_estimation(0.5,-0.45,0.25)
different_exact = exact(0.5,-0.45,0.25)

Nt = np.linspace(0,10,50)



plt.clf()

plt.plot(Nt, JxJy[0],"o",alpha=0.7, label=r"$J_x=J_y\neq J_z$")
plt.plot(Nt, JxJyJz[0],"o",alpha=0.7, label=r"$J_x=J_y=J_z$")
plt.plot(Nt, different[0],"o",alpha=0.7, label=r"$J_x\neq J_y\neq J_z$")

plt.plot(Nt, JxJy_exact[0],"k-", linewidth =0.8)
plt.plot(Nt, JxJyJz_exact[0],"k-", linewidth =0.8)
plt.plot(Nt, different_exact[0],"k-", linewidth =0.8)

plt.grid(True); plt.legend(loc="lower right"); plt.xlabel("Jt"); plt.ylabel(r"$\langle H\rangle /\hbar^2$"); plt.title("Conservation of energy")
plt.show()
plt.savefig("./results/energy-conservation-fake.png", dpi=400)

plt.clf()

# plt.plot(Nt, JxJy[1],"o",alpha=0.5, color="red", label=r"$\langle S^xI+IS^x\rangle/\hbar^2$, $J_x=J_y\neq J_z$")
# plt.plot(Nt, JxJy[2],"*",alpha=0.5, color="red", label=r"$\langle S^yI+IS^y\rangle/\hbar^2$, $J_x=J_y\neq J_z$")
plt.plot(Nt, JxJy[3],"o",alpha=0.7, label=r"$\langle S^zI+IS^z\rangle/\hbar$, $J_x=J_y\neq J_z$")
plt.plot(Nt, JxJy_exact[3],"k-",linewidth = 0.8)

# plt.plot(Nt, JxJyJz[1],"o",alpha=0.5, color="green", label=r"$\langle S^xI+IS^x\rangle/\hbar^2$, $J_x=J_y=J_z$")
plt.plot(Nt, JxJyJz[2],"o",alpha=0.7, label=r"$\langle S^yI+IS^y\rangle/\hbar$, $J_x=J_y=J_z$")
plt.plot(Nt, JxJyJz_exact[2],"k-",linewidth = 0.8)
# plt.plot(Nt, JxJyJz[3],"^",alpha=0.5, color="green", label=r"$\langle S^zI+IS^z\rangle/\hbar^2$, $J_x=J_y=J_z$")

# plt.plot(Nt, different[1],"o",alpha=0.5, color="blue", label=r"$\langle S^xI+IS^x\rangle/\hbar^2$, $J_x\neq J_y\neq J_z$")
# plt.plot(Nt, different[2],"*",alpha=0.5, color="blue", label=r"$\langle S^yI+IS^y\rangle/\hbar^2$, $J_x\neq J_y\neq J_z$")
plt.plot(Nt, different[3],"o",alpha=0.7, label=r"$\langle S^zI+IS^z\rangle/\hbar$, $J_x\neq J_y\neq J_z$")
plt.plot(Nt, different_exact[3],"k-",linewidth = 0.8)
plt.grid(True); plt.legend(loc = "lower left"); plt.ylabel(r"$\langle S^iI+IS^i\rangle /\hbar$"); plt.xlabel("Jt"); plt.title("Symmetries")
plt.show()
plt.savefig("./results/symetries-fake.png", dpi=400)