from qmio import QmioRuntimeService
      
import numpy as np
from matplotlib import pyplot as plt
from qiskit import *
import logging
from qiskit_aer import AerSimulator
'expectation value of an operator in a state'
exp_val = lambda A, state: np.dot(state.conj().T,np.dot(A, state))

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

    ntrot = 100
    tau = Parameter('τ')
    qr = QuantumRegister(3,'q')
    cr = ClassicalRegister(3,'c')

    timecirc = QuantumCircuit(qr) 

    #initial states
    timecirc.u(np.pi/6,np.pi/3,0,0)
    timecirc.u(3*np.pi/5,4*np.pi/3,0,1)
    timecirc.u(-np.pi/5,2*np.pi/3,0,2)

    for i in range(0,ntrot):
        N(Jx*(tau/ntrot)/4.0,Jy*(tau/ntrot)/4.0,Jz*(tau/ntrot)/4.0,timecirc,0,1)
        N(Jx*(tau/ntrot)/4.0,Jy*(tau/ntrot)/4.0,Jz*(tau/ntrot)/4.0,timecirc,1,2)
        

    # circuit for estimation of <SxSxI>, <ISxSx>
    circx = timecirc.copy()
    circx.ry(-np.pi/2,range(2)) # measure in x-axis
    circx.measure_all()

    # circuit for estimation of <SySyI>, <ISySy>
    circy = timecirc.copy()
    circy.rz(-np.pi/2,range(2))
    circy.ry(-np.pi/2,range(2)) # measure in y-axis
    circy.measure_all()

    # circuit for estimation of <SzSzI>, <ISzSz>
    circz = timecirc.copy()
    N(Jx*tau/4.0,Jy*tau/4.0,Jz*tau/4.0,circz,0,1)
    circz.measure_all()

    circs = [circx,circy,circz]

    XXI = SparsePauliOp.from_list([("XXI", 0.25)])
    YYI = SparsePauliOp.from_list([("YYI", 0.25)])
    ZZI = SparsePauliOp.from_list([("ZZI", 0.25)])
    IXX = SparsePauliOp.from_list([("IXX", 0.25)])
    IYY = SparsePauliOp.from_list([("IYY", 0.25)])
    IZZ = SparsePauliOp.from_list([("IZZ", 0.25)])

    IIX = SparsePauliOp.from_list([("IIX", 0.5)])
    XII = SparsePauliOp.from_list([("XII", 0.5)])
    IXI = SparsePauliOp.from_list([("IXI", 0.5)])
    IIY = SparsePauliOp.from_list([("IIY", 0.5)])
    YII = SparsePauliOp.from_list([("YII", 0.5)])
    IYI = SparsePauliOp.from_list([("IYI", 0.5)])
    IIZ = SparsePauliOp.from_list([("IIZ", 0.5)])
    ZII = SparsePauliOp.from_list([("ZII", 0.5)])
    IZI = SparsePauliOp.from_list([("IZI", 0.5)])
    observables = [(XXI,IXX),(YYI,IYY),(ZZI,IZZ)]; single_observables = [(XII,IXI,IIX), (YII,IYI,IIY),(ZII,IZI,IIZ)]

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
    backend = AerSimulator()
    circuits = [transpile(circ,backend, optimization_level=2) for circ in circs]

    Nt = np.linspace(0,10,50)

    sXXI = np.zeros(len(Nt))
    sYYI = np.zeros(len(Nt))
    sZZI = np.zeros(len(Nt))
    sIXX = np.zeros(len(Nt))
    sIYY = np.zeros(len(Nt))
    sIZZ = np.zeros(len(Nt))

    sIIX = np.zeros(len(Nt))
    sXII = np.zeros(len(Nt))
    sIXI = np.zeros(len(Nt))
    sIIY = np.zeros(len(Nt))
    sYII = np.zeros(len(Nt))
    sIYI = np.zeros(len(Nt))
    sIIZ = np.zeros(len(Nt))
    sZII = np.zeros(len(Nt))
    sIZI = np.zeros(len(Nt))

    results = [(sXXI, sIXX), (sYYI, sIYY),(sZZI, sIZZ)]; results_single = [(sIIX, sIXI, sXII), (sIIY, sIYI, sYII), (sIIZ, sIZI, sZII)]

    nshots = 1e5

    for i,phi in enumerate(Nt):

        for circ,obs,res, singleobs,singleres in zip(circuits, observables,results, single_observables, results_single):
            
            print("For obs: ", obs, "single obs: ", singleobs)
            circ = circ.assign_parameters([phi])

            counts = backend.run(circ, shots=nshots).result().get_counts()

            # print("res: ", res)

            for pauli, r in zip(obs,res):
                r[i] = expectation(counts, pauli.paulis.to_labels()[0], nshots) * pauli.coeffs[0].real

            for spauli, sr in zip(singleobs, singleres):
                print("For pauli ", spauli)
                sr[i] = expectation(counts, spauli[0].paulis.to_labels()[0], nshots) * spauli.coeffs[0].real

    energy = []
    print(len(results))
    for xxi,ixx,yyi,iyy,zzi,izz in zip(results[0][0],results[0][1],results[1][0],results[1][1],results[2][0], results[2][1]):
        energy.append( -Jx*(xxi+ixx)-Jy*(yyi+iyy)-Jz*(zzi+izz) )

    proy_Sx = []; proy_Sy = []; proy_Sz = []
    for iix,ixi,xii,iiy,iyi,yii,iiz,izi,zii in zip(results_single[0][0],results_single[0][1],results_single[0][2],results_single[1][0],results_single[1][1],results_single[1][2],results_single[2][0], results_single[2][1], results_single[2][2]):
        proy_Sx.append(iix+ixi+xii)
        proy_Sy.append(iiy+iyi+yii)
        proy_Sz.append(iiz+izi+zii)

    return energy,proy_Sx,proy_Sy,proy_Sz

# now theoretical #######################################################
from scipy.linalg import expm
from qiskit.quantum_info import Statevector
def exact(Jx,Jy,Jz):

    XXI = SparsePauliOp.from_list([("XXI",0.25),("IXX",0.25)])
    YYI = SparsePauliOp.from_list([("YYI",0.25),("IYY",0.25)])
    ZZI = SparsePauliOp.from_list([("ZZI",0.25),("IZZ",0.25)])

    H = -Jx*(XXI)-Jy*(YYI)-Jz*(ZZI)

    Hamil = H.to_matrix()


    print("calculando teorico...")

    # circuit for estimation of <SxSx>
    circ = QuantumCircuit(3) 

    #initial states
    circ.u(np.pi/6,np.pi/3,0,0)
    circ.u(3*np.pi/5,4*np.pi/3,0,1)
    circ.u(-np.pi/5,2*np.pi/3,0,2)

    statevector = Statevector.from_instruction(circ).data
    
    Nt = 50
    phi_vals_ = np.linspace(0,10,Nt)

    # spin obs estimation

    Sxop = SparsePauliOp.from_list([("IIX",0.5),("IXI",0.5),("XII",0.5)])
    Syop = SparsePauliOp.from_list([("IIY",0.5),("IYI",0.5),("YII",0.5)])
    Szop = SparsePauliOp.from_list([("IIZ",0.5),("IZI",0.5),("ZII",0.5)])

    # energy estimation
    energy = [];Sx=[];Sy=[];Sz=[]
    for t in phi_vals_:
        U = expm(-1j * Hamil * t)
        evolved_state = np.dot(U,statevector)

        energy.append(np.real(  exp_val(Hamil,evolved_state)    ))
        Sx.append(np.real(exp_val(Sxop.to_matrix(),evolved_state)))
        Sy.append(np.real(exp_val(Syop.to_matrix(),evolved_state)))
        Sz.append(np.real(exp_val(Szop.to_matrix(),evolved_state)))


    return energy, Sx,Sy,Sz



JxJy = energy_estimation(0.5,0.5,0.25)
JxJy_exact = exact(0.5,0.5,0.25)

JxJyJz = energy_estimation(0.5,0.5,0.5)
JxJyJz_exact = exact(0.5,0.5,0.5)

different = energy_estimation(0.5,-0.45,0.25)
different_exact = exact(0.5,-0.45,0.25)

Nt = np.linspace(0,10,50)

print(JxJy[0])

plt.clf()

plt.plot(Nt, JxJy[0],"o",alpha=0.7, label=r"$J_x=J_y\neq J_z$")
plt.plot(Nt, JxJyJz[0],"o",alpha=0.7, label=r"$J_x=J_y=J_z$")
plt.plot(Nt, different[0],"o",alpha=0.7, label=r"$J_x\neq J_y\neq J_z$")

plt.plot(Nt, JxJy_exact[0],"k-", linewidth =0.8)
plt.plot(Nt, JxJyJz_exact[0],"k-", linewidth =0.8)
plt.plot(Nt, different_exact[0],"k-", linewidth =0.8)

plt.grid(True); plt.legend(loc="lower right"); plt.xlabel("Jt"); plt.ylabel(r"$\langle H\rangle /\hbar^2$"); plt.title("Conservation of energy")
plt.show()
plt.savefig("./results/energy-conservation-three-aer.png", dpi=400)

plt.clf()

# plt.plot(Nt, JxJy[1],"o",alpha=0.5, color="red", label=r"$\langle S^xI+IS^x\rangle/\hbar^2$, $J_x=J_y\neq J_z$")
# plt.plot(Nt, JxJy[2],"*",alpha=0.5, color="red", label=r"$\langle S^yI+IS^y\rangle/\hbar^2$, $J_x=J_y\neq J_z$")
plt.plot(Nt, JxJy[3],"o",alpha=0.7, label=r"$\langle S^zI+IS^z\rangle/\hbar$, $J_x=J_y\neq J_z$")
plt.plot(Nt, JxJy_exact[3],"b-",linewidth = 0.8)

# plt.plot(Nt, JxJyJz[1],"o",alpha=0.5, color="green", label=r"$\langle S^xI+IS^x\rangle/\hbar^2$, $J_x=J_y=J_z$")
plt.plot(Nt, JxJyJz[2],"o",alpha=0.7, label=r"$\langle S^yI+IS^y\rangle/\hbar$, $J_x=J_y=J_z$")
plt.plot(Nt, JxJyJz_exact[2],"-",color="orange",linewidth = 0.8)
# plt.plot(Nt, JxJyJz[3],"^",alpha=0.5, color="green", label=r"$\langle S^zI+IS^z\rangle/\hbar^2$, $J_x=J_y=J_z$")

# plt.plot(Nt, different[1],"o",alpha=0.5, color="blue", label=r"$\langle S^xI+IS^x\rangle/\hbar^2$, $J_x\neq J_y\neq J_z$")
# plt.plot(Nt, different[2],"*",alpha=0.5, color="blue", label=r"$\langle S^yI+IS^y\rangle/\hbar^2$, $J_x\neq J_y\neq J_z$")
plt.plot(Nt, different[3],"o",alpha=0.7, label=r"$\langle S^zI+IS^z\rangle/\hbar$, $J_x\neq J_y\neq J_z$")
plt.plot(Nt, different_exact[3],"g-",linewidth = 0.8)
plt.grid(True); plt.legend(loc = "lower left"); plt.ylabel(r"$\langle S^iI+IS^i\rangle /\hbar$"); plt.xlabel("Jt"); plt.title("Symmetries")
plt.show()
plt.savefig("./results/symetries-three-aer.png", dpi=400)