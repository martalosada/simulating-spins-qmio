from qiskit.circuit import QuantumCircuit
from qiskit import transpile
from qmiotools.integrations.qiskitqmio import QmioBackend

backend=QmioBackend() # loads the last calibration file from the directory $QMIO_CALIBRARTIONS

# Creates a circuit qwith 2 qubits
c=QuantumCircuit(3,3)
c.u(2*tau,np.pi,np.pi,qr) #apply exp(-iHt/ħ)
c.barrier(qr)
c.ry(-np.pi/2,2) #rotation to measure <Sx>

c.rz(-np.pi/2,1) 
c.ry(-np.pi/2,1) #rotation to measure <Sy>
c.measure_all()

#Transpile the circuit using the optimization_level equal to 2
c=transpile(c,backend,optimization_level=2)

#Execute the circuit with 1000 shots. Must be executed from a node with a QPU.
job=backend.run(c,shots=1000)

#Return the results
print(job.result().get_counts())