clear all

%couplings
Jx = 0.5;
Jy = -0.45;
Jz = 0.25;

%vector rep for |+> (up) and |-> (down)
up = [1; 0];
down = [0;1];

%theta angles
th1 = pi/6;
th2 = 3*pi/5;
th3 = -pi/5;

%phi angles
ph1 = pi/3;
ph2 = 4*pi/3;
ph3 = 2*pi/3;

%|psi> = cos(theta/2)|+> + e^(i phi)sin(theta/2)|-> 
psi1 = [cos(th1/2); exp(1i*ph1)*sin(th1/2)];
psi2 = [cos(th2/2); exp(1i*ph2)*sin(th2/2)];
psi3 = [cos(th3/2); exp(1i*ph3)*sin(th3/2)];

%tensor product represented by Kronecker product of vectors
psi = kron(psi1,kron(psi2,psi3));

%initialize H
H = zeros(2^3,2^3);

%Pauli spin operators, so = identity(2x2)
sx = 0.5*[[0,1];[1,0]];
sy = 0.5*[[0,-1i];[1i,0]];
sz = 0.5*[[1,0];[0,-1]];
so = [[1,0];[0,1]];

H = H + -Jx*kron(so,kron(sx,sx));
H = H + -Jx*kron(sx,kron(sx,so));

H = H + -Jy*kron(so,kron(sy,sy));
H = H + -Jy*kron(sy,kron(sy,so));

H = H + -Jz*kron(so,kron(sz,sz));
H = H + -Jz*kron(sz,kron(sz,so));

%number of time steps
Nt = 100;

%time step size
dt = 0.1;

%array to store |psi(t)>
psit = zeros(2^3,Nt);

%initialize first column as |psi(0)>
psit(:,1) = psi;

for i = 1:Nt
    t = (i-1)*dt;
    %time evolve state |psi(t)> = exp(-iHt)|psi(0)>
    %expm(A) performs matrix exponentiation of matrix A
    psit(:,i) = expm(-1i*t*H)*psi;
end

%arrays to store spin expectation values
Szt = zeros(3,Nt);
Sz3 = kron(sz,kron(so,so));
Sz2 = kron(so,kron(sz,so));
Sz1 = kron(so,kron(so,sz));


for i = 1:Nt
    %<Sz1(t)> = <psi(t)| Sz1 |psi(t)>
    Szt(1,i) = (psit(:,i)')*(Sz1*psit(:,i));
    Szt(2,i) = (psit(:,i)')*(Sz2*psit(:,i));
    Szt(3,i) = (psit(:,i)')*(Sz3*psit(:,i));
    
end

%negligible imaginary parts can be trimmed
Szt = real(Szt);
tspan = dt*(0:Nt-1);

%plot expectation values
figure(1)
plot(tspan,Szt(1,:),'r--')
hold on
plot(tspan,Szt(2,:),'b-.')
plot(tspan,Szt(3,:),'k-')
hold off
xlabel('Jt')
ylabel('<S_{i}^{z}')
legend('i=1','i=2','i=3')