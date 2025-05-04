clear all
tic
N = 20;
D = 2^N;

%H for TIME EVOLUTION
Hi = zeros(floor(0.5*N)*D,1); %to store i index for nonzero elements
Hj = zeros(floor(0.5*N)*D,1); %to store j index for nonzero elements
Hh = zeros(floor(0.5*N)*D,1); %stores value of H(i,j)
Hk = 1;                       %counter of nonzero elements

J = 1.0;  %exchange interaction  
V = 10.0; %nearest-neighbor "basic" interaction; leave zero

%build hash list to map states to indices
Ao = dec2bin(0:2^N-1);
A = zeros(2^N,length(Ao(1,:)));
T = zeros(D,1);
p = sqrt(100*(1:N)+3);

for i = 1:D
    for j = 1:length(Ao(1,:))
        A(i,j) = bin2dec(Ao(i,j));
    end
    T(i) = sum(A(i,:).*p);
end
[T,id] = sort(T);

%builds domain wall statevector |+++...+-...--->
psi = zeros(2^N,1);
psi(2^N-(2^(N/2))+1)=1;

%loop to build H
for i = 1:D    
    A1 = A(i,:);
    for j = 1:N
        j1 = j+1;
        if j1<=N
            %nearest-neighbor hopping
            if (A1(j)==0) && (A1(j1)==1)
                Ao = A1;
                Ao(j)=1;
                Ao(j1)=0;
                To = sum(Ao.*p);
                idx = ismembc2(To,T);
                %if idx>0
                %H(i,id(idx)) = H(i,id(idx))-0.5*J;
                Hi(Hk) = i;
                Hj(Hk) = id(idx);
                Hh(Hk) = -0.5*J;
                Hk = Hk + 1;
            elseif (A1(j)==1) && (A1(j1)==0)
                Ao = A1;
                Ao(j)=0;
                Ao(j1)=1;
                To = sum(Ao.*p);
                idx = ismembc2(To,T);
                %if idx>0
                %H(i,id(idx)) = H(i,id(idx))-0.5*J;
                Hi(Hk) = i;
                Hj(Hk) = id(idx);
                Hh(Hk) = -0.5*J;
                Hk = Hk + 1;
            
                elseif (A1(j)==1) && (A1(j1)==1)
                %nearest-neighbor interaction (Jz = V)
                Hi(Hk) = i;
                Hj(Hk) = i;
                Hh(Hk) = V;
                Hk = Hk + 1;       
            end
        end
    end
end

%define sparse matrix for H
H = sparse(Hi(1:Hk-1),Hj(1:Hk-1),Hh(1:Hk-1),D,D);

toc
tic

Nt = 100; %time steps
tmax = 10; %max time
t = linspace(0,tmax,Nt);
psit = zeros(D,Nt); %to store |psi(t)>

for n = 1:Nt
    %|psi(t)> = exp(-iHt)|psi(0)>
    psit(:,n) = expv(-1i*t(n),H,psi);
end

%array to store number density n = <sz> + 1/2
nj = zeros(N,Nt);

for j = 1:N
    nji = zeros(floor(0.5*N)*D,1);
    njj = zeros(floor(0.5*N)*D,1);
    njv = zeros(floor(0.5*N)*D,1);
    njk = 1;    
    
    for i = 1:D %select state
        if (j>1) && (j<N)
            if (A(i,j-1)==0) && (A(i,j+1)==1)
                Ao = A(i,:);
                Ao(j-1)=1;
                Ao(j+1)=0;
                To = sum(p.*Ao);
                idx = ismembc2(To,T);
                
            end
        end
        if A(i,j)==1
            nji(njk) = i;
            njj(njk) = i;
            njv(njk) = 1;
            njk = njk + 1;
            %nj_op(i,i) = nj_op(i,i) + 1;
        end
    end
    
    nj_op = sparse(nji(1:njk-1),njj(1:njk-1),njv(1:njk-1),D,D);
    
    for n = 1:Nt
        nj(j,n) = (psit(:,n)')*nj_op*psit(:,n);
    end
end
    
toc

figure(2)
imagesc(nj-0.5)
colorbar
colormap('turbo')
title('<Sz>')
