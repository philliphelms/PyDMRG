function la = sepEnergyCalc(L,p,a,b,q,g,d,s)
%==========================
%Copyright (c) Ushnish Ray
%All rights reserved
%==========================

%L = 10;
%p = 1;
%a = 0.35; %Left insert 
%b = 2/3; %Right remove
beta = s;

clsz = L; %L must be a multiple of cluster size
%%
%q = 0; %1-p; %Left move
%g = 0; %1-a; %Left remove 
%d = 0; %1-b; %Right insert
%beta = 0.5*log((a*b)/(g*d)*(p/q)^(L-1))/(L+1);

%Current
pw = p*exp(-beta);
qw = q*exp(beta);
aw = a*exp(-beta);
bw = b*exp(-beta);
gw = g*exp(beta);
dw = d*exp(beta);

%Activity
%{
pw = p*exp(beta);
qw = q*exp(beta);
aw = a*exp(beta);
bw = b*exp(beta);
gw = g*exp(beta);
dw = d*exp(beta);
%}
%%
nv = zeros(L,1);
cdv = zeros(L,1);
cv = zeros(L,1);
sproj = zeros(L/clsz,2^clsz);
isproj = zeros(L/clsz,2^clsz);

%Initial Guess
for i=1:L
        nv(i) = 0.1;
        cdv(i) = 0.1;
        cv(i) = 0.1;
end

%%
%Construct Operators
ma = zeros(2,2);
ma(1,1) = -a;
ma(1,2) = gw;
ma(2,1) = aw;
ma(2,2) = -g;
ma = kron(ma,eye(2^(clsz-1),2^(clsz-1)));

mb = zeros(2,2);
mb(1,1) = -d;
mb(1,2) = bw;
mb(2,1) = dw;
mb(2,2) = -b;
mb = kron(eye(2^(clsz-1),2^(clsz-1)),mb);

mi = zeros(4,4);
mi(2,2) = -q;
mi(2,3) = pw;
mi(3,2) = qw;
mi(3,3) = -p;
if(clsz>2)
    mc = zeros(2^clsz,2^clsz);
    for s = 1:(clsz-1)
       ls = s-1;
       rs = clsz-(s+1);
       
       if(ls==0)
           piece = kron(mi,eye(2^rs));
       elseif(rs==0)
           piece = kron(eye(2^ls),mi);
       else
           piece = kron(kron(eye(2^ls),mi),eye(2^rs));
       end
       mc = mc + piece;
    end
else
    mc = mi;
end

%%
%Single site observables
cop = zeros(2^clsz,2^clsz,clsz);
cdop = zeros(2^clsz,2^clsz,clsz);
nop = zeros(2^clsz,2^clsz,clsz);

for s = 1:clsz
    ls = s-1;
    rs = clsz-s;
    if(ls==0)
        cop(:,:,s) = kron([0,1;0,0;],eye(2^rs));
        cdop(:,:,s) = kron([0,0;1,0;],eye(2^rs));
        nop(:,:,s) = kron([0,0;0,1;],eye(2^rs));
    elseif(rs==clsz)
        cop(:,:,s) = kron(eye(2^ls),[0,1;0,0;]);
        cdop(:,:,s) = kron(eye(2^ls),[0,0;1,0;]);
        nop(:,:,s) = kron(eye(2^ls),[0,0;0,1;]);
    else
        cop(:,:,s) = kron(kron(eye(2^ls),[0,1;0,0;]),eye(2^rs));
        cdop(:,:,s) = kron(kron(eye(2^ls),[0,0;1,0;]),eye(2^rs));
        nop(:,:,s) = kron(kron(eye(2^ls),[0,0;0,1;]),eye(2^rs));
    end
end
%%
maxitr = 1000;
if(L==clsz)
    maxitr = 1;
end

for iter = 1:maxitr
        
%    iter
    
    nv_new = zeros(L,1);
    cv_new = zeros(L,1);
    cdv_new = zeros(L,1);
    lam = zeros(L/clsz,2^clsz);
    Ms = zeros(2^clsz,2^clsz,L/clsz);
    for c = 1:L/clsz
        li = (c-1)*clsz+1
        ri = (c-1)*clsz+clsz
        
        %Left couple
        LC = zeros(2,2);
        if(c > 1)
            LC(1,1) = -p*nv(li-1);
            LC(1,2) = qw*cdv(li-1);
            LC(2,1) = pw*cv(li-1);
            LC(2,2) = -q*(1-nv(li-1));
            LC = kron(LC,eye(2^(clsz-1),2^(clsz-1)));
        else
            LC = ma;
        end
        
        
        %Right couple
        RC = zeros(2,2);
        if(c<L/clsz)
            RC(1,1) = -q*nv(ri+1);
            RC(1,2) = pw*cdv(ri+1);
            RC(2,1) = qw*cv(ri+1);
            RC(2,2) = -p*(1-nv(ri+1));
            RC = kron(eye(2^(clsz-1),2^(clsz-1)),RC);
        else
            RC = mb;
        end
        
   
        M = mc + LC + RC;
        
     %   mc
     %   LC
     %   RC
     %   LC+RC
     %   M
        
        %Diagonalize
        [RV,E,LV] = eig(M);
        [se,idx] = sort(diag(E));

        
        pp = -1; val = -100;
        for i = 1:(2^clsz)
            if(imag(E(i,i))<1.0e-10 && E(i,i)>val)
                pp = i;
                val = E(i,i);
            end
        end
        lam(c,1) = E(pp,pp);

        %Find gap         
        %if(iter == 1)
        %   ser = sort(diag(real(E)),'descend');         
        %   display(sprintf('%6.3e', ser(1)-ser(2)));
        %end
        
        j = 2;
        for i = 1:2^clsz
            if(i ~= pp)
                lam(c,j) = E(i,i);
                j = j + 1;
            end
        end
        
        %Remember to remove the constant terms
        %otherwise we will artifically end up raising the eigenvalue
        %For computing Doob transformed matrices not really need to 
        %worry, since we only care about the eigenvectors
        
        %Compute expectation values as needed from M
        IRV = inv(RV);
        lpsi = IRV(pp,:);
        rpsi = RV(:,pp);
        for s = 1:clsz
           nv_new((c-1)*clsz+s) = lpsi*nop(:,:,s)*rpsi;
           cv_new((c-1)*clsz+s) = lpsi*cop(:,:,s)*rpsi;
           cdv_new((c-1)*clsz+s) = lpsi*cdop(:,:,s)*rpsi;
        end
 
        sproj(c,:) = RV(:,pp);
        isproj(c,:) = IRV(pp,:);
        
        %{
        display(lam(c,1));
        display(nv_new)
        display(cv_new)
        display(cdv_new)
        pause
        %}
    end

    %lam(:,1)
    

    figure(2)
    %subplot(1,2,1);
    plot(1:L,nv_new,'-ob');
    %subplot(1,2,2);
    %plot(1:L,lam(:,1),'-ok',1:L,lam(:,2),'-xg');
    drawnow;

    
    nvdiff = norm(nv_new-nv);
    cvdiff = norm(cv_new-cv);
    cdvdiff = norm(cdv_new-cdv);
    
    nv = nv_new;
    cv = cv_new;
    cdv = cdv_new;
    
    %display(sprintf('%10.6e %10.6e %10.6e # %10.6e %10.6e %10.6e',cvdiff,cdvdiff,nvdiff,sum(nv_new)/L,sum(cdv_new)/L,sum(cv_new)/L));
        
    if(cvdiff<1.0e-7 && nvdiff<1.0e-7 && cdvdiff<1.0e-7)
        display(sprintf('%10.6e %10.6e %10.6e # %10.6e %10.6e %10.6e',cvdiff,cdvdiff,nvdiff,sum(nv_new)/L,sum(cdv_new)/L,sum(cv_new)/L));
        break
    end
end

%%

%Compute lambda

%Do intra cluster
la = 0.0;
for c = 1:L/clsz
   Lpsi = isproj(c,:); 
   Rpsi = sproj(c,:)';
   m = mc;
   if(c == 1)
       m = m + ma;
   end
   
   if(c == L/clsz)
      m = m + mb; 
   end
   
   la = la + Lpsi*m*Rpsi;
end

%Do inter cluster
if(clsz ~= L)
    mi_x = kron(eye(2^(clsz-1),2^(clsz-1)),mi);
    mi_x = kron(mi_x,eye(2^(clsz-1),2^(clsz-1)));
    for c = 1:L/clsz-1
       Rpsi = kron(sproj(c,:),sproj(c+1,:))';
       Lpsi = kron(isproj(c,:),isproj(c+1,:));

       la = la + Lpsi*mi_x*Rpsi;
    end
end

%%
fprintf('Energy is %0.5f\n',la);
%LAMT = sum(lam(:,1));
display(sprintf('R %f %16.9e %10.6e',beta,la,nvdiff));
%display(sprintf('%d %6.3e %16.9e %10.6e %10.6e %10.6e',iter,sum(nv_new),la,nvdiff,cvdiff,cdvdiff));
%%
%{
figure(2)
subplot(1,2,1);
plot(1:L,nv_new,'-ob');
subplot(1,2,2);
plot(1:L/2,lam(:,1),'-ok',1:L/2,lam(:,2),'-xg',1:L/2,lam(:,3),'-or',1:L/2,lam(:,4),'-xb');
%}
