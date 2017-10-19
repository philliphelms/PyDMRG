close all; clear; clc;

% Parameters
L = 10;
alpha = 0.35;
beta = 2/3;
gamma = 0;
delta = 0;
q = 0;
p = 1;

s = linspace(-1,1,2);
E_vec = zeros(size(s));
for i = 1:length(s)
    E_vec(i) = sepEnergyCalc(L,p,alpha,beta,q,gamma,delta,s(i));
end
plot(s,E_vec/(L+1),'b.','MarkerSize',10)
dE = E_vec(2:end)-E_vec(1:end-1);
ds = s(2:end)-s(1:end-1);
dE_ds = dE./ds;
hold on;
plot(s(2:end),-dE_ds/(L+1),'r.','MarkerSize',10)
set(gca,'fontsize',16)
grid on;
%ylabel('$\frac{\mu}{\left(L+1\right)}$','fontsize',24,'interpreter','latex')
xlabel('$s$','interpreter','latex','fontsize',36)
lg = legend('$\mu\frac{1}{\left(L+1\right)}$','$\frac{\partial \mu}{\partial s}\frac{1}{\left(L+1\right)}$');
set(lg,'interpreter','latex')
set(lg,'fontsize',24)
set(lg,'location','westoutside')