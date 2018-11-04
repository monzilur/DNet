function [f, dfdtheta] = loss_function_mDNet(theta, v, args, test)
% [f, dfdtheta] = loss_function_DNet(theta, v, args,test)
% You can chosse between L1 or L2 regularisation
% DYNAMIC NETWORK with the membrane time constant on the weighted average of
% inputs
% define an objective function and gradient
%---------------------------------------------------

%check if doing testing rather than training
if nargin<4
    test = 0;
end

% get the data
vin = v{1};
fq = size(vin,1)*size(vin,2);
T = size(vin,3);
vin = reshape(vin,fq,T);
vout = v{2};

% get other options
lam = args{1}; % regularization constant
regtype = args{2}; % regularization type
tautype = args{3};
nonlin = args{4};

% get the variables (weights and biases)
W_jk = theta{1};
W_ij = theta{2};
b_k = theta{3};
b_j = theta{4};
d_k = theta{5};
d_tau_k = ftau(d_k,tautype);
d_j = theta{6};
d_tau_j = ftau(d_j,tautype);
delay = floor(theta{7});

% get the network structures
I = fq; % number input units
J = size(W_ij,1); % number hidden units
K = size(W_jk,1); % number of output units

%-------------------------------------------
% Gradients
u_previous = zeros(J,1);
v_previous = zeros(K,1);
dv_hat_dw_jk_previous = zeros(J,1);
du_dw_previous = zeros(I,J);
dv_hat_dw_ij_previous = zeros(I,J);
dv_hat_db_k_previous = zeros(K,1);
du_j_db_j_previous = zeros(J,K);
dv_hat_db_j_previous = zeros(J,K);
dv_hat_dd_k_previous = zeros(K,1);
du_j_dd_j_previous = zeros(J,K);
dv_hat_dd_j_previous = zeros(J,K);

u_j=zeros(J,T);
v_hat=zeros(K,T);
dv_hat_dw_jk=zeros(J,T);
du_j_dw_ij=zeros(I,J,T);
dv_hat_dw_ij=zeros(I,J,T);
dv_hat_db_k=zeros(K,T);
du_j_db_j=zeros(J,T);
dv_hat_db_j=zeros(J,T);
dv_hat_dd_k=zeros(K,T);
du_j_dd_j=zeros(J,T);
dv_hat_dd_j=zeros(J,T);

for t = delay+1:T

z_j = W_ij * vin(:,t-delay) + b_j;
uact(:,t) = z_j;
fsig_zj=fnonlin(z_j,nonlin);
u_j(:,t) = (1-d_tau_j).*u_previous + d_tau_j.*fsig_zj;

z_k = W_jk*u_j(:,t) + b_k;
vact(:,t) = z_k;
fsig_zk=fnonlin(z_k,nonlin);
v_hat(:,t) = (1-d_tau_k).*v_previous + d_tau_k.*fsig_zk;


% part of the derivatives
fprime_zk = fprimenonlin(z_k,nonlin);
fprime_ij = fprimenonlin(z_j,nonlin);

%output weights
dv_hat_dw_jk(:,t) = (1-d_tau_k)*dv_hat_dw_jk_previous + d_tau_k*u_j(:,t)*fprime_zk; % dv_hat_dw_jk
dv_hat_dw_jk_previous = dv_hat_dw_jk(:,t);

%backpropagation
du_j_dw_ij(:,:,t) = bsxfun(@times,(1-d_tau_j)',du_dw_previous) + bsxfun(@times,d_tau_j',vin(:,t-delay)*fprime_ij');
du_dw_previous = du_j_dw_ij(:,:,t);

%input weights
dv_hat_dw_ij(:,:,t) = (1-d_tau_k)*dv_hat_dw_ij_previous + d_tau_k*fprime_zk*bsxfun(@times,W_jk,du_j_dw_ij(:,:,t)); % dv_hat_dw_ij
dv_hat_dw_ij_previous = dv_hat_dw_ij(:,:,t);

%output bias
dv_hat_db_k(:,t) = (1-d_tau_k)*dv_hat_db_k_previous + d_tau_k*fprime_zk;
dv_hat_db_k_previous = dv_hat_db_k(:,t);

%input bias
du_j_db_j(:,t) = (1-d_tau_j).*du_j_db_j_previous + d_tau_j.*fprime_ij;
du_j_db_j_previous = du_j_db_j(:,t);
dv_hat_db_j(:,t) = (1-d_tau_k)*dv_hat_db_j_previous + d_tau_k*fprime_zk*(W_jk'.*du_j_db_j(:,t));
dv_hat_db_j_previous = dv_hat_db_j(:,t);

%time constants
primetau_dk=fprimetau(d_k,tautype);
primetau_dj=fprimetau(d_j,tautype);

%output time constants
dv_hat_dd_k(:,t) = (1-d_tau_k)*dv_hat_dd_k_previous - v_previous*primetau_dk + fsig_zk*primetau_dk;
dv_hat_dd_k_previous = dv_hat_dd_k(:,t);

%input time constants
du_j_dd_j(:,t) = (1-d_tau_j).*du_j_dd_j_previous - u_previous.*primetau_dj + fsig_zj.*primetau_dj;
du_j_dd_j_previous = du_j_dd_j(:,t);
dv_hat_dd_j(:,t) = (1-d_tau_k)*dv_hat_dd_j_previous + d_tau_k*fprime_zk*(W_jk'.*du_j_dd_j(:,t));
dv_hat_dd_j_previous = dv_hat_dd_j(:,t);

u_previous = u_j(:,t);
v_previous = v_hat(:,t);

end
%get the squared error
a_t = v_hat - vout;
funreg = (0.5*sum(a_t.^2))/T;

%regularize
switch regtype
    
    case 'sq'
regul = lam.*sum(W_jk(:).^2) + lam.*sum(W_ij(:).^2);
    case 'abs'
regul = lam.*sum(abs(W_jk(:))) + lam.*sum(abs(W_ij(:)));
    case 'none'
regul = 0; 

end

f = funreg + regul; %funreg already normalised for amount of data

% derivatives of output weights and biases
dEdW_jk = sum(bsxfun(@times,dv_hat_dw_jk,a_t),2)/T;
dEdW_jk = dEdW_jk';

dEdW_ij = zeros(I,J);
for ii = 1:T;
dEdW_ij = dEdW_ij + (dv_hat_dw_ij(:,:,ii)*a_t(ii));
end
dEdW_ij = dEdW_ij'/T;

dEdb_k = (dv_hat_db_k*a_t')/T;
dEdb_j = (dv_hat_db_j*a_t')/T;

dEdd_k = (dv_hat_dd_k*a_t')/T;
dEdd_j = (dv_hat_dd_j*a_t')/T;

% derivative of regularization
switch regtype
    case 'sq'
dEdW_jk = dEdW_jk + 2.*lam.*W_jk;
dEdW_ij = dEdW_ij + 2.*lam.*W_ij;
%dEdb = dEdb + 2.*lam.*b_k;
%dEdd_k = dEdd_k + 2.*lam.*d_k;
%dEdd_j = dEdd_j + 2.*lam.*d_j;
    case 'abs'
dEdW_jk = dEdW_jk + lam.*sign(W_jk);
dEdW_ij = dEdW_ij + lam.*sign(W_ij);
%dEdb = dEdb + lam.*sign(b_k);    
%dEdd_k = dEdd_k + lam.*sign(d_k); 
%dEdd_j = dEdd_j + lam.*sign(d_j);
end

ddelay = 0;

% give the gradients the same order as the parameters
dfdtheta = {dEdW_jk, dEdW_ij, dEdb_k, dEdb_j, dEdd_k, dEdd_j, ddelay};

%----------------------------------------------
% If testing rather than training give the predictions and other
% characteristics of the network
if test
    output.f = f;
    output.funreg = funreg;
    output.vout = vout;
    output.vin = vin; 
    output.vhat = v_hat; 
    output.uout = u_j;
    output.uact = uact;
    output.vact = vact;
    f = output; 
    
end