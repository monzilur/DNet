function [f, dfdtheta] = loss_function_sDNet(theta, v, args, test)
% [f, dfdtheta] = loss_function_DNet(theta, v, args,test)
% You can chosse between L1 or L2 regularisation
% A variant of DYNAMIC NETWORK with the synaptic time constant
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
J = length(b_j); % number hidden units
K = length(b_k); % number of output units

%-------------------------------------------
% objective function
vact_j_previous = zeros(J,1);
vact_k_previous = zeros(K,1);
dvact_k_dw_ij_previous = zeros(I,J);
dvact_j_dw_ij_previous = zeros(I,J);
dvact_k_dw_jk_previous = zeros(K,1);
dvact_k_db_j_previous = zeros(J,K);
dvact_j_db_j_previous = zeros(J,K);
dvact_k_db_k_previous = zeros(K,1);
dvact_k_dd_k_previous = zeros(K,1);
dvact_j_dd_j_previous = zeros(J,K);
dvact_k_dd_j_previous = zeros(J,K);

vact_j=zeros(J,T);
vact_k=zeros(K,T);
u_j=zeros(J,T);
v_hat=zeros(K,T);
dvact_k_dw_jk=zeros(J,T);
dvact_k_dw_ij=zeros(I,J,T);
dvact_j_dw_ij=zeros(I,J,T);
dvact_k_db_k=zeros(K,T);
dvact_k_db_j=zeros(J,T);
dvact_j_db_j=zeros(J,T);
dvact_k_dd_k=zeros(K,T);
dvact_k_dd_j=zeros(J,T);
dvact_j_dd_j=zeros(J,T);

du_j_dw_ij=zeros(I,J,T);
du_j_db_j=zeros(J,T);
du_j_dd_j=zeros(J,T);

dv_hat_dw_jk=zeros(J,T);
dv_hat_dw_ij=zeros(I,J,T);
dv_hat_db_k=zeros(K,T);
dv_hat_db_j=zeros(J,T);
dv_hat_dd_k=zeros(K,T);
dv_hat_dd_j=zeros(J,T);

for t = delay+1:T

z_j = W_ij * vin(:,t-delay) + b_j;
vact_j(:,t) = (1-d_tau_j).*vact_j_previous + d_tau_j.*z_j;
fnonlin_j=fnonlin(vact_j(:,t),nonlin);
u_j(:,t) = fnonlin_j;

z_k = W_jk*u_j(:,t) +b_k;
vact_k(:,t) = (1-d_tau_k)*vact_k_previous + d_tau_k*z_k;
fnonlin_k=fnonlin(vact_k(:,t),nonlin);
v_hat(:,t) = fnonlin_k;

% part of the derivatives
fprime_vact_k = fprimenonlin(vact_k(:,t),nonlin);
fprime_vact_j = fprimenonlin(vact_j(:,t),nonlin);

%output weights
dvact_k_dw_jk(:,t) = (1-d_tau_k)*dvact_k_dw_jk_previous + d_tau_k*u_j(:,t);
dvact_k_dw_jk_previous = dvact_k_dw_jk(:,t);
dv_hat_dw_jk(:,t) = fprime_vact_k * dvact_k_dw_jk(:,t);

%backpropagation
%input weights
dvact_j_dw_ij(:,:,t) = bsxfun(@times,(1-d_tau_j)',dvact_j_dw_ij_previous) + ...
    vin(:,t-delay)*d_tau_j'; % dv_hat_dw_ij
dvact_j_dw_ij_previous = dvact_j_dw_ij(:,:,t);
du_j_dw_ij(:,:,t) = bsxfun(@times,fprime_vact_j', dvact_j_dw_ij(:,:,t));
dvact_k_dw_ij(:,:,t) = (1-d_tau_k)*dvact_k_dw_ij_previous + ...
    d_tau_k*bsxfun(@times,W_jk,du_j_dw_ij(:,:,t));
dvact_k_dw_ij_previous = dvact_k_dw_ij(:,:,t);
dv_hat_dw_ij(:,:,t) = fprime_vact_k * dvact_k_dw_ij(:,:,t);

%output bias
dvact_k_db_k(:,t) = (1-d_tau_k)*dvact_k_db_k_previous + d_tau_k;
dvact_k_db_k_previous = dvact_k_db_k(:,t);
dv_hat_db_k(:,t) = fprime_vact_k * dvact_k_db_k(:,t);

%input bias
dvact_j_db_j(:,t)= (1-d_tau_j).*dvact_j_db_j_previous + d_tau_j;
dvact_j_db_j_previous = dvact_j_db_j(:,t);
du_j_db_j(:,t) = fprime_vact_j.* dvact_j_db_j(:,t);
dvact_k_db_j(:,t) = (1-d_tau_k)*dvact_k_db_j_previous + ...
    d_tau_k*(W_jk'.*du_j_db_j(:,t));
dvact_k_db_j_previous = dvact_k_db_j(:,t);
dv_hat_db_j(:,t) = fprime_vact_k*dvact_k_db_j(:,t);

%time constants
primetau_dk=fprimetau(d_k,tautype);
primetau_dj=fprimetau(d_j,tautype);

%output time constants
dvact_k_dd_k(:,t) = (1-d_tau_k)*dvact_k_dd_k_previous - ...
    vact_k_previous*primetau_dk + z_k*primetau_dk;
dvact_k_dd_k_previous = dvact_k_dd_k(:,t);
dv_hat_dd_k(:,t) = fprime_vact_k*dvact_k_dd_k(:,t);

%input time constants
dvact_j_dd_j(:,t) = (1-d_tau_j).*dvact_j_dd_j_previous - ...
    vact_j_previous.*primetau_dj + z_j.*primetau_dj;
dvact_j_dd_j_previous = dvact_j_dd_j(:,t);
du_j_dd_j(:,t) = fprime_vact_j .* dvact_j_dd_j(:,t);
dvact_k_dd_j(:,t) = (1-d_tau_k)*dvact_k_dd_j_previous + ...
    d_tau_k*(W_jk'.*du_j_dd_j(:,t));
dvact_k_dd_j_previous = dvact_k_dd_j(:,t);
dv_hat_dd_j(:,t) = fprime_vact_k * dvact_k_dd_j(:,t);

vact_k_previous = vact_k(:,t);
vact_j_previous = vact_j(:,t);

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

%----------------------------------------------
% derivatives of output weights and biases
dEdW_jk = sum(bsxfun(@times,dv_hat_dw_jk,a_t),2)/T;
dEdW_jk = dEdW_jk';

dEdW_ij = zeros(I,J);
for ii = 1:T
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
    case 'abs'
dEdW_jk = dEdW_jk + lam.*sign(W_jk);
dEdW_ij = dEdW_ij + lam.*sign(W_ij);
end

ddelay = 0;

% give the gradients the same order as the parameters
dfdtheta = {dEdW_jk, dEdW_ij, dEdb_k, dEdb_j, dEdd_k, dEdd_j, ddelay};

%----------------------------------------------
%% If testing rather than training give the predictions and other
%characteristics of the network
if test
    output.f = f;
    output.funreg = funreg;
    output.vout = vout;
    output.vin = vin; 
    output.vhat = v_hat; 
    output.uout = u_j;
    output.vact.j = vact_j;
    output.vact.k = vact_k;
    f = output; 
    
end