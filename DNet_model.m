function v_hat = DNet_model(X_fht,theta,tautype)
% v_hat = DNet_model(X_fht,theta)
% X_fht tensor of stimuli data
% theta is a cell array of network parameters

% get the variables (weights and biases)
W_jk = theta{1};
W_ij = theta{2};
b_k = theta{3};
b_j = theta{4};
d_k = theta{5};
d_tau_k = ftau(d_k,tautype);
d_j= theta{6};
d_tau_j = ftau(d_j,tautype);
delay = floor(theta{7});

I = size(W_ij,2);
   
% get the data
T = size(X_fht,3);
vin = reshape(X_fht,I,T);

u_previous = zeros(size(b_j));
v_previous = zeros(size(b_k));

for t = 1+delay:T
    z_j = W_ij * vin(:,t-delay) + b_j;
    uact(:,t) = z_j;
    fsig_zj=fsigmoid(z_j);
    u_j(:,t) = (1-d_tau_j).*u_previous + d_tau_j.*fsig_zj;

    z_k = W_jk*u_j(:,t) + b_k;
    vact(:,t) = z_k;
    fsig_zk=fsigmoid(z_k);
    v_hat(:,t) = (1-d_tau_k).*v_previous + d_tau_k.*fsig_zk;
end
end
