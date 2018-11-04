load('test_data_5ms');
train_nfht = X_nfht(1:floor(size(X_nfht,1)*0.8),:,:,:);
train_nt = y_nt(1:floor(size(X_nfht,1)*0.8),:);

test_nfht = X_nfht(floor(size(X_nfht,1)*0.8)+1:end,:,:,:);
test_nt = y_nt(floor(size(X_nfht,1)*0.8)+1:end,:);

%% train a model based on test data
% [theta,train_err] = fit_DNet_model(X_fht,y_t,tautype,...
%    regtype,lam,net_str,nonlin,num_pass,model,theta_init)
delay = 1;
J = 10; % number of hidden units
K = 1; % number of output units
C = 0.5;
I = size(X_nfht,2)*size(X_nfht,3);
W_jk = C*2*(rand(K,J)-0.5)/sqrt(J+K);
W_ij = C*2*(rand(J,I)-0.5)/sqrt(I+J);
b_k = C*2*(rand(K,1)-0.5)/sqrt(J+K);
b_j = C*2*(rand(J,1)-0.5)/sqrt(I+J);
d_k = exprnd(2,K,1);
d_j = exprnd(2,J,1);
theta_init = {W_jk,W_ij,b_k,b_j,d_k,d_j,delay};
[theta,train_err]=fit_DNet_model(train_nfht,train_nt,'sq','abs',1e-5,...
    {J K},'sigmoid',40,'mDNet',theta_init);
% for crossvalidation use v2
% [theta,train_err,optim_lam,MSE]=fit_DNet_model_v2(train_nft,...
%     train_ynt,30,'abs');

%% now use the model to make prediction
for i = 1:size(test_nfht,1)
    v_hat(i,:) = DNet_model(shiftdim(test_nfht(i,:,:,:),1),theta,'sq');
end

%% plot the results
n_column = 5;
n_row = ceil(J/n_column)+1;

% error function
subplot(n_row,2,1)
loglog(train_err)
xlabel('iteration')
ylabel('err')

% data and prediction
subplot(n_row,2,2)
hold on
plot(test_nt(1,:),'b');
plot(v_hat(1,:),'r');
legend('data','prediction');

% hidden units sorted by variance of their weight matrix
[~,ind] = sort(var(theta{2},[],2),'descend');
for ii=1:length(ind)
  II=ind(ii);
  subplot(n_row,n_column,n_column+ii)
  weights = reshape(theta{2}(II,:),size(X_nfht,2),size(X_nfht,3));
  weights = weights * sign(theta{1}(II));
  maxabs = max(abs(weights(:)));
  imagesc(weights,[-maxabs maxabs]);
  axis xy;
end
