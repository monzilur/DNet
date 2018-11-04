load('test_data');
train_nfht = X_nfht(1:floor(size(X_nfht,1)*0.8),:,:,:);
train_nt = y_nt(1:floor(size(X_nfht,1)*0.8),:);

test_nfht = X_nfht(floor(size(X_nfht,1)*0.8)+1:end,:,:,:);
test_nt = y_nt(floor(size(X_nfht,1)*0.8)+1:end,:);

%% train a model based on test data
% [theta,train_err] = fit_DNet_model(X_fht,y_t,tautype,...
%    regtype,lam,net_str,nonlin,num_pass,model,theta_init)
n_hidden = 10;
[theta,train_err]=fit_DNet_model(train_nfht,train_nt,'sq','abs',1e-5,...
    {n_hidden 1},'sigmoid',20,'mDNet');
% for crossvalidation use v2
% [theta,train_err,optim_lam,MSE]=fit_DNet_model_v2(train_nft,...
%     train_ynt,30,'abs');

%% now use the model to make prediction
for i = 1:size(test_nfht,1)
    v_hat(i,:) = DNet_model(shiftdim(test_nfht(i,:,:,:),1),theta,'sq');
end

%% plot the results
n_column = 5;
n_row = ceil(n_hidden/n_column)+1;

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
[~,ind] = sort(var(theta{2}(1:end-1,:),[],2),'descend');
for ii=1:length(ind)
  II=ind(ii);
  subplot(n_row,n_column,n_column+ii)
  weights = reshape(theta{2}(II,:),size(X_nfht,2),size(X_nfht,3));
  weights = weights * sign(theta{1}(II));
  maxabs = max(abs(weights(:)));
  imagesc(weights,[-maxabs maxabs]);
  axis xy;
end
