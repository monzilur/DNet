function effective_HU = get_STRF_from_DNet(theta,dt,nf,nh,tau_type)
% effective_HU = get_STRF_from_DNet(theta,dt,nf,nh,tau_type)
% effective_HU is a structure that contains weights, tau and ie_score of
% the STRFs estimated by DNet
% Mandatory arguments: theta, dt
% Optional arguments: nf, nh, tau_type
% In the abscence of nf and nh - weight matrix will be returned as a vector
% In the abscence of tau_type, the default is set to 'sq'
%
% Author: Monzilur Rahman
% Email: monzilur.rahman@gmail.com
% http:www.monzilurrahman.com
% Year: 2019

if ~exist('tau_type','var')
    tau_type = 'sq';
end

OU_weights = theta{1};
effective_HU_ind = find(abs(OU_weights) >= sum(abs(OU_weights))*0.05);

[~,sorted_II] = sort(abs(OU_weights(effective_HU_ind)),'descend');

for ii=1:length(sorted_II)
    ind = effective_HU_ind(sorted_II(ii));
    W_unit = theta{2}(ind,:);
    
    if and(exist('nf','var'),exist('nh','var'))
        W_unit = reshape(W_unit,nf,nh);
    end
    sign_unit = sign(OU_weights(ind));
    
    effective_HU(ii).output_weight = OU_weights(ind);
    effective_HU(ii).STRF_weights = W_unit*sign_unit;
    effective_HU(ii).STRF_tau = dt *(1./ftau(theta{5},tau_type) + ...
        1./ftau(theta{6}(ind),tau_type));
    effective_HU(ii).STRF_IE_score = sign_unit*sum(W_unit(:))/sum(abs(W_unit(:)));
end

end