function [p] = fprimenonlin(x,fun)
% [p] = fprimenonlin(x)
% First derivative of sigmoid function expressed as the sigmoid function
% itself
% func option:
% sigmoid: s(x) = 1 / (1+exp(-x))
% exp: exp(x)

switch fun
    case 'sigmoid'
        p=(1./(1+exp(-x))).*(1.-(1./(1+exp(-x))));
    case 'exp'
        p = exp(x);
end

end

