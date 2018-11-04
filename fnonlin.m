function [p] = fnonlin(x,fun)
% [p] = fnonlin(x)
% func option:
% sigmoid: s(x) = 1 / (1+exp(-x))
% exp: exp(x)

switch fun
    case 'sigmoid'
        p = 1./(1+exp(-x));
    case 'exp'
        p = exp(x);
end
        
end

