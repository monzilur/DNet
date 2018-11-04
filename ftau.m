function [y]=ftau(d,type)

switch type
    case 'sq'
    y=1./(1+d.^2);

    case 'abs'
    y=1./(1+abs(d));
    
    case 'sig'
    y=1./(1+exp(-d));
    
    case 'log'
    y=1./log(d);
end
end