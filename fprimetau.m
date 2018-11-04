function [y]=fprimetau(d,type)
switch type
    case 'sq'
        z=ftau(d,'sq');
        y=-2.*d.*(z.^2);
    case 'abs'
        z=ftau(d,'abs');
        y=-(d.*z.^2)./abs(d); 
    case 'sig'
        z=ftau(d,'sig');
        y=z.*(1.-z);
end