function [ y ] = transformation( x, type )

switch type
    case -1
        y = x;
    case 0
        y = x.^3;
    case 1
        y = 2*x./(1+x.^2);
    case 2
        y = 0.5*( tanh(6*x-3) + tanh(6*x+3) );
    case 3
        y = x+0.75*sin(x.^2)+0.5*x.^2;
    case 4
        y = 0.002*exp(x)+0.5*x;
end

end

