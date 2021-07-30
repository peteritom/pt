function y = adsr( A, D, S, R, levelS, duration, Fs, model)

t = 0:1/Fs:duration;
N = length(t);
fA = floor(A*N); % A,D,S,R values are between 0..1
fD = floor(D*N);
fS = floor(S*N);
fR = floor(R*N);
Ne = fA+fD+fS+fR;
if (Ne<N)
    fS = fS + N-Ne;
end
switch model
    case 0
        a = linspace(0,1,fA);
        d = linspace(1,levelS,fD);
        s = linspace(levelS,levelS,fS);
        r = linspace(levelS,0,fR);
        y = [a';d';s';r'];
        
    case 1
        a = linspace(0,1,fA).^2;
        d = linspace(1,levelS,fD).^2;
        s = linspace(levelS,levelS,fS).^2;
        r = linspace(levelS,0,fR).^2;
        y = [a';d';s';r'];
        
    case 2
        a = 10.^(linspace(0,1,fA));
        d = 10.^(linspace(1,levelS,fD));
        s = 10.^linspace(levelS,levelS,fS);
        r = 10.^(linspace(levelS,0,fR));
        y = [a';d';s';r'];
end
y = y./max(y);
end

