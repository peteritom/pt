function [ y ] = gen_sin( A, fx, fs, duration )

t = 0:1/fs:duration;
y = A*sin(2*pi*fx*t);


end

