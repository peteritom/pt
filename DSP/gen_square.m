function [ y ] = gen_square( A, fx, fs, duration, duty )

t = 0:1/fs:duration;
y = A*square(2*pi*fx*t, duty);


end

