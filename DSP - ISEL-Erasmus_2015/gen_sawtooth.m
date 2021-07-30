function [ y ] = gen_sawtooth( A, fx, fs, duration, width )

t = 0:1/fs:duration;
y = A*sawtooth(2*pi*fx*t, width);


end