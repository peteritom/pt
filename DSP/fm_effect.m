function [ y ] = fm_effect( A, fx, fs, duration, fm, Am )

t = 0:1/fs:duration;
y = A*(1+square(2.5*pi*fx*t)).*square(2*pi*fx*t+Am.*square(2*pi*fm*t));

end

