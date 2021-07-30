close all; clear; clc;
%%
Fs = 44100;
dt = 1/Fs;
duration = 1;
fx = 440;
A = 1;
t = 0:dt:duration;

signal = A*sin(2*pi*t*fx);

Am = 2;
fm = 10*fx;

mod = Am*sawtooth(2*pi*t*fm);

pwm = zeros(length(signal),1);

for i = 1:length(signal)
if (signal(i)>=mod(i))
    pwm(i)=1;
else
    pwm(i)=0;
end
end
%%

soundsc(pwm,Fs)
%%
figure
subplot(3,1,1)
plot(signal)
subplot(3,1,2)
plot(mod)
subplot(3,1,3)
plot(pwm)