function [ y ] = subtract_synth( x , cutofflow, cutoffhigh, bp1, bp2, low, high, band)

[B1, A1] = butter(7,cutofflow, 'low');
[B2, A2] = butter(7,cutoffhigh, 'high');
[B3, A3] = butter(7,[bp1 bp2], 'bandpass');

y1 = filter(B1,A1,x);
y2 = filter(B2,A2,x);
y3 = filter(B3,A3,x);
y = 2*low*y1+2*high*y2+2*band*y3;
end

