function [ y ] = gen_sound( amps, freqs, fs, duration, type, duty, noise, trType )

N_s = length(0:1/fs:duration);
N = length(amps);
A = zeros(N_s,N);
switch type
    case 0
        for i = 1:N
            A(:,i) = gen_sin( amps(i), freqs(i), fs, duration)';
            A(:,i) = transformation(A(:,i),trType);
        end
    case 1
        for i = 1:N
            A(:,i) = gen_square( amps(i), freqs(i), fs, duration, duty)';
            A(:,i) = transformation(A(:,i),trType);
        end
    case 2
        for i = 1:N
            A(:,i) = gen_sawtooth( amps(i), freqs(i), fs, duration, duty)';
            A(:,i) = transformation(A(:,i),trType);
        end
end


y = sum(A,2)';
y = y + randn(size(y)) * noise;

end

