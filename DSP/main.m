clear; close all; clc;
%%

isSound = false;
isPlot = true;
isADSR = true;
isSoundSave = true;
Sound_save_filename = 'export_sounds\EXTRA_subtr_chord_sin_k100_tr5.wav';
isPlotSave = true;
Plot_save_filename = 'export_sounds\EXTRA_subtr_chord_sin_k100_tr5.jpeg';
%% reading the .wav file, plotting it, playing the sound

fileName1 = 'prodigy.wav';
fileName2 = 'flute1.wav';
fileName3 = 'flute2.wav';
fileName4 = 'piano1.wav';
fileName5 = 'marimba1.wav';
fileName6 = 'violin1.wav';
fileName7 = 'tria1.wav';
fileName8 = 'tamas1.wav';
fileName9 = 'benny.wav';
fileName10 = 'benny_bass.wav';
[y, Fs] = audioread(fileName8);
info = audioinfo(fileName8);
duration = info.Duration;
N = length(y);
mode = 0; 
% 0: sin
% 1: square
% 2: sawtooth
duty = 50;
noise = 0.0001;
k = 100;
trType = 4;
%     case -1
%         y = x;
%     case 0
%         y = x.^3;
%     case 1
%         y = 2*x./(1+x.^2);
%     case 2
%         y = 0.5*( tanh(6*x-3) + tanh(6*x+3) );
%     case 3
%         y = x+0.75*sin(x.^2)+0.5*x.^2;
%     case 4
%         y = 0.002*exp(x)+0.5*x;
%% adsr( A, D, S, R, levelS, duration, Fs)

% marimba, piano
% A=0.03; 
% D=0.5; 
% R=0.1;
% S = 1-(A+D+R);
% lS = 0.3;
% model_adsr = 0;

% flute
A=0.15; 
D=0.1; 
R=0.1;
S = 1-(A+D+R);
lS = 0.8;
model_adsr = 2;
if isADSR
adsr_ = adsr( A, D, S, R, lS, duration, Fs, model_adsr);
adsr_ = adsr_(1:N);
else
adsr_ = ones(N,1);
end
%%
signal = y(:,1); % the first ch is the signal

%% collecting the k most relevant spectral component
xdft = 1/N * abs(fft(signal));
xdft = xdft(1:floor(N/2+1)); % ketoldalsavos spektrum miatt
freq = (0:Fs/N:Fs/2)';
[sortedValues,sortIndex] = sort(xdft,'descend');
kAmpl = sortedValues(1:k);
kFreq = freq(sortIndex(1:k));
%% generating and playing a signal with the k sine and k amplitude

yHat = gen_sound( kAmpl, kFreq, Fs, duration, mode, duty, noise, trType)';
yHat = adsr_ .* yHat(1:N);

% gain = 5;
% delay = 10;
% wet = 0.8;
% yHat = reverb(yHat, gain, delay, wet);

% cutofflow = 0.5;
% cutoffhigh = 0.1;
% bp1 = 0.1;
% bp2 = 0.11;
% low = 1;
% high = 0;
% band = 1-low-high; 
% 
% yHat = subtract_synth(yHat,cutofflow,cutoffhigh, bp1, bp2, low, high, band);
% ratio = max(signal)/max(yHat);
% yHat = ratio*yHat;
%% error signal
% errorSignal = signal-yHat;
% if isSound
% soundsc(errorSignal,Fs)
% pause
% end
if isSound
soundsc(signal,Fs)
pause
end
if isSound
soundsc(yHat,Fs)
end
if isPlot
h = figure;
subplot(2,1,2)
plot(psd(spectrum.periodogram,yHat,'Fs',Fs,'NFFT',length(yHat)));
title('Reconstructed signal','FontSize',14)
xlhand = get(gca,'xlabel');
set(xlhand,'fontsize',12)
xlhand = get(gca,'ylabel');
set(xlhand,'fontsize',12)
set(gca,'FontSize',12)
subplot(2,1,1)
plot(psd(spectrum.periodogram,signal,'Fs',Fs,'NFFT',length(signal)));
title('Original signal','FontSize',14)
xlhand = get(gca,'xlabel');
set(xlhand,'fontsize',12)
xlhand = get(gca,'ylabel');
set(xlhand,'fontsize',12)
set(gca,'FontSize',12)
if isPlotSave
    saveas(h,Plot_save_filename)
end
end
% if isPlot
%     figure
%     plot(adsr)
%     title('ADSR')
% end
% if isPlot
% figure
% plot(errorSignal)
% title('Error signal')
% end


if isSoundSave
    audiowrite(Sound_save_filename,yHat,Fs)
end