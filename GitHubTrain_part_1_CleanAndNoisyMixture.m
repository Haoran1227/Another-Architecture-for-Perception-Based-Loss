%--------------------------------------------------------------------------
% GitHubTrain_part_1_CleanAndNoisyMixture - Loading clean speech and
% noise, generating mixture signal (noisy) with 6 SNRs, generating
% frame-wise frequency amplitudes for clean and noisy speech. 
% Note that the clean speech signals are from Grid corpous (downsampled to
% 16 kHz) dataset which should be downsampled to 8khz.
%
% Given data:
%             Grid corpous (clean speech) and noise datasets.
%
% Output data:
%             s_speech             : whole clean speech signal
%                                    (for part 2 usage)
%             speech_fft_abs_clean : frequency amplitudes for clean speech
%                                    (for part 3 usage)
%             mixture_fft_abs      : frequency amplitudes for noisy speech
%                                    (for part 3 usage)
%
% Created by Ziyue Zhao
% Technische Universit?t Braunschweig
% Institute for Communications Technology (IfN)
% 2019 - 05 - 23
%
% Modified by Haoran Zhao
% Friedrich-Alexander-Universit?t Erlangen-N��rnberg
% 2020 - 10 - 15

% Use is permitted for any scientific purpose when citing the paper:
% Z. Zhao, S. Elshamy, and T. Fingscheidt, "A Perceptual Weighting Filter
% Loss for DNN Training in Speech Enhancement", arXiv preprint arXiv:
% 1905.09754.

%--------------------------------------------------------------------------
clear;
addpath(genpath(pwd));
% --- Settings
num_snr_mix = 6; % Number of the mixed SNRs
Fs = 8000;

% This term determines how long the total training and validation speech is.
% Here 11min training speech and 2min validation speech. Total 13min
speech_length = 13*60*Fs;

% -- Set the noise levels:
% -21 for -5 dB SNR, -26 for 0 dB SNR, -31 for 5dB SNR, -36 for 10dB SNR,
% -41 for 15dB SNR, -46 for 20dB SNR
noi_lev_vec = -21:-5:-46;
% -- Frequency domain parameters
fram_leng = 256; % window length
fram_shift = fram_leng/2; % frame shift
freq_coeff_leng = fram_shift + 1; % half-plus-one frequency coefficients

% --- Input directories
database_dir = '.\Audio Data\training_and_validation_speech\';
noise_dir = '.\Audio Data\training_noise.wav';

% --- Output directories
train_sspeech_dir = '.\train\speech_clean_s_speech.mat';
train_clean_dir = '.\train\speech_fft_abs_clean_6snrs.mat';
train_mixture_dir = '.\train\mixture_fft_abs_6snrs.mat';

%% Read clean speech and produce frequency amplitudes
% --- Loop for loading clean speech
s1 = cell(1,1);
num1 = 0;
database_file = dir([database_dir '\*.wav']);
for i = 1:size(database_file,1)
    in_file = [database_dir database_file(i).name];
    fprintf('  %s --> \n', in_file);

    % -- read as .raw file
    [speech_file_wav,fs] = audioread(in_file);
    if fs ~= Fs    % Resampling
        speech_file_wav = resample(speech_file_wav,Fs,fs);
    end
    speech_file = speech_file_wav(:,1).*(2^15);
    speech_int16 = int16(speech_file);

    % -- normalize to -26 dBoV
    [act_lev_speech, rms_lev_speech, gain_speech] = actlev('-sf 8000 -lev -26', speech_int16);
    speech_scaled_int16 = speech_int16 * gain_speech;
    speech_scaled = double(speech_scaled_int16);

    % -- save the processed data to different cells
    num1 = num1+1;
    s1{num1} = speech_scaled;
end

% --- Document the length of each speech file and save to s1_speech
num_element1 = 0;
for nn=1:num1
    num_element1 = num_element1 + length(s1{1,nn});
end
s1_speech = zeros(num_element1,1);

% --- Concatenate all files to one vector
num_cal1 = 0;
for mm = 1:num1
    num_cal1 = num_cal1+length(s1{1,mm});
    s1_speech(num_cal1-length(s1{1,mm})+1:num_cal1,1) = s1{1,mm};
end

% --- Truncate the speech into speech_length
s1_speech = s1_speech(1:speech_length);
audiowrite('./train/clean_speech.wav',s1_speech./2^15,Fs)
% --- Copy 6 times for 6 SNRs
s_speech=[s1_speech;s1_speech;s1_speech;s1_speech;s1_speech;s1_speech];

% --- frame-wise FFT processing
wd = hanning(fram_leng,'periodic');
num_frame = (floor(length(s1_speech)*num_snr_mix/fram_shift)-1);
speech_fft_abs_clean = zeros(freq_coeff_leng,num_frame);
clear s1 speech_file_wav speech_file speech_file speech_scaled_int16 speech_int16 speech_scaled
for jj=1:num_frame
    % -- Get frequency amplitude
    speech_wd = s_speech(1+fram_shift*(jj-1):fram_leng+fram_shift*(jj-1),1).*wd;
    speech_fft = fft(speech_wd); % FFT for the clear speech
    fft_abs = abs(speech_fft); % get the amplitude spectrogram
    speech_fft_abs_clean(:,jj) = fft_abs(1:freq_coeff_leng);
    % -- Display progress
    if mod(jj,10000) == 0,
        disp(['Percentage of frames finished (FFT): ' num2str( (jj/num_frame)* 100) '%']);
    end
end

% --- Save the clean speech frequency amplitude (129 coeff. from 256 FFT points)
save(train_clean_dir,'speech_fft_abs_clean','-v7.3')
save(train_sspeech_dir,'s_speech','-v7.3');
clear s_speech

%% Read noise and produce frequency amplitudes for mixture
% --- read noise
[noise_wav,~]=audioread(noise_dir);
noise_raw=noise_wav.*(2^15); % transfer to raw file

% --- Trim the noise to the same length with the speech
noise_raw = noise_raw(1:speech_length,1);
noise_int16 = int16(noise_raw);

% --- Adjust the noise level according to the set SNR
noise = cell(1,1);
num_n = 0;
for act_n = noi_lev_vec
    num_n = num_n+1;
    noise_contr = ['-sf 8000 -lev ' num2str(act_n) ' -rms'];
    [~, ~, gain_noise] = actlev(noise_contr, noise_int16);
    noise_int16_scale = noise_int16.*gain_noise;
    noise_scale = double(noise_int16_scale);
    noise{num_n} = noise_scale;
end
clear noise_raw noise_int16 speech_scaled noise_int16_scale

% --- mix the speech with SNRs
mixed_speech_cell = cell(1,1);
l_mix = min(num_element1,length(noise_scale));% minimum length of s1_speech and noise_scale
for cc = 1:num_n
    mixed_speech_raw = noise{cc}(1:l_mix,1)+s1_speech(1:l_mix,1);
    mixed_speech_cell{cc} = mixed_speech_raw;
end
clear s1_speech noise mixed_speech_raw noise_scale

% --- Save to one matrix: mixed_speech
num_element2 = 0;
for nn = 1:num_n
    num_element2=num_element2+length(mixed_speech_cell{1,nn});
end
mixed_speech=zeros(num_element2,1);

num_cal2 = 0;
for mm = 1:num_n
    num_cal2 = num_cal2+length(mixed_speech_cell{1,mm});
    mixed_speech(num_cal2-length(mixed_speech_cell{1,mm})+1:num_cal2,1) = mixed_speech_cell{1,mm};
end
l_mix=num_element2;
clear mixed_speech_cell 
audiowrite('./train/mixture_signal.wav',mixed_speech./max(abs(mixed_speech)),Fs)
%--- FFT processing
wd = hanning(fram_leng,'periodic');
l_process = floor(l_mix/fram_shift)-1;
mixture_fft_abs = zeros(freq_coeff_leng,l_process);
for jj = 1:l_process
    speech_wd = mixed_speech(1+fram_shift*(jj-1):fram_leng+fram_shift*(jj-1),1).*wd;  %segment the clear speech using hanning window
    speech_fft = fft(speech_wd); % FFT for the noisy speech
    fft_abs = abs(speech_fft); % get the amplitude spectrogram
    mixture_fft_abs(:,jj) = fft_abs(1:freq_coeff_leng);
    % -- Display progress
    if mod(jj,10000) == 0,
        disp(['Percentage of frames finished: ' num2str( (jj/l_process)* 100) '%']);
    end
end

% --- Save mixture
save(train_mixture_dir,'mixture_fft_abs','-v7.3')






