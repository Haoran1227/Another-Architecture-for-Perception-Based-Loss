%--------------------------------------------------------------------------
% GitHubTrain_part_0_ResamplingTheNoiseFiles - Loading and resampling the
% noise files, generating the training noise and test noise
% Note that the noise is from the freesound dataset which sampling rate is 
% various. 
%
% Created by Haoran Zhao
% Friedrich-Alexander-Universit?t Erlangen-N��rnberg
% 2020 - 10 - 15

%--------------------------------------------------------------------------

% ---Settings
clear
addpath(genpath(pwd));
foldername = '.\Audio Data\Raw_noise\';
training_noise_path = '.\Audio Data\training_noise.wav';
test_noise_path = '.\Audio Data\test_noise.wav';
required_freq = 8000;
training_length = 13;   % the minutes of the training noise
test_length = 2;        % the minutes of the test noise

% --- Loop for loading the noise
s = cell(1,1);
num1 = 0;
filename = dir([foldername,'*.wav']);
for i = 1:size(filename,1)
    [sig,fs] = audioread([foldername,filename(i).name]);

    % Resampling
    if fs ~= required_freq
        sig = resample(sig,required_freq,fs);
    end

    % -- change to .raw file
    noise_file = sig(:,1).*(2^15);
    noise_int16 = int16(noise_file);

    % -- normalize to -26 dBoV
    [act_lev_noise, rms_lev_noise, gain_noise] = actlev('-sf 8000 -lev -26', noise_int16);
    noise_scaled_int16 = noise_int16 * gain_noise;
    noise_scaled = double(noise_scaled_int16);

    % -- save the processed data to different cells
    num1 = num1 + 1;
    s{num1} = noise_scaled;
end

% --- Document the length of each speech file and save to s1_speech
num_element = 0;
for nn=1:num1
    num_element = num_element + length(s{1,nn});
end

if num_element<15*60*required_freq
    num_element = 15*60*required_freq;
end

s1_noise = zeros(num_element,1);

% --- Concatenate all files to one vector
num_cal1 = 0;
for mm = 1:num1
    num_cal1 = num_cal1+length(s{1,mm});
    s1_noise(num_cal1-length(s{1,mm})+1:num_cal1,1) = s{1,mm};
end

% --- Truncate the whole noise into training_noise(include val_noise in it) and test_noise
training_samples = training_length*60*required_freq;
test_samples = test_length*60*required_freq;
audiowrite(test_noise_path,s1_noise(1:test_samples)./max(abs(s1_noise)),required_freq);
% choose the start position of the test noise randomly
start = randi([test_samples+1,num_element-training_samples+1]);
audiowrite(training_noise_path,s1_noise(start:start+training_samples-1)./max(abs(s1_noise)),required_freq);