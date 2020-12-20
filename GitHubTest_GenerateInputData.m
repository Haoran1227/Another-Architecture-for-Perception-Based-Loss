%--------------------------------------------------------------------------
% GitHubTest_GenerateInputData - Test data generation for DNN inference, 
% incl. MSE loss (baseline), weight filter loss (proposed), pmesq loss 
% ��proposed), for white- and black-box measurement.
% Note that the clean speech signals are from Grid corpous (downsampled to
% 8 kHz) dataset.
% speech and noise datasets are selected differently compared to training stage.
% Test signal length is set to 120s.
%
% Given data:
%             Grid corpous (clean speech) and noise datasets.
%
% Output data:
%             test_input_abs_unnorm     : auxiliary input of noisy speech
%             test_input_y              : input of noisy speech
%             test_input_s              : input of clean speech
%             test_input_n              : input of noise speech
%             y_phase, s_phase, n_phase : phase information
%
%
% Created by Ziyue Zhao
% Technische Universit?t Braunschweig
% Institute for Communications Technology (IfN)
% 2019 - 05 - 23
%
% Modified by Haoran Zhao
% Friedrich-Alexander-Universit?t Erlangen-N��rnberg
% 2020 - 10 - 15
%
% Use is permitted for any scientific purpose when citing the paper:
% Z. Zhao, S. Elshamy, and T. Fingscheidt, "A Perceptual Weighting Filter
% Loss for DNN Training in Speech Enhancement", arXiv preprint arXiv:
% 1905.09754.
%
%--------------------------------------------------------------------------

clear;
addpath(genpath(pwd));

% --- Set the noise levels:
% -21 for -5 dB SNR, -26 for 0 dB SNR, -31 for 5dB SNR, -36 for 10dB SNR,
% -41 for 15dB SNR, -46 for 20dB SNR
noi_lev_vec = {-21,-26,-31,-36,-41,-46};

for noi_lev_num = 1 : length(noi_lev_vec)
    clearvars -except noi_lev_vec noi_lev_num;
    noi_lev = noi_lev_vec{noi_lev_num};
    fprintf(' working on %sdB case --> \n', num2str(-noi_lev-26));

    % --- Settings
    noi_situ_model_str = '6snrs';
    Fs = 8000;
    speech_length = 120*Fs; % test speech length which can be tuned. Here, 120s.
    % -- Frequency domain parameters
    fram_leng = 256; % window length
    fram_shift = fram_leng/2; % frame shift
    freq_coeff_leng = fram_shift + 1; % half-plus-one frequency coefficients

    % --- Directories
    database_dir = '.\Audio Data\test_speech\';
    noi_file_name = '.\Audio Data\test_noise.wav';

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

    % --- Truncate the speech into 120 sec
    s_vec = s1_speech(1:speech_length);

    % --- Load noise files
    [noi_test_wav,~] = audioread(noi_file_name);
    noi_test_wav = noi_test_wav .* 2^15;

    % --- Trim to same length as s_vec: n_vec
    n_vec = noi_test_wav(1:speech_length);
    n_vec = int16(n_vec);

    % --- Make the noise level according to the set SNR
    noise_contr = ['-sf 8000 -lev ' num2str(noi_lev) ' -rms'];
    [~, ~, gain_noise] = actlev(noise_contr, n_vec);
    n_vec_scale = n_vec .* gain_noise;
    n_vec_scale = double(n_vec_scale);

    % --- Mix to generate noisy speech: y_vec
    y_vec_all = s_vec + n_vec_scale;

    n_vec_all = n_vec_scale;
    s_vec_all = s_vec;
    s_vec_all_leng = length(s_vec_all);
    audiowrite(['./test data/' num2str(-noi_lev-26) 'dB_test_clean.wav'], s_vec_all./max(abs(s_vec_all)), Fs)
    audiowrite(['./test data/' num2str(-noi_lev-26) 'dB_test_mixture.wav'], y_vec_all./max(abs(y_vec_all)), Fs)

    % --- Compute FFT coeffi. and phase for the 3 signals
    wd = hanning(fram_leng,'periodic');
    num_fram = floor(s_vec_all_leng/fram_shift)-1;
    for k = 1:num_fram
        % -- y 
        y_wd = y_vec_all(1+fram_shift*(k-1) : fram_leng+fram_shift*(k-1),1) .* wd;  % segment the clear speech using hanning window
        y_fft = fft(y_wd);   % FFT for the noisy speech
        y_fft_abs = abs(y_fft);          % get the amplitude spectrogram
        test_input_abs_unnorm(:,k) = y_fft_abs(1:freq_coeff_leng);
        y_phase(:,k) = angle(y_fft);

        % -- s 
        s_wd = s_vec_all(1+fram_shift*(k-1):fram_leng+fram_shift*(k-1),1).*wd;  % segment the clear speech using hanning window
        s_fft = fft(s_wd);   % FFT for the noisy speech
        s_fft_abs=abs(s_fft);          % get the amplitude spectrogram
        test_input_s(:,k)=s_fft_abs(1:freq_coeff_leng);
        s_phase(:,k) = angle(s_fft);

        % -- n 
        n_wd = n_vec_all(1+fram_shift*(k-1):fram_leng+fram_shift*(k-1),1).*wd;  % segment the clear speech using hanning window
        n_fft = fft(n_wd);   % FFT for the noisy speech
        n_fft_abs=abs(n_fft);          % get the amplitude spectrogram
        test_input_n(:,k)=n_fft_abs(1:freq_coeff_leng);
        n_phase(:,k) = angle(n_fft);

        % -- Display progress
         if mod(k,15000) == 0,
            disp(['Percentage of frames prepared: ' num2str( (k/num_fram)* 100) '%']);
        end
    end
    test_input_abs_unnorm = test_input_abs_unnorm.';
    test_input_s = test_input_s.';
    test_input_n = test_input_n.';

    % --- Normalization to get: test_input_y
    load(['.\training data\mean_training_' noi_situ_model_str '.mat']); 
    load(['.\training data\std_training_' noi_situ_model_str '.mat']); 
    for k = 1:size(test_input_abs_unnorm,1)
        test_input_y(k,:) = (test_input_abs_unnorm(k,:) - mean_training)./std_training;
        % -- Display 
         if mod(k,15000) == 0,
            disp(['Percentage of frames normalized: ' num2str( (k/num_fram)* 100) '%']);
        end
    end

    % --- Save to test data directory
    save(['./test data/test_input_y_abs_snr_' num2str(noi_lev) '_model_' noi_situ_model_str '_test_data.mat'],'test_input_y'); % y_norm   
    save(['./test data/test_input_abs_unnorm_snr_' num2str(noi_lev) '_model_' noi_situ_model_str '_test_data.mat'],'test_input_abs_unnorm'); % y
    save(['./test data/test_input_s_snr_' num2str(noi_lev) '_model_' noi_situ_model_str '_test_data.mat'],'test_input_s'); % s
    save(['./test data/test_input_n_snr_' num2str(noi_lev) '_model_' noi_situ_model_str '_test_data.mat'],'test_input_n'); % n
    % --- Save phase mat
    save(['./test data/test_phase_mats_snr_' num2str(noi_lev) '_model_' noi_situ_model_str '_test_data.mat'],'y_phase','s_phase','n_phase');
end