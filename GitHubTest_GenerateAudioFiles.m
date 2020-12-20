%--------------------------------------------------------------------------
% GitHubTest_GenerateAudioFiles - Loading DNN inferenced data and   
% reconstruct to the waveform of speech signals for white- and black-box 
% measurement.
% Note that the clean speech signals are from Grid corpous (downsampled to
% 8 kHz) dataset.
%
% Given data:
%             Grid corpous (clean speech) and ChiMe-3 (noise) datasets.
%             test_s_hat                : masked noisy speech
%             test_s_tilt               : masked clean speech
%             test_n_tilt               : masked noise
%             y_phase, s_phase, n_phase : phase information
%
% Output data:
%             All speech waveforms can be choosen to be saved or not.
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
% --- Settings
% --- Set the noise levels:
% -21 for -5 dB SNR, -26 for 0 dB SNR, -31 for 5dB SNR, -36 for 10dB SNR,
% -41 for 15dB SNR, -46 for 20dB SNR
noi_lev_vec = {-21,-26,-31,-36,-41,-46};
save_files_flag = 1; % 1- Save all generated files; 0- Not save
modle_type_str_vec = {'baseline','log_power_MSE','weight_filter_AMR_direct_freqz','PESQ'}; % run all models to compare
noi_situ_model_str = '6snrs';
Fs = 8000;
speech_length = 120*Fs;  % test speech length which can be tuned. Here, 120s.

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

for noi_lev_num = 1 : length(noi_lev_vec)
    noi_lev = noi_lev_vec{noi_lev_num};
    fprintf('Working on %s case--> \n', num2str(noi_lev));
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

    y_vec_all = y_vec_all.';
    n_vec_all = n_vec_all.';
    s_vec_all = s_vec_all.';

    %% Generate s_tilde, n_tilde, and s_hat speech
    % --- Run for all modle_type_str
    for k_model_type = 1 : length(modle_type_str_vec)
        modle_type_str = modle_type_str_vec{k_model_type};
        % --- Load Python output & load phase matrix
        load(['./test results/mask_dnn_' modle_type_str '_s_hat_snr_' num2str(noi_lev) '_model_' noi_situ_model_str '_test_data.mat']);
        load(['./test results/mask_dnn_' modle_type_str '_s_tilt_snr_' num2str(noi_lev) '_model_' noi_situ_model_str '_test_data.mat']);
        load(['./test results/mask_dnn_' modle_type_str '_n_tilt_snr_' num2str(noi_lev) '_model_' noi_situ_model_str '_test_data.mat']);
        load(['./test data/test_phase_mats_snr_' num2str(noi_lev) '_model_' noi_situ_model_str '_test_data.mat']);

        % --- Generate long vectors from frames for 3 signals: s_hat, s_tilde, n_tilde
        num_fram = size(test_s_hat,1);
        s_hat_vec = zeros(1,(num_fram+1)*fram_shift);
        s_tilt_vec = zeros(1,(num_fram+1)*fram_shift);
        n_tilt_vec = zeros(1,(num_fram+1)*fram_shift);
        y_phase = y_phase.';
        s_phase = s_phase.';
        n_phase = n_phase.';
        s_hat_mat = zeros(num_fram,fram_leng);
        s_tilt_mat = zeros(num_fram,fram_leng);
        n_tilt_mat = zeros(num_fram,fram_leng);

        for k = 1 : num_fram
            fft_s_hat_half = test_s_hat(k,:);
            fft_s_hat = [fft_s_hat_half, fliplr(fft_s_hat_half(2:fram_shift))];
            fft_s_hat_cmpx = fft_s_hat .* exp(1j .* y_phase(k,:));
            s_hat_temp = real(ifft(fft_s_hat_cmpx,fram_leng));
            s_hat_mat(k,:) = s_hat_temp;

            fft_s_tilt_half = test_s_tilt(k,:);
            fft_s_tilt = [fft_s_tilt_half, fliplr(fft_s_tilt_half(2:fram_shift))];
            fft_s_tiltt_cmpx = fft_s_tilt .* exp(1j .* s_phase(k,:));
            s_tilt_temp = real(ifft(fft_s_tiltt_cmpx,fram_leng));
            s_tilt_mat(k,:) = s_tilt_temp;

            fft_n_tilt_half = test_n_tilt(k,:);
            fft_n_tilt = [fft_n_tilt_half, fliplr(fft_n_tilt_half(2:fram_shift))];
            fft_n_tiltt_cmpx = fft_n_tilt .* exp(1j .* n_phase(k,:));
            n_tilt_temp = real(ifft(fft_n_tiltt_cmpx,fram_leng));
            n_tilt_mat(k,:) = n_tilt_temp;

            % -- Form long vector with overlap-add
            if k == 1
                s_hat_vec(1:fram_shift) = s_hat_mat(1,1:fram_shift);
                s_tilt_vec(1:fram_shift) = s_tilt_mat(1,1:fram_shift);
                n_tilt_vec(1:fram_shift) = n_tilt_mat(1,1:fram_shift);
            elseif k > 1
                s_hat_nach = s_hat_mat(k-1,freq_coeff_leng:fram_leng);
                s_hat_vor  = s_hat_mat(k,1:fram_shift);
                s_hat_vec(1+(k-1)*fram_shift : k*fram_shift) = s_hat_nach + s_hat_vor;

                s_tilt_nach = s_tilt_mat(k-1,freq_coeff_leng:fram_leng);
                s_tilt_vor  = s_tilt_mat(k,1:fram_shift);
                s_tilt_vec(1+(k-1)*fram_shift : k*fram_shift) = s_tilt_nach + s_tilt_vor;

                n_tilt_nach = n_tilt_mat(k-1,freq_coeff_leng:fram_leng);
                n_tilt_vor  = n_tilt_mat(k,1:fram_shift);
                n_tilt_vec(1+(k-1)*fram_shift : k*fram_shift) = n_tilt_nach + n_tilt_vor;
            end

            % -- Display progress
             if mod(k,12000) == 0,
                disp(['Percentage of frames formed: ' num2str( (k/num_fram)* 100) '%']);
            end
        end


        % -- Form the files
        s_hat_temp = s_hat_vec;
        s_tilt_temp = s_tilt_vec;
        n_tilt_temp = n_tilt_vec;
        y_vec_temp = y_vec_all;
        s_vec_temp = s_vec_all;
        n_vec_temp = n_vec_all;

        % -- Save files or not
        if save_files_flag == 1
            save(['./generated_files/' num2str(noi_lev) '/s_hat_test_data_snr_' num2str(noi_lev) '_model_' noi_situ_model_str '_' modle_type_str '.mat'],'s_hat_temp');
            save(['./generated_files/' num2str(noi_lev) '/y_test_data_snr_' num2str(noi_lev) '.mat'],'y_vec_temp');
            save(['./generated_files/' num2str(noi_lev) '/s' '.mat'],'s_vec_temp');
            save(['./generated_files/' num2str(noi_lev) '/n_test_data_snr_' num2str(noi_lev) '.mat'],'n_vec_temp');

            audiowrite(['./generated_wavs/' num2str(-noi_lev-26) 'dB_s_hat_enhanced_model_' modle_type_str '.wav'],s_hat_temp./(2*max(abs(s_hat_temp),[],2)),Fs);
            audiowrite(['./generated_wavs/' num2str(-noi_lev-26) 'dB_y_mixture.wav'],y_vec_temp./max(abs(y_vec_temp),[],2),Fs);
        end

        % -- Possible white- and black-box measurements here ...
    end
end
