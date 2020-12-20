# Evaluation of perception-based loss for speech enhancement

Please find here the scripts referring to the papers [A Perceptual Weighting Filter Loss for DNN Training in Speech Enhancement](https://arxiv.org/pdf/1905.09754.pdf) and [A Deep Learning Loss Function Based on the Perceptual Evaluation of the Speech Quality](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8468124). 

In this repository we provide the source code for training/validation data preparation (including the amplitude response for the perceptual weighting filter), network training/validation (including the perceptual weighting filter loss and PESQ-based loss), network inference, enhanced speech waveform reconstruction, and measurements. The code was based on the project of perceptual weighting filter loss written by [Ziyue Zhao](https://ziyuezhao.github.io/) and the project of PMSQE written by Juan Manuel Mart´ın-Donas. Then intergrated and modified by Haoran Zhao.

## Introduction

In this project, two baseline losses and two perception-based losses are evaluated for the speech enhanement applications. The mean squared error (MSE) loss and log-power MSE loss are tested as basline. The perceptual weighting filter loss and the PESQ-based loss are evaluated and compared.

## Prerequisites

- [Matlab](https://www.mathworks.com/) 2014a or later
- [Python](https://www.python.org/) 3.6
- CPU or NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-toolkit) 9.0 [CuDNN](https://developer.nvidia.com/cudnn) 7.0.5


## Getting Started

### Installation

- Install [TensorFlow](https://www.tensorflow.org/) 1.14.0 and [Keras](https://www.tensorflow.org/) 2.3.1
- Some Python packages need to be installed, please see detailed information in the Python scripts.
- Install [Matlab](https://www.mathworks.com/)

### Datasets

Note that in this project the clean speech signals are taken from the [Grid corpus](https://doi.org/10.1121/1.2229005) (downsampled to 8 kHz) and noise signals are taken from the [ChiMe-3](https://ieeexplore.ieee.org/abstract/document/7404837/) database and [FreesoundDataset](https://annotator.freesound.org/fsd/release/FSD50K/s). In order to run the scripts in this project, the abovementioned databases are assumed to be available locally.

The training and validation speech files (downsampled to 8kHz) are placed under the directory of `./Audio Data/training_and_validation_speech/`. The test speech files (downsampled to 8kHz) are placed under the directory of `./Audio Data/test_speech/`. The noise files (No need to be resampled) are placed under the directory of `./Audio Data/Raw_noise/`. You must make sure the total length of the noise is larger than the total length of training, validation and test speech. (see `GitHubTrain_part_0_ResamplingTheNoiseFiles.m` and `GitHubTrain_part_1_CleanAndNoisyMixture.m` for the detailed directory structure of the datasets).

### Training and validation data preparation

 - Run the Matlab script to resample and generate the test noise and trianing(including validation) noise: 
```bash
matlab GitHubTrain_part_0_ResamplingTheNoiseFiles.m
```
 - Run the Matlab script to generate the frame-wise spectral amplitudes for clean and noisy speech under various SNRs: 
```bash
matlab GitHubTrain_part_1_CleanAndNoisyMixture.m
```
 - Run the Matlab script to generate the frame-wise spectral amplitudes response for the perceptual weighting filter:
```bash
matlab GitHubTrain_part_2_WghFilterResponse.m
```
 - Run the Matlab script to generate the training/validation data for the DNN model based on the output data from part 1 and 2:
```bash
matlab GitHubTrain_part_3_TrainValidDataPrepare.m
```

### Train the DNN models

 - As a baseline approach, run the Python script to train the DNN model with the **MSE loss** based on the prepared training/validation data:
```bash
python GitHub_mask_dnn_baseline_train.py
```

 - As the second baseline approach, run the Python script to train the DNN model with the **log-power MSE loss** based on the same training/validation data:
```bash
python GitHub_mask_dnn_log_power_MSE_train.py
```

 - Run the Python script to train the DNN model with the **perceptual weghting filter loss** based on the same training/validation data:
```bash
python GitHub_mask_dnn_weight_filter_train.py
```

 - Run the Python script to train the DNN model with the **PESQ-based loss** based on the same training/validation data:
```bash
python GitHub_mask_dnn_PESQ_train.py
```

### Test data preparation 

 - Run the Matlab script to generate the test input data for the inference of DNN models:
```bash
matlab GitHubTest_GenerateInputData.m
```

### Inference of the DNN models

 - As a baseline approach, run the Python script to test the trained DNN model with the **MSE loss** using the same test data:
```bash
python GitHub_all_test_mask_dnn_baseline.py
```

 - As a baseline approach, run the Python script to test the trained DNN model with the **log-power MSE loss** using the same test data:
```bash
python GitHub_all_test_mask_dnn_log_power_MSE.py
```

 - Run the Python script to test the trained DNN model with the **perceptual weighting filter loss** using the prepared test data:
```bash
python GitHub_all_test_mask_dnn_weight_filter.py
```

 - Run the Python script to test the trained DNN model with the **PESQ-based loss** using the prepared test data:
```bash
python GitHub_all_test_mask_dnn_PESQ.py
```

### Enhanced speech reconstruction

 - Run the Matlab script to reconstruct the enhanced speech signals with DNN models using the four kinds of loss functions, respectively:
```bash
matlab GitHubTest_GenerateAudioFiles.m
```

### Evaluation

 - Run the Python script to take an evaluation of the models with four loss functions, metric includes SDR, PESQ and STOI:
```bash
python Measurement.py
```

### Evaluation by Automatic Speech Recognizer (ASR)
 - Run the Python script to segment the generated enhanced speech:
```bash
python Segmentation.py
```

 - Run the Python script to implement ASR, and the word error rate will be added after the csv files (average measurement) generated by Measurement.py:
```bash
python main.py
```


