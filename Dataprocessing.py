def cal_de_feature(signal):
    kde = gaussian_kde(signal)
    x = np.linspace(min(signal), max(signal), 1000)
    pdf = kde(x)
    differential_entropy = -simpson(y=pdf * np.log(pdf), x=x)
    return differential_entropy

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    import numpy as np
import torch
import os
import pickle
from scipy.stats import gaussian_kde
from scipy.integrate import simpson  # 使用 simpson 替代 simps
file_path = './DREAMER_npz/DREAMER_1_1.npz'
save_path = './DREAMER/DREAMER_DE_LDS_train'
isExists = os.path.exists(save_path)
if isExists:
    pass
else:
    os.makedirs(save_path)

import os
data_dir = './DREAMER_npz/'
files = os.listdir(data_dir)
window = np.hanning(128)
delta_band = (1, 4)
theta_band = (4, 8)
alpha_band = (8, 15)
beta_band = (15, 30)
gamma_band = (30, 50)
freqs = np.fft.fftfreq(128, d=1.0 / 128.0)
delta_mask = np.logical_and(np.abs(freqs) >= delta_band[0], np.abs(freqs) <= delta_band[1])
theta_mask = np.logical_and(np.abs(freqs) >= theta_band[0], np.abs(freqs) <= theta_band[1])
alpha_mask = np.logical_and(np.abs(freqs) >= alpha_band[0], np.abs(freqs) <= alpha_band[1])
beta_mask  = np.logical_and(np.abs(freqs) >= beta_band[0] , np.abs(freqs) <=  beta_band[1])
gamma_mask = np.logical_and(np.abs(freqs) >= gamma_band[0], np.abs(freqs) <= gamma_band[1])
for j in range(20,23):
    random_target = torch.randperm(18)[0:5]
    train_data, train_Valence, train_Arousal, train_Dominance = [], [], [], []
    test_data, test_Valence, test_Arousal, test_Dominance = [], [], [], []
    filename = files[j*18]
    number = filename[8:10] if len(filename) == 16 else filename[8:9]
    # print(number)
    for i in range(18):
        file_path = data_dir + files[j*18 + i]
        npz_data = np.load(file_path, allow_pickle=True)
        # print(file_path)
        de_feature = []
        for k in range(14):
            baseline = npz_data['baseline'][0][:,k]
            mean_base = np.mean(baseline)
            stimuli = npz_data['stimuli'][0][:,k] - mean_base
            de_feature_channel = []
            de_feature_channel_smoothed = []
            for n in range(len(stimuli) // 128):
                data  = stimuli[n * 128: (n + 1) * 128]
                windowed_signal = data * window
                freq_signal = np.fft.fft(windowed_signal)
                
                freq_delta_signal = freq_signal * delta_mask
                freq_theta_signal = freq_signal * theta_mask
                freq_alpha_signal = freq_signal * alpha_mask
                freq_beta_signal  = freq_signal *  beta_mask
                freq_gamma_signal = freq_signal * gamma_mask

                time_delta_signal = np.real(np.fft.ifft(freq_delta_signal))
                time_theta_signal = np.real(np.fft.ifft(freq_theta_signal))
                time_alpha_signal = np.real(np.fft.ifft(freq_alpha_signal))
                time_beta_signal  = np.real(np.fft.ifft( freq_beta_signal))
                time_gamma_signal = np.real(np.fft.ifft(freq_gamma_signal))

                delta_de = cal_de_feature(time_delta_signal)
                theta_de = cal_de_feature(time_theta_signal)
                alpha_de = cal_de_feature(time_alpha_signal)
                beta_de  = cal_de_feature( time_beta_signal)
                gamma_de = cal_de_feature(time_gamma_signal)

                de_feature_channel.append([delta_de, theta_de, alpha_de, beta_de, gamma_de])
            check_1 = np.array(de_feature_channel)
            # print(check_1.shape)
            for c in range(5):
                smoothed_entropies = moving_average(check_1[:,c], window_size=5)
                de_feature_channel_smoothed.append(smoothed_entropies)
            de_feature.append(de_feature_channel_smoothed)
            print('channel_' + str(k))
        check_2 = np.array(de_feature)
        de_feature_re = []
        for m in range(check_2.shape[2]):
            c = []
            for n in range(check_2.shape[0]):
                b = []
                for k in range(check_2.shape[1]):
                    b.append(check_2[n, k, m])
                c.append(b)
            de_feature_re.append(c)
        de_feature_re = np.array(de_feature_re)
        scoreValence = [npz_data['scoreValence'][0]] * len(de_feature_re)
        scoreArousal = [npz_data['scoreArousal'][0]] * len(de_feature_re)
        scoreDominance = [npz_data['scoreDominance'][0]] * len(de_feature_re)
        if i in random_target:
            if len(test_data) == 0:
                test_data = de_feature_re
                test_Valence = scoreValence
                test_Arousal = scoreArousal
                test_Dominance = scoreDominance
            else:
                test_data = np.vstack((test_data,de_feature_re))
                test_Valence = np.hstack((test_Valence,scoreValence))
                test_Arousal = np.hstack((test_Arousal,scoreArousal))
                test_Dominance = np.hstack((test_Dominance,scoreDominance))
        else:
            if len(train_data) == 0:
                train_data = de_feature_re
                train_Valence = scoreValence
                train_Arousal = scoreArousal
                train_Dominance = scoreDominance
            else:
                train_data = np.vstack((train_data,de_feature_re))
                train_Valence = np.hstack((train_Valence,scoreValence))
                train_Arousal = np.hstack((train_Arousal,scoreArousal))
                train_Dominance = np.hstack((train_Dominance,scoreDominance))
        print('subject' + str(j) + '-----step' + str(i) + ' finished')
    save_dir = save_path + '/' + number +'.npz'
    
    np.savez(save_dir, train_data = train_data, test_data = test_data, train_Valence = train_Valence,
             train_Arousal = train_Arousal, train_Dominance = train_Dominance, test_Valence = test_Valence,
             test_Arousal = test_Arousal, test_Domiance = test_Dominance)
    print('file_saved successfully '+ save_dir)