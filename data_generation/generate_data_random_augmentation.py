import numpy as np
import librosa
from sklearn.model_selection import train_test_split

np.random.seed(72)

path_to_wavs = "path/to/wavs"
path_out = "random_augmentation_data"
rain_idx = list(range(39))
no_rain_idx = list(range(2462))

rain_idx_train_eval, rain_idx_test = train_test_split(rain_idx, test_size=0.1)
rain_idx_train, rain_idx_eval = train_test_split(rain_idx_train_eval, test_size=0.2)

no_rain_idx_train_eval, no_rain_idx_test = train_test_split(no_rain_idx, test_size=0.1)
no_rain_idx_train, no_rain_idx_eval = train_test_split(no_rain_idx_train_eval, test_size=0.2)


# Rain Train
rain_data_train = np.zeros((len(no_rain_idx_train), 4322))
jj = 0
while jj < rain_data_train.shape[0]:
    for kk, ii in enumerate(rain_idx_train):

        filename = f"{path_to_wavs}/rain_{ii}.wav"
        y1, sr1 = librosa.load(filename)
        if jj >= len(rain_idx_train):
            no_rain_to_mix = np.random.choice(no_rain_idx)
            no_rain_filename = f"{path_to_wavs}/no_rain_{no_rain_to_mix}.wav"
            y2, sr2 = librosa.load(no_rain_filename)
            if len(y2) < len(y1):
                y2 = np.pad(y2, pad_width=(0, len(y1) - len(y2)))
            elif len(y1) < len(y2):
                y1 = np.pad(y1, pad_width=(0, len(y2) - len(y1)))
            weight_rain = 0.5 * np.random.rand() + 0.5
            weight_no_rain = 1 - weight_rain
            y1 = weight_rain * y1 + weight_no_rain * y2

        y1 = y1 - np.nanmean(y1)
        mfcc = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=20).reshape(-1, 1).squeeze()
        zc = ((y1[:-1] * y1[1:]) < 0).sum()
        norm = np.linalg.norm(y1)
        if jj < rain_data_train.shape[0]:
            rain_data_train[jj, 0] = zc
            rain_data_train[jj, 1] = norm
            rain_data_train[jj, 2 : len(mfcc) + 2] = mfcc

        jj += 1


np.save(f"{path_out}/rain_data_train.npy", rain_data_train)


# Rain Eval
rain_data_eval = np.zeros((len(no_rain_idx_eval), 4322))
jj = 0
while jj < rain_data_eval.shape[0]:
    for kk, ii in enumerate(rain_idx_eval):

        filename = f"{path_to_wavs}/rain_{ii}.wav"
        y1, sr1 = librosa.load(filename)
        if jj >= len(rain_idx_eval):
            no_rain_to_mix = np.random.choice(no_rain_idx)
            no_rain_filename = f"{path_to_wavs}/no_rain_{no_rain_to_mix}.wav"
            y2, sr2 = librosa.load(no_rain_filename)
            if len(y2) < len(y1):
                y2 = np.pad(y2, pad_width=(0, len(y1) - len(y2)))
            elif len(y1) < len(y2):
                y1 = np.pad(y1, pad_width=(0, len(y2) - len(y1)))
            weight_rain = 0.5 * np.random.rand() + 0.5
            weight_no_rain = 1 - weight_rain
            y1 = weight_rain * y1 + weight_no_rain * y2

        y1 = y1 - np.nanmean(y1)
        mfcc = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=20).reshape(-1, 1).squeeze()
        zc = ((y1[:-1] * y1[1:]) < 0).sum()
        norm = np.linalg.norm(y1)
        if jj < rain_data_eval.shape[0]:
            rain_data_eval[jj, 0] = zc
            rain_data_eval[jj, 1] = norm
            rain_data_eval[jj, 2 : len(mfcc) + 2] = mfcc
        jj += 1

np.save(f"{path_out}/rain_data_eval.npy", rain_data_eval)


# Rain Test
rain_data_test = np.zeros((len(no_rain_idx_test), 4322))
jj = 0
while jj < rain_data_test.shape[0]:
    for kk, ii in enumerate(rain_idx_test):

        filename = f"{path_to_wavs}/rain_{ii}.wav"
        y1, sr1 = librosa.load(filename)
        if jj >= len(rain_idx_test):
            no_rain_to_mix = np.random.choice(no_rain_idx)
            no_rain_filename = f"{path_to_wavs}/no_rain_{no_rain_to_mix}.wav"
            y2, sr2 = librosa.load(no_rain_filename)
            if len(y2) < len(y1):
                y2 = np.pad(y2, pad_width=(0, len(y1) - len(y2)))
            elif len(y1) < len(y2):
                y1 = np.pad(y1, pad_width=(0, len(y2) - len(y1)))
            weight_rain = 0.5 * np.random.rand() + 0.5
            weight_no_rain = 1 - weight_rain
            y1 = weight_rain * y1 + weight_no_rain * y2

        y1 = y1 - np.nanmean(y1)
        mfcc = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=20).reshape(-1, 1).squeeze()
        zc = ((y1[:-1] * y1[1:]) < 0).sum()
        norm = np.linalg.norm(y1)
        if jj < rain_data_test.shape[0]:
            rain_data_test[jj, 0] = zc
            rain_data_test[jj, 1] = norm
            rain_data_test[jj, 2 : len(mfcc) + 2] = mfcc
        jj += 1

np.save(f"{path_out}/rain_data_test.npy", rain_data_test)


no_rain_data_train = np.zeros((len(no_rain_idx_train), 4322))
for kk, ii in enumerate(no_rain_idx_train):
    filename = f"{path_to_wavs}/no_rain_{ii}.wav"
    y1, sr1 = librosa.load(filename)

    y1 = y1 - np.nanmean(y1)
    mfcc = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=20).reshape(-1, 1).squeeze()
    zc = ((y1[:-1] * y1[1:]) < 0).sum()
    norm = np.linalg.norm(y1)
    no_rain_data_train[kk, 0] = zc
    no_rain_data_train[kk, 1] = norm
    no_rain_data_train[kk, 2 : len(mfcc) + 2] = mfcc

np.save(f"{path_out}/no_rain_data_train.npy", no_rain_data_train)

no_rain_data_eval = np.zeros((len(no_rain_idx_eval), 4322))
for kk, ii in enumerate(no_rain_idx_eval):
    filename = f"{path_to_wavs}/no_rain_{ii}.wav"
    y1, sr1 = librosa.load(filename)

    y1 = y1 - np.nanmean(y1)
    mfcc = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=20).reshape(-1, 1).squeeze()
    zc = ((y1[:-1] * y1[1:]) < 0).sum()
    norm = np.linalg.norm(y1)
    no_rain_data_eval[kk, 0] = zc
    no_rain_data_eval[kk, 1] = norm
    no_rain_data_eval[kk, 2 : len(mfcc) + 2] = mfcc

np.save(f"{path_out}/no_rain_data_eval.npy", no_rain_data_eval)

no_rain_data_test = np.zeros((len(no_rain_idx_test), 4322))
for kk, ii in enumerate(no_rain_idx_test):
    filename = f"{path_to_wavs}/no_rain_{ii}.wav"
    y1, sr1 = librosa.load(filename)

    y1 = y1 - np.nanmean(y1)
    mfcc = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=20).reshape(-1, 1).squeeze()
    zc = ((y1[:-1] * y1[1:]) < 0).sum()
    norm = np.linalg.norm(y1)
    no_rain_data_test[kk, 0] = zc
    no_rain_data_test[kk, 1] = norm
    no_rain_data_test[kk, 2 : len(mfcc) + 2] = mfcc

np.save(f"{path_out}/no_rain_data_test.npy", no_rain_data_test)
