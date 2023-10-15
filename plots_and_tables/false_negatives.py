import numpy as np
import librosa
import matplotlib.pyplot as plt
from librosa.display import specshow

params = {
    "axes.labelsize": 14,
    "font.size": 14,
    "legend.fontsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "text.usetex": True,
    "font.family": "serif",
    "axes.titley": 1.02,
}
plt.rcParams.update(params)

base_rain_samples = [6, 13, 17]
path_to_wavs = "path/to/wavs"

#%%
y1, sr1 = librosa.load(f"{path_to_wavs}/rain_{base_rain_samples[0]}.wav")
melspec1 = librosa.feature.melspectrogram(y=y1, sr=sr1)
melspec_db1 = librosa.power_to_db(melspec1, ref=np.max)
y1 = y1 - np.nanmean(y1)
mfcc = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=20).reshape(-1, 1).squeeze()
zc1 = ((y1[:-1] * y1[1:]) < 0).sum() / 5
norm1 = np.linalg.norm(y1)

y2, sr2 = librosa.load(f"{path_to_wavs}/rain_{base_rain_samples[1]}.wav")
melspec2 = librosa.feature.melspectrogram(y=y2, sr=sr2)
melspec_db2 = librosa.power_to_db(melspec2, ref=np.max)
y2 = y2 - np.nanmean(y2)
mfcc = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=20).reshape(-1, 1).squeeze()
zc2 = ((y2[:-1] * y2[1:]) < 0).sum() / 5
norm2 = np.linalg.norm(y2)

y3, sr3 = librosa.load(f"{path_to_wavs}/rain_{base_rain_samples[2]}.wav")
melspec3 = librosa.feature.melspectrogram(y=y3, sr=sr3)
melspec_db3 = librosa.power_to_db(melspec3, ref=np.max)
y3 = y3 - np.nanmean(y3)
mfcc = librosa.feature.mfcc(y=y3, sr=sr3, n_mfcc=20).reshape(-1, 1).squeeze()
zc3 = ((y3[:-1] * y3[1:]) < 0).sum() / 5
norm3 = np.linalg.norm(y3)


#%%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
im1 = specshow(melspec_db1, x_axis="time", y_axis="mel", sr=sr1, fmax=2**13, ax=ax1)
cb1 = fig.colorbar(im1, ax=ax1, format="%+2.0f dB")
ax1.xaxis.set_major_formatter(librosa.display.TimeFormatter())
ax1.set_xticks(list(range(6)))
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Frequency (Hz)")
ax1.set_title(
    f"Sample {base_rain_samples[0]}: ZCR = {int(zc1)}" + r" s$^{-1}$, " + f"$l^2$-norm = {norm1:.1f}", fontsize=14
)

im2 = specshow(melspec_db2, x_axis="time", y_axis="mel", sr=sr2, fmax=2**13, ax=ax2)
cb2 = fig.colorbar(im2, ax=ax2, format="%+2.0f dB")
ax2.xaxis.set_major_formatter(librosa.display.TimeFormatter())
ax2.set_xticks(list(range(6)))
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Frequency (Hz)")
ax2.set_title(
    f"Sample {base_rain_samples[1]}: ZCR = {int(zc2)}" + r" s$^{-1}$, " + f"$l^2$-norm = {norm2:.1f}", fontsize=14
)

im3 = specshow(melspec_db3, x_axis="time", y_axis="mel", sr=sr3, fmax=2**13, ax=ax3)
cb3 = fig.colorbar(im3, ax=ax3, format="%+2.0f dB")
ax3.xaxis.set_major_formatter(librosa.display.TimeFormatter())
ax3.set_xticks(list(range(6)))
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Frequency (Hz)")
ax3.set_title(
    f"Sample {base_rain_samples[2]}: ZCR = {int(zc3)}" + r" s$^{-1}$, " + f"$l^2$-norm = {norm3:.1f}", fontsize=14
)

plt.rcParams.update(params)
fig.set_size_inches(16, 4)
fig.tight_layout(pad=1)
plt.show()
