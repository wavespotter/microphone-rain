import matplotlib.pyplot as plt
import numpy as np
import librosa
from librosa.display import specshow

params = {
    "axes.labelsize": 20,
    "font.size": 20,
    "legend.fontsize": 16,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "text.usetex": True,
    "font.family": "serif",
}
plt.rcParams.update(params)

path_to_wavs = "path/to/wavs"
#%%
fs = 48e3
y1, sr1 = librosa.load(f"{path_to_wavs}/rain_2.wav")
mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=20)
melspec1 = librosa.feature.melspectrogram(y=y1, sr=sr1)
melspec_db1 = librosa.power_to_db(melspec1, ref=np.max)

y2, sr2 = librosa.load(f"{path_to_wavs}/no_rain_0.wav")
mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=20)
melspec2 = librosa.feature.melspectrogram(y=y2, sr=sr2)
melspec_db2 = librosa.power_to_db(melspec2, ref=np.max)

#%%
fig, ((ax3, ax4), (ax1, ax2)) = plt.subplots(nrows=2, ncols=2)

im1 = specshow(melspec_db1, x_axis="time", y_axis="mel", sr=sr1, fmax=2**13, ax=ax1)
cb1 = fig.colorbar(im1, ax=ax1, format="%+2.0f dB")
ax1.xaxis.set_major_formatter(librosa.display.TimeFormatter())
ax1.set_xticks(list(range(6)))
ax1.set_title("(c) Rain Mel Spectrogram")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Frequency (Hz)")

im2 = specshow(melspec_db2, x_axis="time", y_axis="mel", sr=sr2, fmax=2**13, ax=ax2)
cb2 = fig.colorbar(im2, ax=ax2, format="%+2.0f dB")
ax2.xaxis.set_major_formatter(librosa.display.TimeFormatter())
ax2.set_xticks(list(range(6)))
ax2.set_title("(d) No-Rain Mel Spectrogram")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Frequency (Hz)")

im3 = specshow(mfcc1, x_axis="time", ax=ax3, sr=sr1, vmin=-700, vmax=200)
cb3 = fig.colorbar(im3, ax=ax3)
ax3.xaxis.set_major_formatter(librosa.display.TimeFormatter())
ax3.set_xlabel("")
ax3.set_ylabel("MFCC Index")
ax3.set_title("(a) Rain MFCCs")
ax3.set_yticks(np.arange(0, 20, 2))
ax3.set_xticks(list(range(6)))

im4 = specshow(mfcc2, x_axis="time", ax=ax4, sr=sr2, vmin=-700, vmax=200)
cb4 = fig.colorbar(im4, ax=ax4)
ax4.set_xlabel("")
ax4.set_ylabel("MFCC Index")
ax4.set_title("(b) No-Rain MFCCs")
ax4.set_yticks(np.arange(0, 20, 2))
ax4.set_xticks(list(range(6)))


plt.rcParams.update(params)
fig.set_size_inches(18, 12)
fig.tight_layout(pad=1)
plt.show()
