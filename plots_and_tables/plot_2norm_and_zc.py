import matplotlib.pyplot as plt
import numpy as np


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

rain_color = "#096ED3"
no_rain_color = "sandybrown"

#%% Loading the raw
rain_train = np.load("../data_generation/random_oversampling_data/rain_data_train.npy")
rain_eval = np.load("../data_generation/random_oversampling_data/rain_data_eval.npy")
rain_test = np.load("../data_generation/random_oversampling_data/rain_data_test.npy")

rain = np.concatenate((rain_train, rain_eval, rain_test))
rain = np.unique(rain, axis=0)

no_rain_train = np.load("../data_generation/random_oversampling_data/no_rain_data_train.npy")
no_rain_eval = np.load("../data_generation/random_oversampling_data/no_rain_data_eval.npy")
no_rain_test = np.load("../data_generation/random_oversampling_data/no_rain_data_test.npy")

no_rain = np.concatenate((no_rain_train, no_rain_eval, no_rain_test))

rain_zc = rain[:, 0]
rain_norm = rain[:, 1]
no_rain_zc = no_rain[:, 0]
no_rain_norm = no_rain[:, 1]

rain_zc_mean = np.nanmean(rain_zc / 5)
no_rain_zc_mean = np.nanmean(no_rain_zc / 5)
rain_norm_mean = np.nanmean(rain_norm)
no_rain_norm_mean = np.nanmean(no_rain_norm)

#%% Plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
h1 = ax1.hist(no_rain_zc / 5, bins="auto", label=f"No Rain, $\mu = ${int(no_rain_zc_mean)}", color=no_rain_color)
ax12 = ax1.twinx()
h2 = ax12.hist(rain_zc / 5, bins="auto", alpha=0.4, label=f"Rain, $\mu = ${int(rain_zc_mean)}", color=rain_color)
ax1.set_ylabel("Counts, No-Rain")
ax12.set_ylabel("Counts, Rain")
ax1.set_xlabel(r"ZCR (s$^{-1}$)")
ax1.set_title("(a)")

# added these three lines
lns = h1[2] + h2[2]
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)

h1 = ax2.hist(no_rain_norm, bins="auto", label=f"No Rain, $\mu = ${no_rain_norm_mean:.1f}", color=no_rain_color)
ax22 = ax2.twinx()
h2 = ax22.hist(rain_norm, bins="auto", alpha=0.4, label=f"Rain, $\mu = ${rain_norm_mean:.1f}", color=rain_color)
ax2.set_ylabel("Counts, No-Rain")
ax22.set_ylabel("Counts, Rain")
ax2.set_xlabel(r"$l^2$-norm")
ax2.set_title("(b)")

# added these three lines
lns = h1[2] + h2[2]
labs = [l.get_label() for l in lns]
ax2.legend(lns, labs, loc=0)


#%% And getting the augmented data
rain_train = np.load("../data_generation/random_augmentation_data/rain_data_train.npy")
rain_eval = np.load("../data_generation/random_augmentation_data/rain_data_eval.npy")
rain_test = np.load("../data_generation/random_augmentation_data/rain_data_test.npy")

rain = np.concatenate((rain_train, rain_eval, rain_test))

no_rain_train = np.load("../data_generation/random_augmentation_data/no_rain_data_train.npy")
no_rain_eval = np.load("../data_generation/random_augmentation_data/no_rain_data_eval.npy")
no_rain_test = np.load("../data_generation/random_augmentation_data/no_rain_data_test.npy")

no_rain = np.concatenate((no_rain_train, no_rain_eval, no_rain_test))

rain_zc = rain[:, 0]
rain_norm = rain[:, 1]
no_rain_zc = no_rain[:, 0]
no_rain_norm = no_rain[:, 1]

rain_zc_mean = np.nanmean(rain_zc / 5)
no_rain_zc_mean = np.nanmean(no_rain_zc / 5)
rain_norm_mean = np.nanmean(rain_norm)
no_rain_norm_mean = np.nanmean(no_rain_norm)

#%% Plot
ax3.hist(no_rain_zc / 5, bins="auto", label=f"No Rain, $\mu = ${int(no_rain_zc_mean)}", color=no_rain_color)
ax3.hist(rain_zc / 5, bins="auto", alpha=0.4, label=f"Rain, $\mu = ${int(rain_zc_mean)}", color=rain_color)
ax3.set_ylabel("Counts")
ax3.set_xlabel(r"ZCR (s$^{-1}$)")
ax3.set_title("(c)")
ax3.legend()


ax4.hist(no_rain_norm, bins="auto", label=f"No Rain, $\mu = ${no_rain_norm_mean:.1f}", color=no_rain_color)
ax4.hist(rain_norm, bins="auto", alpha=0.4, label=f"Rain, $\mu = ${rain_norm_mean:.1f}", color=rain_color)
ax4.set_ylabel("Counts")
ax4.set_xlabel(r"$l^2$-norm")
ax4.set_title("(d)")
ax4.legend()

fig.set_size_inches(14, 12)
plt.rcParams.update(params)
fig.tight_layout(pad=0.5)
plt.show()
