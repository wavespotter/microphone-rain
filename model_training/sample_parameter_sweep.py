import numpy as np
import keras
from keras import layers, optimizers, regularizers
from keras.callbacks import EarlyStopping

file_path = "../data_generation/random_oversampling_data"
out_path = "/where/to/save/models"

rain_train = np.load(f"{file_path}/rain_data_train.npy")
rain_eval = np.load(f"{file_path}/rain_data_eval.npy")
rain_test = np.load(f"{file_path}/rain_data_test.npy")

no_rain_train = np.load(f"{file_path}/no_rain_data_train.npy")
no_rain_eval = np.load(f"{file_path}/no_rain_data_eval.npy")
no_rain_test = np.load(f"{file_path}/no_rain_data_test.npy")

X_train = np.concatenate((rain_train, no_rain_train))
y_train = np.concatenate((np.ones((rain_train.shape[0],)), np.zeros((no_rain_train.shape[0]))))

X_eval = np.concatenate((rain_eval, no_rain_eval))
y_eval = np.concatenate((np.ones((rain_eval.shape[0],)), np.zeros((no_rain_eval.shape[0]))))

X_test = np.concatenate((rain_test, no_rain_test))
y_test = np.concatenate((np.ones((rain_test.shape[0],)), np.zeros((no_rain_test.shape[0]))))
#%% normalization
def normalize(data):
    return (data - np.nanmean(X_train, axis=0)) / np.nanstd(X_train, axis=0)


X_train_norm = normalize(X_train)
X_eval_norm = normalize(X_eval)
X_test_norm = normalize(X_test)

#%% Parameter sweep
def build_and_compile_model(layer_size, l2_strength, num_hidden_layers, activation_function, learning_rate):

    model_layers = [
        layers.Dense(layer_size, activation=activation_function, kernel_regularizer=regularizers.L2(l2=l2_strength))
    ] * num_hidden_layers
    model_layers += [layers.Dense(1)]
    model = keras.Sequential(model_layers)
    model.compile(loss="mse", optimizer=optimizers.Adam(learning_rate=learning_rate))

    return model


# Parameters
callback = EarlyStopping(monitor="val_loss", patience=100, restore_best_weights=True, start_from_epoch=1000)
layer_sizes = [4, 8, 16]
l2_list = [1e-5, 1e-3, 1e-1]
num_hidden_layers_list = [8]
activation_function_list = ["relu"]
learning_rate_list = [1e-5, 1e-4, 1e-3]
binary_threshold = np.arange(0.01, 1.0, 0.01)

out = {}
kk = 0
for layer_size in layer_sizes:
    for l2_strength in l2_list:
        for num_hidden_layers in num_hidden_layers_list:
            for activation_function in activation_function_list:
                for learning_rate in learning_rate_list:
                    model = build_and_compile_model(
                        layer_size, l2_strength, num_hidden_layers, activation_function, learning_rate
                    )
                    history = model.fit(
                        X_train_norm,
                        y_train,
                        validation_data=(X_eval_norm, y_eval),
                        epochs=2000,
                        verbose=0,
                        callbacks=callback,
                    )

                    y_pred = model.predict(X_eval_norm)
                    eval_score = np.zeros_like(binary_threshold)
                    for ii, bt in enumerate(binary_threshold):
                        m = keras.metrics.BinaryAccuracy(threshold=bt)
                        m.update_state(y_eval.squeeze(), y_pred.squeeze())
                        eval_score[ii] = m.result().numpy()
                    optimal_threshold = binary_threshold[np.argmax(eval_score)]
                    best_eval_score = np.max(eval_score)
                    out[
                        (
                            kk,
                            layer_size,
                            l2_strength,
                            num_hidden_layers,
                            activation_function,
                            learning_rate,
                            optimal_threshold,
                        )
                    ] = best_eval_score
                    print(
                        f"layer_size = {layer_size}, alpha = {l2_strength}, layers={num_hidden_layers}, lr_init={learning_rate}, act func={activation_function}, threshold={optimal_threshold}, score={best_eval_score}"
                    )
                    model.save(f"{out_path}/model_{kk}")
                    np.save(f"{out_path}/keras_sweep.npy", out)
                    kk += 1
