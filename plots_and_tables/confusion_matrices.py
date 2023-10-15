import numpy as np
import keras
from keras import layers, optimizers, regularizers
from sklearn.metrics import confusion_matrix


def build_and_compile_model(layer_size, l2_strength, num_hidden_layers, activation_function, learning_rate):

    model_layers = [
        layers.Dense(layer_size, activation=activation_function, kernel_regularizer=regularizers.L2(l2=l2_strength))
    ] * num_hidden_layers
    model_layers += [layers.Dense(1)]
    model = keras.Sequential(model_layers)
    model.compile(loss="mse", optimizer=optimizers.Adam(learning_rate=learning_rate))

    return model


#%% Random Oversampling
file_path = "../data_generation/random_oversampling_data"

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

# normalization
def normalize(data):
    return (data - np.nanmean(X_train, axis=0)) / np.nanstd(X_train, axis=0)


X_train_norm = normalize(X_train)
X_eval_norm = normalize(X_eval)
X_test_norm = normalize(X_test)


model = keras.models.load_model(f"../model_training/ro_model")
threshold = 0.25

# Test set accuracy
y_pred_nn = model.predict(X_test_norm)
m = keras.metrics.BinaryAccuracy(threshold=threshold)
m.update_state(y_test.squeeze(), y_pred_nn.squeeze())
print(f"Test set accuracy: {m.result().numpy()}")
cm_test = confusion_matrix(y_test.squeeze(), y_pred_nn.squeeze() >= threshold)
print(cm_test)
y_pred_test = y_pred_nn.squeeze() < threshold
no_rain_idx_test = np.arange(len(no_rain_test))
bad_indices_test_ro = no_rain_idx_test[
    np.where(y_pred_test[: len(y_pred_test) // 2] | ~y_pred_test[len(y_pred_test) // 2 :])[0]
]

# Eval set accuracy
y_pred_nn = model.predict(X_eval_norm)
m = keras.metrics.BinaryAccuracy(threshold=threshold)
m.update_state(y_eval.squeeze(), y_pred_nn.squeeze())
print(f"Eval set accuracy: {m.result().numpy()}")
cm_eval = confusion_matrix(y_eval.squeeze(), y_pred_nn.squeeze() >= threshold)
print(cm_eval)
y_pred_eval = y_pred_nn.squeeze() < threshold
no_rain_idx_eval = np.arange(len(no_rain_eval))
bad_indices_eval_ro = no_rain_idx_eval[
    np.where(y_pred_eval[: len(y_pred_eval) // 2] | ~y_pred_eval[len(y_pred_eval) // 2 :])[0]
]

# Training set accuracy
y_pred_nn = model.predict(X_train_norm)
m = keras.metrics.BinaryAccuracy(threshold=threshold)
m.update_state(y_train.squeeze(), y_pred_nn.squeeze())
print(f"Training set accuracy: {m.result().numpy()}")
cm_train = confusion_matrix(y_train.squeeze(), y_pred_nn.squeeze() >= threshold)
print(cm_train)
print("-" * 60)


#%% Random Augmentation
file_path = "../data_generation/random_augmentation_data"

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

# normalization
def normalize(data):
    return (data - np.nanmean(X_train, axis=0)) / np.nanstd(X_train, axis=0)


X_train_norm = normalize(X_train)
X_eval_norm = normalize(X_eval)
X_test_norm = normalize(X_test)


model = keras.models.load_model(f"../model_training/ra_model")
threshold = 0.4

# Test set accuracy
y_pred_nn = model.predict(X_test_norm)
m = keras.metrics.BinaryAccuracy(threshold=threshold)
m.update_state(y_test.squeeze(), y_pred_nn.squeeze())
print(f"Test set accuracy: {m.result().numpy()}")
cm_test = confusion_matrix(y_test.squeeze(), y_pred_nn.squeeze() >= threshold)
print(cm_test)
y_pred_test = y_pred_nn.squeeze() < threshold
no_rain_idx_test = np.arange(len(no_rain_test))
false_negatives_test_ra = no_rain_idx_test[np.where(y_pred_test[: len(y_pred_test) // 2])[0]]
false_positives_test_ra = no_rain_idx_test[np.where(~y_pred_test[len(y_pred_test) // 2 :])[0]] + len(y_pred_test) // 2

# Eval set accuracy
y_pred_nn = model.predict(X_eval_norm)
m = keras.metrics.BinaryAccuracy(threshold=threshold)
m.update_state(y_eval.squeeze(), y_pred_nn.squeeze())
print(f"Eval set accuracy: {m.result().numpy()}")
cm_eval = confusion_matrix(y_eval.squeeze(), y_pred_nn.squeeze() >= threshold)
print(cm_eval)
y_pred_eval = y_pred_nn.squeeze() < threshold
no_rain_idx_eval = np.arange(len(no_rain_eval))
false_negatives_eval_ra = no_rain_idx_eval[np.where(y_pred_eval[: len(y_pred_eval) // 2])[0]]
false_positives_eval_ra = no_rain_idx_eval[np.where(~y_pred_eval[len(y_pred_eval) // 2 :])[0]] + len(y_pred_eval) // 2

# Training set accuracy
y_pred_nn = model.predict(X_train_norm)
m = keras.metrics.BinaryAccuracy(threshold=threshold)
m.update_state(y_train.squeeze(), y_pred_nn.squeeze())
print(f"Training set accuracy: {m.result().numpy()}")
cm_train = confusion_matrix(y_train.squeeze(), y_pred_nn.squeeze() >= threshold)
print(cm_train)
print("-" * 60)
