"""Keras model that early-stops on the test set (example input for pariksha)."""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

X = np.load("signals.npy")
y = np.load("labels.npy")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = MinMaxScaler().fit_transform(X_test)

model = keras.Sequential([keras.layers.Dense(64, activation="relu"), keras.layers.Dense(4)])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test),
          callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
print(model.evaluate(X_test, y_test))
