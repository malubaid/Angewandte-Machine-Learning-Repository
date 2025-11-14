# models.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback


# -----------------------------------
# 🔥 Train/test split
# -----------------------------------

def prepare_train_test(df_encoded):
    label_cols = [c for c in df_encoded.columns if c.startswith("label_")]
    feature_cols = [c for c in df_encoded.columns if not c.startswith("label_")]

    y_onehot = df_encoded[label_cols].values.astype(int)
    y_labels = y_onehot.argmax(axis=1)

    # Chronological per-class split
    df = df_encoded.copy()
    df["y_label"] = y_labels

    train_idx, test_idx = [], []
    for lbl, grp in df.groupby("y_label"):
        n = len(grp)
        n_train = int(0.8 * n)
        train_idx += grp.index[:n_train].tolist()
        test_idx += grp.index[n_train:].tolist()

    X_train = df.loc[train_idx, feature_cols].values
    X_test = df.loc[test_idx, feature_cols].values
    y_train = y_labels[train_idx]
    y_test = y_labels[test_idx]

    X_train_scaled = StandardScaler().fit_transform(X_train)
    X_test_scaled = StandardScaler().fit(X_train).transform(X_test)

    class_names = [str(i) for i in range(1, 8)]

    return (
        X_train_scaled, X_test_scaled,
        y_train, y_test,
        y_onehot[train_idx], y_onehot[test_idx],
        label_cols, feature_cols, class_names,
        None
    )


# -----------------------------------
# 🔥 Random Forest
# -----------------------------------

def train_random_forest(X_train, y_train, X_test, y_test, class_names):
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\nRandom Forest Accuracy:", acc)
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot()
    plt.show()

    return rf, acc


# -----------------------------------
# 🔥 kNN sweep
# -----------------------------------

def sweep_knn(X_train, y_train, X_test, y_test, class_names):
    k_values = list(range(3, 18, 2))
    test_acc = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        test_acc.append(knn.score(X_test, y_test))

    best_k = k_values[np.argmax(test_acc)]
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_train, y_train)

    print(f"\nBest k: {best_k}")
    print("Test Accuracy:", max(test_acc))

    return best_knn, best_k, max(test_acc)


# -----------------------------------
# 🔥 NN architecture visualization
# -----------------------------------

def plot_nn_architecture():
    print("Showing Neural Network Architecture...")

    layer_sizes = [13, 64, 32, 7]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")

    x = np.arange(len(layer_sizes))
    ymax = max(layer_sizes)

    for i, size in enumerate(layer_sizes):
        ys = np.linspace(ymax, 0, size)
        ax.scatter(np.full(size, x[i]), ys)

    plt.title("NN Architecture: 13 → 64 → 32 →7")
    plt.show()


# -----------------------------------
# 🔥 Train Neural Network
# -----------------------------------

class TargetAccuracy(Callback):
    def __init__(self, target=0.79):
        super().__init__()
        self.target = target

    def on_epoch_end(self, epoch, logs=None):
        if logs["val_accuracy"] >= self.target:
            print("\nReached 79% validation accuracy. Stopping...")
            self.model.stop_training = True


def train_neural_network(X_train, y_train_onehot, X_test, y_test_onehot, target_val_accuracy):

    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(7, activation="sigmoid")
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(0.001),
        metrics=["accuracy"]
    )

    callback = TargetAccuracy(target=target_val_accuracy)

    history = model.fit(
        X_train, y_train_onehot,
        validation_data=(X_test, y_test_onehot),
        epochs=50,
        batch_size=32,
        callbacks=[callback],
        verbose=1
    )

    print("\nNN Test Accuracy:", history.history["val_accuracy"][-1])

    return model, history, history.history["val_accuracy"][-1]


# -----------------------------------
# 🔥 Ensemble Voting
# -----------------------------------

def weighted_ensemble(rf, knn, nn_model, X_test, y_test, class_names, weights):
    rf_pred = rf.predict(X_test)
    knn_pred = knn.predict(X_test)
    nn_pred = nn_model.predict(X_test).argmax(axis=1)

    pred_matrix = np.vstack([
        np.repeat(rf_pred[np.newaxis, :], weights["rf"], axis=0),
        np.repeat(nn_pred[np.newaxis, :], weights["nn"], axis=0),
        np.repeat(knn_pred[np.newaxis, :], weights["knn"], axis=0)
    ])

    ensemble_pred = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=7).argmax(),
        axis=0,
        arr=pred_matrix
    )

    acc = accuracy_score(y_test, ensemble_pred)

    print("\nEnsemble Accuracy:", acc)
    print(classification_report(y_test, ensemble_pred, target_names=class_names))

    cm = confusion_matrix(y_test, ensemble_pred)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot()
    plt.show()

    return acc, ensemble_pred
