# utils.py

import pathlib
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------
# No hardcoded ROOT_PATH — everything is passed from main
# -----------------------------------------------------

SAMPLE_RATE = 48000

SPECIFIC_LABELS = ['E', 'Z', 'R', 'H', 'F', 'X', 'S']

LABEL_MAPPING = {
    'E': 1, 'Z': 2, 'R': 3,
    'H': 4, 'F': 5, 'X': 6, 'S': 7
}


# -----------------------------------
# 🔥 Audio feature extraction
# -----------------------------------

def get_features(clip, sample_rate=SAMPLE_RATE):
    try:
        mfccs = librosa.feature.mfcc(
            y=clip,
            sr=sample_rate,
            n_mfcc=13
        )
        return np.mean(mfccs, axis=1)
    except Exception as e:
        print(f"Feature error on clip: {e}")
        return None


def split_files(file_path, clip_length_sec=2, sample_rate=SAMPLE_RATE):
    try:
        y, sr = librosa.load(file_path, sr=sample_rate)
        clip_samples = clip_length_sec * sample_rate

        y_split = []
        i = 0
        while i <= len(y):
            y_split.append(y[i:i + clip_samples])
            i += clip_samples

        return y_split
    except Exception as e:
        print(f"Error splitting file {file_path}: {e}")
        return None


# -----------------------------------
# 🔥 Create dataframes from audio
# -----------------------------------

def build_dataframes_by_label(root_path: str):
    dataframes_by_label = {label: [] for label in SPECIFIC_LABELS}

    for file_path in pathlib.Path(root_path).rglob("*.*"):
        label = file_path.stem[0]

        if label in SPECIFIC_LABELS:
            clips = split_files(file_path)
            if clips is None:
                continue

            for clip in clips:
                features = get_features(clip)
                if features is not None:
                    dataframes_by_label[label].append({
                        "MFCCs": features,
                        "label": label
                    })

            print(f"Processed: {file_path}")

    return {label: pd.DataFrame(data)
            for label, data in dataframes_by_label.items()}


# -----------------------------------
# 🔥 Expand MFCC arrays & one-hot encode
# -----------------------------------

def expand_mfcc_columns_and_encode(dataframes_by_label: dict):

    # Map labels to integers
    for lbl in dataframes_by_label:
        df = dataframes_by_label[lbl]
        df["label"] = df["label"].map(LABEL_MAPPING)

    # Expand MFCC columns
    for lbl, df in dataframes_by_label.items():
        for i in range(1, 14):
            df[f"mfcc{i}"] = df["MFCCs"].apply(lambda x, idx=i: float(x[idx-1]))

        df.drop(columns=["MFCCs"], inplace=True)

    # Combine all
    combined_df = pd.concat(dataframes_by_label.values(), ignore_index=True)

    # One-hot encode labels
    df_encoded = pd.get_dummies(combined_df, columns=["label"], prefix="label")

    # Ensure consistent class columns
    for i in range(1, 8):
        col = f"label_{i}"
        if col not in df_encoded:
            df_encoded[col] = 0

    # Order columns
    mfcc_cols = [c for c in df_encoded.columns if c.startswith("mfcc")]
    label_cols = [f"label_{i}" for i in range(1, 8)]
    df_encoded = df_encoded[mfcc_cols + label_cols]

    return combined_df, df_encoded


# -----------------------------------
# 🔥 Save CSV
# -----------------------------------

def save_dataframes(combined_df, df_encoded, base_path="."):
    combined_df.to_csv(f"{base_path}/combined_df.csv", index=False)
    df_encoded.to_csv(f"{base_path}/df_encoded.csv", index=False)

    print(f"Saved: {base_path}/combined_df.csv")
    print(f"Saved: {base_path}/df_encoded.csv")


# -----------------------------------
# 🔥 Correlation & histograms
# -----------------------------------

def plot_correlation_and_drop(combined_df):
    corr_matrix = combined_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

    # Remove unwanted MFCCs
    for col in ["mfcc12", "mfcc5", "mfcc3"]:
        if col in combined_df.columns:
            combined_df.drop(columns=[col], inplace=True)

    return combined_df


def plot_histograms(df):
    df.hist(figsize=(12, 10))
    plt.show()
