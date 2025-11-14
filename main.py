# main.py

import pathlib
from utils import (
    build_dataframes_by_label,
    expand_mfcc_columns_and_encode,
    save_dataframes,
    plot_correlation_and_drop,
    plot_histograms,
)
from models import (
    prepare_train_test,
    train_random_forest,
    sweep_knn,
    plot_nn_architecture,
    train_neural_network,
    weighted_ensemble,
)


def main():

    # ------------------------------------------------------
    # 🔥 Automatically detect project root and subdirectories
    # ------------------------------------------------------
    project_root = pathlib.Path(__file__).resolve().parent

    data_root = project_root / "data"
    output_dir = project_root / "output"

    print("\n=== Dynamic Path Configuration ===")
    print(f"Project root : {project_root}")
    print(f"Data folder  : {data_root}")
    print(f"Output folder: {output_dir}")
    print("==================================\n")

    # Create output directory if missing
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------
    # 🔥 STEP 1: Build dataframes from audio
    # ----------------------------------------
    print(">> Reading audio files and extracting MFCC features...")
    dataframes_by_label = build_dataframes_by_label(str(data_root))

    combined_df, df_encoded = expand_mfcc_columns_and_encode(dataframes_by_label)

    # ----------------------------------------
    # 🔥 STEP 2: Save output CSV files
    # ----------------------------------------
    print("\n>> Saving CSV files...")
    save_dataframes(combined_df, df_encoded, base_path=str(output_dir))

    # ----------------------------------------
    # 🔥 STEP 3: Feature analysis
    # ----------------------------------------
    print("\n>> Feature correlation analysis...")
    combined_df = plot_correlation_and_drop(combined_df)

    print("\n>> Plotting histograms...")
    plot_histograms(combined_df)

    # ----------------------------------------
    # 🔥 STEP 4: Prepare train/test data
    # ----------------------------------------
    (
        X_train_scaled,
        X_test_scaled,
        y_train_labels,
        y_test_labels,
        y_train_onehot,
        y_test_onehot,
        label_cols,
        feature_cols,
        class_names,
        scaler,
    ) = prepare_train_test(df_encoded)

    # ----------------------------------------
    # 🔥 STEP 5: Train models
    # ----------------------------------------
    rf, rf_acc = train_random_forest(
        X_train_scaled, y_train_labels,
        X_test_scaled, y_test_labels,
        class_names
    )

    best_knn, best_k, best_knn_acc = sweep_knn(
        X_train_scaled, y_train_labels,
        X_test_scaled, y_test_labels,
        class_names
    )

    plot_nn_architecture()

    nn_model, history, nn_acc = train_neural_network(
        X_train_scaled, y_train_onehot,
        X_test_scaled, y_test_onehot,
        target_val_accuracy=0.79
    )

    # ----------------------------------------
    # 🔥 STEP 6: Ensemble model
    # ----------------------------------------
    ensemble_acc, ensemble_pred = weighted_ensemble(
        rf, best_knn, nn_model,
        X_test_scaled, y_test_labels,
        class_names,
        weights={"rf": 3, "nn": 3, "knn": 2},
    )

    # ----------------------------------------
    # 🔥 Final summary
    # ----------------------------------------
    print("\n=== FINAL ACCURACIES ===")
    print(f"Random Forest: {rf_acc:.4f}")
    print(f"Best k-NN (k={best_k}): {best_knn_acc:.4f}")
    print(f"Neural Network: {nn_acc:.4f}")
    print(f"Ensemble:       {ensemble_acc:.4f}")
    print("=========================")

    print("\nDone.")


if __name__ == "__main__":
    main()
