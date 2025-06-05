import os
import pickle
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import SMOTE

from data_loader import load_and_validate_data
from preprocessing import preprocess_data_advanced, feature_engineering
from modeling import hyperparameter_tuning, evaluate_model
from evaluation import (
    get_odds,
    optimize_thresholds,
    calculate_profit_loss,
    compile_results,
    save_best_thresholds,
    save_results,
    save_predictions_and_decisions,
    apply_thresholds_and_make_decisions,
)

# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def main():
    """Run the end-to-end training and prediction workflow."""
    # ------------------------
    # Training phase
    # ------------------------

    input_file = 'Girdi ve Ciktilar.xlsx'
    data = load_and_validate_data(input_file)
    if data is None or data.empty:
        logging.error("Veri yüklenemedi veya doğrulanamadı.")
        return

    # Generate additional features and drop rows with missing values
    data = feature_engineering(data)
    data.dropna(inplace=True)

    X = data.drop(columns="Cikti 2")
    y = data["Cikti 2"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=96)
    # Fit preprocessing pipeline using training data
    X_train_scaled, preprocessor = preprocess_data_advanced(X_train)

    # Persist preprocessor for later use with new data
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    # Apply same preprocessing to the test set
    X_test_scaled = preprocessor.transform(X_test)

    # Define selected features based on your specific needs
    selected_features = X.columns.tolist()  # Use all features

    X_train_selected = pd.DataFrame(X_train_scaled, columns=X.columns)[selected_features]
    X_test_selected = pd.DataFrame(X_test_scaled, columns=X.columns)[selected_features]

    class_distribution = y_train.value_counts()
    min_class_samples = class_distribution.min()
    k_neighbors_value = max(1, min(5, min_class_samples - 1))

    # Balance classes using SMOTE when possible
    if min_class_samples > 1:
        X_resampled, y_resampled = SMOTE(k_neighbors=k_neighbors_value).fit_resample(X_train_selected, y_train)
    else:
        X_resampled, y_resampled = X_train_selected, y_train

    # Hyperparameter tuning for Random Forest
    rf_params = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30]}
    rf_best = hyperparameter_tuning(RandomForestRegressor(random_state=42), rf_params, X_resampled, y_resampled)

    # Evaluate the tuned model
    y_pred_rf, mse_rf, mae_rf, r2_rf = evaluate_model(rf_best, X_test_selected, y_test)

    test_indices = y_test.index
    # Collect betting odds for profit calculation
    odds = get_odds(data, test_indices)

    # Optimize decision thresholds to maximize profit
    best_thresholds_rf, max_profit_rf = optimize_thresholds(y_test.to_numpy(), y_pred_rf, odds)
    prediction_result_rf, profit_loss_rf, is_correct_rf = calculate_profit_loss(y_test.to_numpy(), y_pred_rf, odds, best_thresholds_rf)

    # Compile and save evaluation results
    result_df_rf = compile_results(data, test_indices, y_pred_rf, prediction_result_rf, is_correct_rf, profit_loss_rf, best_thresholds_rf, max_profit_rf)

    save_best_thresholds(best_thresholds_rf, 'best_thresholds_rf.pkl')

    save_results(result_df_rf, 'rf_tahmin_sonuclari.xlsx')

    # ------------------------
    # Prediction on new data
    # ------------------------
    new_data_file = 'YeniGirdiX.xlsx'
    new_data = load_and_validate_data(new_data_file)
    new_data = new_data.drop(columns=["Cikti 1", "Cikti 2"], errors='ignore')
    new_data = feature_engineering(new_data)
    missing_cols = set(X_train_selected.columns) - set(new_data.columns)
    for col in missing_cols:
        new_data[col] = 0  # ensure column order matches training data
    new_data = new_data[X_train_selected.columns]

    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)

    new_data_scaled = preprocessor.transform(new_data)
    if new_data_scaled.shape[1] != X_train_selected.shape[1]:
        raise ValueError(f"Expected input shape ({X_train_selected.shape[1]}) but got ({new_data_scaled.shape[1]})")

    # Example placeholder for additional models
    new_data_scaled_reshaped = {
        'rf': new_data_scaled,
    }
    new_data_predictions = {model_name: models[model_name].predict(new_data_scaled_reshaped[model_name]) for model_name in models}

    new_data_decisions = {
        model_name: apply_thresholds_and_make_decisions(new_data_predictions[model_name].flatten(), thresholds[model_name])
        for model_name in models
    }

    save_predictions_and_decisions(new_data, new_data_predictions, new_data_decisions, 'YeniGirdi_Tahmin_Karar_Sonuclari.xlsx')

    existing_data_file = 'YeniGirdiX_Tahmin_Karar_Sonuclari.xlsx'
    existing_data = pd.read_excel(existing_data_file)

    inputs = existing_data[['Girdi 1', 'Girdi 2', 'Girdi 3']].values
    outputs = existing_data[[f'Tahmin_{model}' for model in models]].values

    efficiency_scores = diversified_dea_analysis(inputs, outputs)

    best_decisions = []
    for i, row in existing_data.iterrows():
        best_decision = None
        max_efficiency = -1
        for model in models:
            efficiency = efficiency_scores[i]
            if efficiency is not None and efficiency > max_efficiency:
                max_efficiency = efficiency
                best_decision = row[f'Karar_{model}']
        best_decisions.append(best_decision)

    existing_data['En Iyi Karar'] = best_decisions
    existing_data.to_excel('YeniGirdiX_Tahmin_Karar_Sonuclari_DEA.xlsx', index=False)
    print("YeniGirdiX tahmin ve karar sonuçları DEA ile belirlenmiş hali 'YeniGirdi_Tahmin_Karar_Sonuclari_DEA.xlsx' dosyasına kaydedildi.")

if __name__ == '__main__':
    main()
