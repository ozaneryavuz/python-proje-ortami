import os
import pandas as pd
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from imblearn.over_sampling import SMOTE
import pickle
import logging

# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Loglama konfigürasyonu
logging.basicConfig(level=logging.INFO)

# Veri Yükleme ve Doğrulama
def load_and_validate_data(input_file):
    try:
        data = pd.read_excel(input_file, sheet_name='Sayfa1')
        data.columns = data.columns.str.replace('ı', 'i').str.replace('Ç', 'C').str.replace('ç', 'c').str.replace('ğ', 'g')

        # Veri doğrulama
        expected_columns = ['Girdi 1', 'Girdi 2', 'Girdi 3', 'Cikti 2']
        for col in expected_columns:
            if col not in data.columns:
                raise ValueError(f"Beklenen sütun eksik: {col}")

        return data
    except Exception as e:
        logging.error(f"Error loading or validating data: {e}")
        return None

# Gelişmiş Veri Ön İşleme
def preprocess_data_advanced(data):
    numerical_features = data.select_dtypes(include=[np.number]).columns
    categorical_features = data.select_dtypes(include=['object']).columns

    numerical_pipeline = Pipeline(steps=[
        ('imputer', IterativeImputer(random_state=42)),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ]
    )
    return preprocessor.fit_transform(data), preprocessor

def feature_engineering(data):
    logging.info("Özellik mühendisliği gerçekleştiriliyor")
    new_features = pd.DataFrame()
    num_features = data.shape[1] - 1  # Assuming last column is the target

    new_features['mean_odds'] = data.iloc[:, :-1].mean(axis=1)
    new_features['std_odds'] = data.iloc[:, :-1].std(axis=1)
    new_features['var_odds'] = data.iloc[:, :-1].var(axis=1)
    new_features['range_odds'] = data.iloc[:, :-1].max(axis=1) - data.iloc[:, :-1].min(axis=1)
    new_features['sum_odds'] = data.iloc[:, :-1].sum(axis=1)
    new_features['prod_odds'] = data.iloc[:, :-1].prod(axis=1)

    return pd.concat([data, new_features], axis=1)

# Model Değerlendirme
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, mse, mae, r2

# GridSearchCV ile Random Forest Optimizasyonu
def hyperparameter_tuning(model, param_grid, X_train, y_train):
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Kar/Zarar Analizi
def get_odds(data, test_indices):
    filtered_data = data.loc[test_indices]
    girdi_columns = [col for col in filtered_data.columns if col.startswith('Girdi')]
    odds = {col: filtered_data[col].values / 100 for col in girdi_columns}
    return odds

def profit_calculator(thresholds, y_test_clean, y_pred_clean, odds):
    high_thresh, low_thresh = thresholds
    profit = 0
    for actual, pred, *odds_values in zip(y_test_clean, y_pred_clean, *odds.values()):
        result = determine_result(pred, high_thresh, low_thresh)
        if result == "Ev Sahibi Kazanır" and actual > 0:
            profit += odds_values[0] - 1
        elif result == "Beraberlik" and actual == 0:
            profit += odds_values[1] - 1
        elif result == "Deplasman Kazanır" and actual < 0:
            profit += odds_values[2] - 1
        else:
            profit -= 1
    return -profit

def optimize_thresholds(y_test_clean, y_pred_clean, odds):
    space = [Real(-5, 5, name='high_thresh'), Real(-5, 5, name='low_thresh')]
    result = gp_minimize(lambda thresholds: profit_calculator(thresholds, y_test_clean, y_pred_clean, odds), space, n_calls=150, random_state=42, verbose=True)
    return result.x, -result.fun

def save_best_thresholds(thresholds, filename):
    with open(filename, 'wb') as f:
        pickle.dump(thresholds, f)

def load_best_thresholds(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def calculate_profit_loss(y_test_clean, y_pred_clean, odds, best_thresholds):
    profit_loss = []
    prediction_result = []
    is_correct = []
    high_thresh, low_thresh = best_thresholds
    for actual, pred, *odds_values in zip(y_test_clean, y_pred_clean, *odds.values()):
        result = determine_result(pred, high_thresh, low_thresh)
        prediction_result.append(result)
        if result == "Ev Sahibi Kazanır":
            if actual > 0:
                profit_loss.append(odds_values[0] - 1)
                is_correct.append(True)
            else:
                profit_loss.append(-1)
                is_correct.append(False)
        elif result == "Beraberlik":
            if actual == 0:
                profit_loss.append(odds_values[1] - 1)
                is_correct.append(True)
            else:
                profit_loss.append(-1)
                is_correct.append(False)
        elif result == "Deplasman Kazanır":
            if actual < 0:
                profit_loss.append(odds_values[2] - 1)
                is_correct.append(True)
            else:
                profit_loss.append(-1)
                is_correct.append(False)
    return prediction_result, profit_loss, is_correct

def determine_result(pred, high_thresh, low_thresh):
    if pred > high_thresh:
        return "Ev Sahibi Kazanır"
    elif low_thresh <= pred <= high_thresh:
        return "Beraberlik"
    else:
        return "Deplasman Kazanır"

# Sonuçların Derlenmesi ve Kaydedilmesi
def compile_results(data, test_indices, y_pred_clean, prediction_result, is_correct, profit_loss, best_thresholds, max_profit):
    filtered_data = data.loc[test_indices]
    result_df = pd.DataFrame({
        'Girdi 1': filtered_data['Girdi 1'].values,
        'Girdi 2': filtered_data['Girdi 2'].values,
        'Girdi 3': filtered_data['Girdi 3'].values,
        'Gerçek Çıktı 2': filtered_data['Cikti 2'].values,
        'Tahmin Edilen Çıktı 2': y_pred_clean.flatten(),
        'Tahmin Sonucu': prediction_result,
        'Doğru Tahmin': is_correct,
        'Kar/Zarar': profit_loss
    })

    profit_summary = pd.DataFrame({
        'Ev Sahibi Kazanır Eşiği': [best_thresholds[0]],
        'Deplasman Kazanır Eşiği': [best_thresholds[1]],
        'Max Kar': [max_profit]
    })

    result_df = pd.concat([profit_summary, result_df], ignore_index=True)
    return result_df

def save_results(result_df, output_file):
    result_df.to_excel(output_file, index=False)
    print(f"Results saved to '{output_file}'")

# Yeni Verilerde Tahmin ve Karar Alma
def save_predictions_and_decisions(new_data, predictions, decisions, output_file):
    result_df = pd.DataFrame(new_data)
    result_df['Tahmin_rf'] = predictions.flatten()
    result_df['Karar_rf'] = decisions
    result_df.to_excel(output_file, index=False)
    print(f"YeniGirdiX tahmin ve karar sonuçları '{output_file}' dosyasına kaydedildi.")

def apply_thresholds_and_make_decisions(predictions, thresholds):
    decisions = []
    for pred in predictions:
        decision = determine_result(pred, thresholds[0], thresholds[1])
        decisions.append(decision)
    return decisions

# Ana İş Akışı
def main():
    input_file = 'Girdi ve Ciktilar.xlsx'
    data = load_and_validate_data(input_file)
    if data is None or data.empty:
        logging.error("Veri yüklenemedi veya doğrulanamadı.")
        return

    data = feature_engineering(data)
    data.dropna(inplace=True)

    X = data.drop(columns="Cikti 2")
    y = data["Cikti 2"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=96)
    X_train_scaled, preprocessor = preprocess_data_advanced(X_train)

    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    X_test_scaled = preprocessor.transform(X_test)

    # Define selected features based on your specific needs
    selected_features = X.columns.tolist()  # Assuming you want to use all features

    X_train_selected = pd.DataFrame(X_train_scaled, columns=X.columns)[selected_features]
    X_test_selected = pd.DataFrame(X_test_scaled, columns=X.columns)[selected_features]

    class_distribution = y_train.value_counts()
    min_class_samples = class_distribution.min()
    k_neighbors_value = max(1, min(5, min_class_samples - 1))

    if min_class_samples > 1:
        X_resampled, y_resampled = SMOTE(k_neighbors=k_neighbors_value).fit_resample(X_train_selected, y_train)
    else:
        X_resampled, y_resampled = X_train_selected, y_train

    # Hyperparameter tuning for Random Forest
    rf_params = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30]}
    rf_best = hyperparameter_tuning(RandomForestRegressor(random_state=42), rf_params, X_resampled, y_resampled)

    # Evaluate Random Forest model
    y_pred_rf, mse_rf, mae_rf, r2_rf = evaluate_model(rf_best, X_test_selected, y_test)

    test_indices = y_test.index
    odds = get_odds(data, test_indices)

    # Optimize thresholds for Random Forest
    best_thresholds_rf, max_profit_rf = optimize_thresholds(y_test.to_numpy(), y_pred_rf, odds)
    prediction_result_rf, profit_loss_rf, is_correct_rf = calculate_profit_loss(y_test.to_numpy(), y_pred_rf, odds, best_thresholds_rf)

    # Compile results for Random Forest
    result_df_rf = compile_results(data, test_indices, y_pred_rf, prediction_result_rf, is_correct_rf, profit_loss_rf, best_thresholds_rf, max_profit_rf)

    # Save best thresholds
    save_best_thresholds(best_thresholds_rf, 'best_thresholds_rf.pkl')

    # Save results
    save_results(result_df_rf, 'rf_tahmin_sonuclari.xlsx')

    # For new data
    new_data_file = 'YeniGirdiX.xlsx'
    new_data = load_and_validate_data(new_data_file)
    new_data = new_data.drop(columns=["Cikti 1", "Cikti 2"], errors='ignore')
    new_data = feature_engineering(new_data)
    missing_cols = set(X_train_selected.columns) - set(new_data.columns)
    for col in missing_cols:
        new_data[col] = 0
    new_data = new_data[X_train_selected.columns]

    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)

    new_data_scaled = preprocessor.transform(new_data)
    if new_data_scaled.shape[1] != X_train_selected.shape[1]:
        raise ValueError(f"Expected input shape ({X_train_selected.shape[1]}) but got ({new_data_scaled.shape[1]})")

    new_data_scaled_reshaped = {

        'rf': new_data_scaled,


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
