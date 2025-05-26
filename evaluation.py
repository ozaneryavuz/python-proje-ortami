import pickle
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real


def get_odds(data, test_indices):
    """Return odds information for profit calculations."""
    filtered_data = data.loc[test_indices]
    girdi_columns = [col for col in filtered_data.columns if col.startswith('Girdi')]
    odds = {col: filtered_data[col].values / 100 for col in girdi_columns}
    return odds


def determine_result(pred, high_thresh, low_thresh):
    """Determine match result category based on prediction and thresholds."""
    if pred > high_thresh:
        return "Ev Sahibi Kazanır"
    elif low_thresh <= pred <= high_thresh:
        return "Beraberlik"
    else:
        return "Deplasman Kazanır"


def profit_calculator(thresholds, y_test_clean, y_pred_clean, odds):
    """Objective function used for threshold optimization."""
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
    # Negative sign because gp_minimize seeks to minimize
    return -profit


def optimize_thresholds(y_test_clean, y_pred_clean, odds):
    """Find profit maximizing high and low thresholds."""
    space = [Real(-5, 5, name='high_thresh'), Real(-5, 5, name='low_thresh')]
    result = gp_minimize(
        lambda t: profit_calculator(t, y_test_clean, y_pred_clean, odds),
        space,
        n_calls=150,
        random_state=42,
        verbose=True
    )
    return result.x, -result.fun


def save_best_thresholds(thresholds, filename):
    """Save threshold tuple to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(thresholds, f)


def load_best_thresholds(filename):
    """Load previously saved thresholds from disk."""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def calculate_profit_loss(y_test_clean, y_pred_clean, odds, best_thresholds):
    """Calculate profit/loss per observation and if prediction was correct."""
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


def compile_results(data, test_indices, y_pred_clean, prediction_result, is_correct, profit_loss, best_thresholds, max_profit):
    """Create a single dataframe holding prediction results and summary."""
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
    """Write the result dataframe to an Excel file."""
    result_df.to_excel(output_file, index=False)
    print(f"Results saved to '{output_file}'")


def save_predictions_and_decisions(new_data, predictions, decisions, output_file):
    """Persist new data predictions and chosen decisions."""
    result_df = pd.DataFrame(new_data)
    result_df['Tahmin_rf'] = predictions.flatten()
    result_df['Karar_rf'] = decisions
    result_df.to_excel(output_file, index=False)
    print(f"YeniGirdiX tahmin ve karar sonuçları '{output_file}' dosyasına kaydedildi.")


def apply_thresholds_and_make_decisions(predictions, thresholds):
    """Convert numeric predictions to categorical decisions."""
    decisions = []
    for pred in predictions:
        decision = determine_result(pred, thresholds[0], thresholds[1])
        decisions.append(decision)
    return decisions

