import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from data_pipeline import fetch_openaq_city, fetch_open_meteo_archive
from preprocess_and_features import make_daily_features

def train_and_save(city='Delhi', start_date='2024-01-01', end_date=None, model_path='model.joblib'):
    pm_df = fetch_openaq_city(city, start_date=start_date, end_date=end_date)
    if pm_df.empty:
        print("No PM data found. Exiting.")
        return
    lat = pm_df['latitude'].median()
    lon = pm_df['longitude'].median()
    met_df = fetch_open_meteo_archive(lat, lon, start_date=start_date, end_date=end_date)
    if met_df.empty:
        print("No meteorology data. Exiting.")
        return
    df_features = make_daily_features(pm_df, met_df)
    feature_cols = ['temperature','relativehumidity','windspeed',
                    'pm25_lag_1','pm25_lag_2','pm25_lag_3','pm25_lag_7','pm25_ma_3','dayofyear']
    X = df_features[feature_cols]
    y = df_features['pm25']
    split_idx = int(len(df_features) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = XGBRegressor(n_estimators=200, learning_rate=0.08, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"Test MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    joblib.dump({'model': model, 'features': feature_cols}, model_path)
    print("Saved model to", model_path)
    return model, feature_cols

if __name__ == "__main__":
    train_and_save(city='Delhi', start_date='2024-01-01', end_date=None, model_path='model.joblib')
