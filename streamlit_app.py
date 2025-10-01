import streamlit as st
import joblib
import pandas as pd
from datetime import date, timedelta
from data_pipeline import fetch_openaq_city, fetch_open_meteo_archive
from preprocess_and_features import make_daily_features

@st.cache_resource
def load_model(path='model.joblib'):
    d = joblib.load(path)
    return d['model'], d['features']

st.title("EarthData→Action: PM2.5 short-term predictor (demo)")

city = st.text_input("City", value="Delhi")
end_date = date.today()
start_date = end_date - timedelta(days=120)

if st.button("Fetch data & predict next day"):
    with st.spinner("Fetching data..."):
        pm_df = fetch_openaq_city(city, start_date=start_date.isoformat(), end_date=end_date.isoformat())
        if pm_df.empty:
            st.error("No PM2.5 data found for that city/time. Try different dates or city name.")
        else:
            lat = pm_df['latitude'].median()
            lon = pm_df['longitude'].median()
            met_df = fetch_open_meteo_archive(lat, lon, start_date=start_date.isoformat(), end_date=end_date.isoformat())
            if met_df.empty:
                st.error("Couldn't fetch meteorology.")
            else:
                df_feats = make_daily_features(pm_df, met_df)
                if len(df_feats)<10:
                    st.warning("Not enough daily rows after feature creation for stable prediction.")
                else:
                    model, feature_cols = load_model()
                    last_row = df_feats.iloc[-1].copy()
                    X_pred = {}
                    X_pred['temperature'] = last_row['temperature']
                    X_pred['relativehumidity'] = last_row['relativehumidity']
                    X_pred['windspeed'] = last_row['windspeed']
                    X_pred['pm25_lag_1'] = last_row['pm25']
                    X_pred['pm25_lag_2'] = df_feats.iloc[-2]['pm25'] if len(df_feats)>=2 else last_row['pm25']
                    X_pred['pm25_lag_3'] = df_feats.iloc[-3]['pm25'] if len(df_feats)>=3 else last_row['pm25']
                    X_pred['pm25_lag_7'] = df_feats.iloc[-7]['pm25'] if len(df_feats)>=7 else last_row['pm25']
                    X_pred['pm25_ma_3'] = df_feats['pm25'].rolling(3).mean().iloc[-1]
                    X_pred['dayofyear'] = (pd.to_datetime(last_row['date']) + pd.Timedelta(days=1)).dayofyear

                    X_in = pd.DataFrame([X_pred])[feature_cols]
                    pred = model.predict(X_in)[0]
                    st.metric("Predicted next-day PM2.5 (µg/m³)", f"{pred:.1f}")
                    st.write("Latest observed daily PM2.5 (last date):", last_row['date'], f"{last_row['pm25']:.1f} µg/m³")
                    st.line_chart(df_feats.set_index('date')['pm25'])

st.markdown("---")
st.markdown("**Notes:** Model is a demo. For production you should: (1) add more features (EO NO2, AOD), (2) do proper cross-validation, (3) retrain frequently, (4) add uncertainty estimates.")
