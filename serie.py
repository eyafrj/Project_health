import os
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
import warnings
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# 1. Connexion PostgreSQL
def get_db_connection():
    USERNAME = os.getenv("POSTGRES_USER", "postgres")
    PASSWORD = os.getenv("POSTGRES_PASSWORD", "123456789")
    HOST = os.getenv("POSTGRES_HOST", "localhost")
    PORT = os.getenv("POSTGRES_PORT", "5432")
    DATABASE = os.getenv("POSTGRES_DB", "DBM")
    DATABASE_URL = f"postgresql+psycopg2://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
    return create_engine(DATABASE_URL, connect_args={'client_encoding': 'LATIN1'})

# 2. Requête SQL
sql = text("""SELECT date_generee FROM "DIM_PATIENT_STAY" """)

# 3. Lecture
engine = get_db_connection()
with engine.connect().execution_options(autocommit=True) as conn:
    df = pd.read_sql_query(sql, conn)

# 4. Prétraitement
df["date_generee"] = pd.to_datetime(df["date_generee"], errors="coerce")
df = df.dropna(subset=["date_generee"])
df["month"] = df["date_generee"].dt.to_period("M").dt.to_timestamp()
monthly = df.groupby("month").size().reset_index(name="y")
monthly = monthly.rename(columns={"month": "ds"})
monthly = monthly[monthly["y"] > 0]

print(f"Total des données historiques : {len(monthly)}")

start_history_date = pd.Timestamp('2025-07-01')
monthly_for_plot = monthly[monthly["ds"] >= start_history_date]

start_forecast_date = pd.Timestamp('2027-02-01')
n_forecast = 24
future_dates = pd.date_range(start=start_forecast_date, periods=n_forecast, freq="MS")
print(f"Dates futures générées à partir de {start_forecast_date} : {future_dates}")

ts_data = monthly.set_index("ds")["y"].fillna(method='ffill')

# 7. Holt-Winters
def holt_winters_model(data, n_forecast):
    try:
        if len(data) >= 24:
            model = ExponentialSmoothing(
                data,
                trend=None,
                seasonal="mul",
                seasonal_periods=12,
                use_boxcox=True,
                initialization_method="estimated"
            ).fit(optimized=True)
        else:
            model = ExponentialSmoothing(
                data,
                trend=None,
                seasonal=None,
                use_boxcox=True,
                initialization_method="estimated"
            ).fit(optimized=True)
        forecast = model.forecast(n_forecast)
        return model, forecast.clip(lower=0)
    except Exception as e:
        print(f"Erreur Holt-Winters: {e}")
        return None, pd.Series([data.mean()] * n_forecast)

# 8. ARIMA
def arima_model(data, n_forecast, future_dates):
    try:
        model = auto_arima(
            data,
            start_p=1, start_q=1,
            max_p=5, max_q=5,
            seasonal=True,
            m=12,
            d=None, D=1,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            seasonal_test="ch",
            n_jobs=-1
        )
        fit = model.fit(data)
        forecast = fit.predict(n_periods=n_forecast)
        return fit, pd.Series(forecast, index=future_dates).clip(lower=0)
    except Exception as e:
        print(f"Erreur ARIMA: {e}")
        return None, pd.Series([data.mean()] * n_forecast, index=future_dates)

# 9. SARIMA
def sarima_model(data, n_forecast, future_dates):
    try:
        model = SARIMAX(
            data,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)
        forecast = results.get_forecast(steps=n_forecast).predicted_mean
        return results, pd.Series(forecast, index=future_dates).clip(lower=0)
    except Exception as e:
        print(f"Erreur SARIMA: {e}")
        return None, pd.Series([data.mean()] * n_forecast, index=future_dates)

# 10. Moyenne mobile
def moving_average(data, n_forecast, future_dates):
    window_size = min(6, len(data))
    ma_value = data.rolling(window_size).mean().iloc[-1]
    return pd.Series([ma_value] * n_forecast, index=future_dates)

# 11. Évaluation des prévisions
def evaluate_forecast(actual, predicted, model_name):
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    print(f"{model_name} -> MAE: {mae:.2f}, R²: {r2:.2f}, MAPE: {mape:.2f}%")

# 12. Génération des prévisions
print("\nGénération des prévisions...")
hw_model, forecast_hw = holt_winters_model(ts_data, n_forecast)
arima_fit, forecast_arima = arima_model(ts_data.values, n_forecast, future_dates)
sarima_fit, forecast_sarima = sarima_model(ts_data.values, n_forecast, future_dates)
forecast_ma = moving_average(ts_data, n_forecast, future_dates)

# 13. Évaluations + MAPE + AIC
try:
    in_sample_hw = hw_model.fittedvalues
    evaluate_forecast(ts_data, in_sample_hw, "Holt-Winters")
    
except Exception as e:
    print(f"Erreur évaluation Holt-Winters: {e}")

try:
    in_sample_arima = pd.Series(arima_fit.predict_in_sample(), index=ts_data.index)
    evaluate_forecast(ts_data, in_sample_arima, "ARIMA")
    
except Exception as e:
    print(f"Erreur évaluation ARIMA: {e}")

try:
    in_sample_sarima = sarima_fit.fittedvalues
    evaluate_forecast(ts_data, in_sample_sarima, "SARIMA")
    
except Exception as e:
    print(f"Erreur évaluation SARIMA: {e}")

# Moyenne mobile
try:
    ma_window = min(6, len(ts_data))
    ma_in_sample = ts_data.rolling(ma_window).mean().dropna()
    ts_data_cut = ts_data[-len(ma_in_sample):]
    evaluate_forecast(ts_data_cut, ma_in_sample, "Moyenne Mobile")
except Exception as e:
    print(f"Erreur évaluation Moyenne Mobile: {e}")

# 14. Affichage
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(monthly_for_plot["ds"], monthly_for_plot["y"], 'ko-', label='Historique')
plt.plot(future_dates, forecast_hw, 'b--o', label='Prévision')
plt.axvline(start_forecast_date, color='red', linestyle='--')
plt.title("Prévision Holt-Winters")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(monthly_for_plot["ds"], monthly_for_plot["y"], 'ko-', label='Historique')
plt.plot(future_dates, forecast_arima, 'g--s', label='Prévision')
plt.axvline(start_forecast_date, color='red', linestyle='--')
plt.title("Prévision ARIMA")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(monthly_for_plot["ds"], monthly_for_plot["y"], 'ko-', label='Historique')
plt.plot(future_dates, forecast_sarima, 'r--d', label='Prévision')
plt.axvline(start_forecast_date, color='red', linestyle='--')
plt.title("Prévision SARIMA")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(monthly_for_plot["ds"], monthly_for_plot["y"], 'ko-', label='Historique')
plt.plot(future_dates, forecast_ma, 'm--x', label='Prévision')
plt.axvline(start_forecast_date, color='red', linestyle='--')
plt.title("Prévision Moyenne Mobile")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
