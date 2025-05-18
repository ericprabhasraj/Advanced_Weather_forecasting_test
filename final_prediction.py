# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from sklearn.preprocessing import RobustScaler
# from xgboost import XGBRegressor
# from tensorflow.keras.models import Model, load_model
# import joblib
# import warnings
# import ipywidgets as widgets
# from IPython.display import display, clear_output
# from matplotlib.dates import DateFormatter, HourLocator
# from sklearn.ensemble import RandomForestRegressor

# warnings.filterwarnings("ignore")

# # --- Helpers ---
# def get_season(month):
#     if month in [2, 3, 4, 5]:
#         return 0  # summer
#     elif month in [6, 7, 8, 9]:
#         return 1  # rainy
#     else:
#         return 2  # winter

# def fetch_data(csv_file, parameter):
#     df = pd.read_csv(csv_file, parse_dates=['datetime'])
#     df.set_index('datetime', inplace=True)
#     if parameter not in df.columns:
#         raise ValueError(f"Parameter '{parameter}' not found in dataset")
#     df[parameter] = np.log1p(df[parameter])  # log transform
#     return df

# def feature_engineering(df, parameter):
#     df['minute'] = df.index.minute
#     df['hour'] = df.index.hour
#     df['month'] = df.index.month
#     df['year'] = df.index.year
#     df['week'] = df.index.isocalendar().week.astype(int)
#     df['doy'] = df.index.dayofyear
#     df['season'] = df['month'].apply(get_season)
#     df['lag1'] = df[parameter].shift(1)
#     df['lag2'] = df[parameter].shift(2)
    
#     for window in [3, 6, 12]:
#         df[f'rolling_mean_{window}'] = df[parameter].rolling(window).mean()
#         df[f'rolling_std_{window}'] = df[parameter].rolling(window).std()
    
#     df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
#     df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
#     df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
#     df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
#     df.bfill(inplace=True)
#     return df

# def create_sequences(data, time_steps, forecast_horizon):
#     X, y = [], []
#     for i in range(len(data) - time_steps - forecast_horizon + 1):
#         X.append(data[i:i + time_steps])
#         y.append(data[i + time_steps:i + time_steps + forecast_horizon, 0])  # target is first column
#     return np.array(X), np.array(y)

# def evaluate_predictions(preds, y_true, scaler, test_scaled, forecast_horizon):
#     predictions, actuals = [], []
#     for i in range(forecast_horizon):
#         dummy_input = np.zeros_like(test_scaled[:len(preds)])
#         dummy_input[:, 0] = preds[:, i]
#         dummy_actual = np.zeros_like(dummy_input)
#         dummy_actual[:, 0] = y_true[:, i]
#         inv_pred = np.expm1(scaler.inverse_transform(dummy_input)[:, 0])
#         inv_actual = np.expm1(scaler.inverse_transform(dummy_actual)[:, 0])
#         predictions.append(inv_pred)
#         actuals.append(inv_actual)
#     return predictions, actuals

# def plot_predictions(timestamps, actuals, preds, horizon, parameter):
#     plt.figure(figsize=(14, 6))
#     plt.plot(timestamps, actuals[0], label='Actual')
#     for i in range(horizon):
#         offset = pd.to_timedelta(i + 1, unit='h')
#         plt.plot(timestamps + offset, preds[i], label=f'Predicted t+{i+1}', linestyle='-')
    
#     ax = plt.gca()
#     ax.xaxis.set_major_locator(HourLocator(byhour=[0,6,12,18]))
#     ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
    
#     start_datetime = timestamps[0].normalize()
#     if timestamps[0] != start_datetime:
#         ax.set_xlim(left=start_datetime)
    
#     plt.xticks(rotation=45)
#     plt.legend()
#     plt.grid(True)
#     plt.title(f"Actual vs Predicted {parameter.capitalize()}")
#     plt.tight_layout()
#     plt.show()

# def predict_model(csv_file, parameter, start_date, end_date, time_steps=1, forecast_horizon=3):
#     # Load and prepare data
#     df = fetch_data(csv_file, parameter)
#     df = feature_engineering(df, parameter)
    
#     features = [parameter, 'doy', 'minute', 'month', 'year', 'week', 'season', 'hour',
#                 'lag1', 'lag2', 'rolling_mean_3', 'rolling_std_3',
#                 'rolling_mean_6', 'rolling_std_6', 'rolling_mean_12', 'rolling_std_12',
#                 'sin_hour', 'cos_hour', 'sin_month', 'cos_month']
    
#     # Get data for prediction period
#     df_range_start = pd.to_datetime(start_date) - pd.Timedelta(hours=time_steps)
#     df_range_end = pd.to_datetime(end_date) + pd.Timedelta(hours=forecast_horizon)
#     df_subset = df.loc[df_range_start:df_range_end, features].dropna()
    
#     # Load models and scaler
#     scaler = joblib.load(f"{parameter}_robust_scaler.pkl")
#     data_scaled = scaler.transform(df_subset)
    
#     # Create sequences
#     X, y_true = create_sequences(data_scaled, time_steps, forecast_horizon)
#     sequence_start_index = df_subset.index[time_steps:time_steps + len(X)]
    
#     # Filter to requested date range
#     start_datetime = pd.to_datetime(start_date).normalize()
#     mask = (sequence_start_index >= start_datetime) & (sequence_start_index <= pd.to_datetime(end_date))
#     X = X[mask]
#     y_true = y_true[mask]
#     sequence_start_index = sequence_start_index[mask]
    
#     if len(X) == 0:
#         raise ValueError("Not enough data for prediction in the selected date range.")
    
#     # Load models
#     cnn_lstm_model = load_model(f"{parameter}_cnn_lstm_feature_extractor.keras")
#     feature_extractor = Model(inputs=cnn_lstm_model.inputs,
#                             outputs=cnn_lstm_model.layers[-3].output)
    
#     # Get deep features
#     X_deep = feature_extractor.predict(X)
    
#     # Load XGBoost models
#     xgb_models = []
#     for horizon in range(forecast_horizon):
#         xgb = XGBRegressor()
#         xgb.load_model(f"{parameter}_xgb_model_t{horizon+1}.json")
#         xgb_models.append(xgb)
    
#     # Prepare combined features for XGBoost
#     X_combined = np.hstack([
#         X_deep,
#         X.reshape(X.shape[0], -1),
#         np.mean(X, axis=1),
#         np.std(X, axis=1)
#     ])
    
#     # Get XGBoost predictions
#     xgb_preds = np.column_stack([
#         model.predict(X_combined) for model in xgb_models
#     ])
    
#     # Prepare meta-features for Random Forest
#     meta_features = np.column_stack([
#         xgb_preds,
#         X[:, -1, :],  # Latest features
#         X[:, -1, 4:8]  # Temporal features (month, year, week, season)
#     ])
    
#     # Load and use Random Forest
#     rf = joblib.load(f"{parameter}_random_forest_meta_model.pkl")
#     final_preds = rf.predict(meta_features)
    
#     # Evaluate and plot
#     predictions, actuals = evaluate_predictions(final_preds, y_true, scaler, data_scaled, forecast_horizon)
#     plot_predictions(sequence_start_index, actuals, predictions, forecast_horizon, parameter)
    
#     # Prepare output
#     output_df = pd.DataFrame({
#         "datetime": sequence_start_index,
#         "actual": actuals[0],
#         "pred_t+1": predictions[0],
#         "pred_t+2": predictions[1],
#         "pred_t+3": predictions[2],
#     })
    
#     print("Final data: ", output_df)
#     output_file = f"{parameter}_predictions.csv"
#     output_df.to_csv(output_file, index=False)
#     print(f"✅ Predictions saved to '{output_file}'")

# # --- UI Setup ---
# param_dropdown = widgets.Dropdown(
#     options=['rainfall', 'precipitation'],
#     value='rainfall',
#     description='Parameter:',
#     style={'description_width': 'initial'}
# )

# start_date_picker = widgets.DatePicker(description='Start Date:')
# end_date_picker = widgets.DatePicker(description='End Date:')
# submit_button = widgets.Button(description='Submit', button_style='success')
# output = widgets.Output()

# def on_submit_clicked(b):
#     with output:
#         clear_output()
#         try:
#             start_date = start_date_picker.value
#             end_date = end_date_picker.value
#             parameter = param_dropdown.value

#             if not start_date or not end_date:
#                 print("❌ Please select both start and end dates.")
#                 return

#             if start_date > end_date:
#                 print("❌ Start date must be before end date.")
#                 return

#             predict_model("TestDataset.csv", parameter, start_date, end_date)

#         except Exception as e:
#             print(f"❌ Error: {e}")

# submit_button.on_click(on_submit_clicked)

# ui = widgets.VBox([
#     param_dropdown,
#     start_date_picker,
#     end_date_picker,
#     submit_button,
#     output
# ])

# display(ui)