# !pip install tensorflow
# !pip install xgboost
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from sklearn.preprocessing import RobustScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Flatten, concatenate, Attention, Concatenate
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping
# import warnings
# import joblib
# import os

# warnings.filterwarnings("ignore")

# # --- Data Loading ---
# def fetch_data_from_csv(csv_file):
#     df = pd.read_csv(csv_file, parse_dates=['datetime'])
#     df.set_index('datetime', inplace=True)
#     df['rainfall'] = np.log1p(df['rainfall'])  # log transform
#     return df

# def get_season(month):
#     if month in [2, 3, 4, 5]:
#         return 0  # summer
#     elif month in [6, 7, 8, 9]:
#         return 1  # rainy
#     else:
#         return 2  # winter

# # --- Feature Engineering ---
# def feature_engineering(df):
#     df['minute'] = df.index.minute
#     df['hour'] = df.index.hour
#     df['month'] = df.index.month
#     df['year'] = df.index.year
#     df['week'] = df.index.isocalendar().week.astype(int)
#     df['doy'] = df.index.dayofyear
#     df['season'] = df['month'].apply(get_season)
#     df['lag1'] = df['rainfall'].shift(1)
#     df['lag2'] = df['rainfall'].shift(2)
#     for window in [3, 6, 12]:
#         df[f'rolling_mean_{window}'] = df['rainfall'].rolling(window).mean()
#         df[f'rolling_std_{window}'] = df['rainfall'].rolling(window).std()
#     df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
#     df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
#     df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
#     df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

#     df.bfill(inplace=True)
#     return df

# # --- Sequence Creation ---
# def create_sequences(data, time_steps, forecast_horizon):
#     X, y = [], []
#     for i in range(len(data) - time_steps - forecast_horizon + 1):
#         X.append(data[i:i + time_steps])
#         y.append(data[i + time_steps:i + time_steps + forecast_horizon, 0])  # rainfall
#     return np.array(X), np.array(y)

# # --- Training and Saving ---
# def train_hybrid_model(df, time_steps=12, forecast_horizon=3):
#     features = ['rainfall', 'doy', 'minute', 'month', 'year', 'week', 'season', 'hour',
#                 'lag1', 'lag2', 'rolling_mean_3', 'rolling_std_3',
#                 'rolling_mean_6', 'rolling_std_6', 'rolling_mean_12', 'rolling_std_12',
#                 'sin_hour', 'cos_hour', 'sin_month', 'cos_month']

#     train = df.loc['2000':'2010', features].dropna()
#     test = df.loc['2010':'2011', features].dropna()

#     scaler = RobustScaler()
#     scaler.fit(train)
#     train_scaled = scaler.transform(train)
#     test_scaled = scaler.transform(test)

#     X_train, y_train = create_sequences(train_scaled, time_steps, forecast_horizon)
#     X_test, y_test = create_sequences(test_scaled, time_steps, forecast_horizon)

#     #CNN-LSTM
#     def create_cnn_lstm_model(input_shape):
#       inputs = Input(shape=input_shape)

#     # CNN Branch
#       cnn = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(inputs)

#       if input_shape[0] >= 2:
#         cnn = MaxPooling1D(pool_size=2)(cnn)

#       cnn = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(cnn)

#       if input_shape[0] >= 4:  # only pool again if sequence length is big enough
#         cnn = MaxPooling1D(pool_size=2)(cnn)

#       cnn = Flatten()(cnn)

#     # LSTM Branch
#       lstm = LSTM(256, return_sequences=True)(inputs)
#       lstm = LSTM(128, return_sequences=True)(lstm)
#       lstm = LSTM(64, return_sequences=True)(lstm)

#     # Attention block (self-attention)
#       attention = Attention()([lstm, lstm])
#       lstm = Concatenate()([lstm, attention])

#     # Continue with two more LSTM layers
#       lstm = LSTM(64, return_sequences=True)(lstm)
#       lstm = LSTM(32)(lstm)

#     # Combine
#       combined = concatenate([cnn, lstm])
#       combined = Dense(64, activation='relu')(combined)
#       combined = tf.keras.layers.Dropout(0.2)(combined)
#       outputs = Dense(input_shape[0])(combined)

#       model = Model(inputs=inputs, outputs=outputs)
#       model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
#       return model


#     # Train CNN-LSTM
#     cnn_lstm_model = create_cnn_lstm_model((time_steps, len(features)))
#     early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

#     history = cnn_lstm_model.fit(
#         X_train, y_train,
#         validation_split=0.2,
#         epochs=30,  # Increased epochs for better convergence
#         batch_size=64,  # Larger batch size
#         callbacks=[early_stop],
#         verbose=1
#     )

#     # Get deep features from CNN-LSTM
#     feature_extractor = Model(inputs=cnn_lstm_model.inputs,
#                             outputs=cnn_lstm_model.layers[-3].output)
#     X_train_deep = feature_extractor.predict(X_train)
#     X_test_deep = feature_extractor.predict(X_test)

#     #Using XGBoost
#     xgb_models = []
#     for horizon in range(forecast_horizon):
#         xgb = XGBRegressor(
#             n_estimators=400,
#             learning_rate=0.08,
#             max_depth=6,
#             subsample=0.8,
#             colsample_bytree=0.7,
#             objective='reg:squarederror'
#         )

#         X_train_combined = np.hstack([
#             X_train_deep,
#             X_train.reshape(X_train.shape[0], -1),  # Original features
#             np.mean(X_train, axis=1),  # Additional temporal aggregation
#             np.std(X_train, axis=1)
#         ])

#         xgb.fit(X_train_combined, y_train[:, horizon])
#         xgb_models.append(xgb)

#     X_test_combined = np.hstack([
#         X_test_deep,
#         X_test.reshape(X_test.shape[0], -1),  # Original features
#         np.mean(X_test, axis=1),
#         np.std(X_test, axis=1)
#     ])

#     xgb_preds = np.column_stack([
#         model.predict(X_test_combined) for model in xgb_models
#     ])


#     #Using Random Forest
#     meta_features = np.column_stack([
#         xgb_preds,
#         X_test[:, -1, :],  # Latest features
#         X_test[:, -1, 4:8]  # Temporal features (month, year, week, season)
#     ])

#     rf = RandomForestRegressor(
#         n_estimators= 300,
#         max_depth = 15,
#         min_samples_split = 5,
#         min_samples_leaf = 2,
#         random_state=42
#     )
#     rf.fit(meta_features, y_test)
#     final_preds = rf.predict(meta_features)

#     # --- Save models and scaler ---
#     cnn_lstm_model.save("cnn_lstm_feature_extractor.h5")
#     for i, model in enumerate(xgb_models):
#         model.save_model(f"xgb_model_t{i+1}.json")
#     joblib.dump(rf, "random_forest_meta_model.pkl")
#     joblib.dump(scaler, "robust_scaler.pkl")

#     return final_preds, y_test, scaler, test_scaled, time_steps, forecast_horizon

# # --- Evaluation ---
# def evaluate_predictions(preds, y_true, scaler, test_scaled, time_steps, forecast_horizon):
#     mae_list, rmse_list, r2_list = [], [], []
#     actuals, predictions = [], []

#     for i in range(forecast_horizon):
#         dummy_input = np.zeros_like(test_scaled[:len(preds)])
#         dummy_input[:, 0] = preds[:, i] if len(preds.shape) > 1 else preds
#         dummy_actual = np.zeros_like(dummy_input)
#         dummy_actual[:, 0] = y_true[:, i]

#         inv_pred = np.expm1(scaler.inverse_transform(dummy_input)[:, 0])
#         inv_actual = np.expm1(scaler.inverse_transform(dummy_actual)[:, 0])

#         mae = mean_absolute_error(inv_actual, inv_pred)
#         rmse = np.sqrt(mean_squared_error(inv_actual, inv_pred))
#         r2 = r2_score(inv_actual, inv_pred)

#         mae_list.append(mae)
#         rmse_list.append(rmse)
#         r2_list.append(r2)

#         predictions.append(inv_pred)
#         actuals.append(inv_actual)

#         print(f"t+{i+1}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

#     return predictions, actuals

# # --- Plotting ---
# def plot_predictions(timestamps, actuals, preds, horizon):
#     plt.figure(figsize=(14, 6))
#     plt.plot(timestamps, actuals[0], label='Actual', marker='o')
#     for i in range(horizon):
#         offset = pd.to_timedelta(i + 1, unit='h')
#         plt.plot(timestamps + offset, preds[i], label=f'Predicted t+{i+1}', linestyle='--')
#     plt.legend()
#     plt.grid(True)
#     plt.title("Actual vs Predicted Rainfall")
#     plt.show()



# # --- Main ---
# def main():
#     csv_file = '/content/TestDataset.csv'  # Change path if needed
#     df = fetch_data_from_csv(csv_file)
#     df = feature_engineering(df)

#     time_steps = 1
#     forecast_horizon = 3

#     final_preds, actuals, scaler, test_scaled, ts, fh = train_hybrid_model(df, time_steps, forecast_horizon)
#     predictions, true_vals = evaluate_predictions(final_preds, actuals, scaler, test_scaled, ts, fh)

#     timestamps = df.loc['2010':'2011'].index[ts:ts + len(predictions[0])]
#     plot_predictions(timestamps, true_vals, predictions, forecast_horizon)
#     output_df = pd.DataFrame({
#     "datetime": timestamps,
#     "actual": true_vals[0],              # Actual t+1 values
#     "pred_t+1": predictions[0],          # Predicted t+1
#     "pred_t+2": predictions[1],          # Predicted t+2
#     "pred_t+3": predictions[2],          # Predicted t+3
#     })
#     output_df.to_csv("rainfall_predictions.csv", index=False)
#     print("Saved predictions to rainfall_predictions.csv")


# if __name__ == "__main__":
#     main()
