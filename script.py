import app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import InputLayer, Dense


from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score

# ----------------- DATA PREPROCESSING --------------
dataset = pd.read_csv("admissions_data.csv")

# print the first five data entries
print(dataset.head())

# display the summary statistics of the data
print(dataset.describe())

# drop the index column from the dataset
dataset = dataset.drop(['Serial No.'], axis=1)

# ----------------- DATA TRAINING --------------
# split data into features and labels parameters
features = dataset.iloc[:, 0:-1]

labels = dataset.iloc[:, -1]

# split data into training and test sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# standardize and derive columns from training data only
numerical_columns = features_train.select_dtypes(include=np.number).columns

ct = ColumnTransformer(
  [("only numeric", StandardScaler(), numerical_columns)], remainder="passthrough"
)

# fit and transform ct to the training data
features_train_scaled = ct.fit_transform(features_train)

# transform test data
features_test_scaled = ct.transform(features_test)


# ----------------- BUILDING THE MODEL --------------
def model_builder():
  my_model = Sequential()
  my_model.add(InputLayer(input_shape = (features_train_scaled.shape[1],)))
  my_model.add(Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
  my_model.add(Dense(32, activation="relu"))
  my_model.add(Dense(1))
  return my_model



# Compare batch sizes
results = {}

for batch_size in [2, 4, 8, 16, 32]:
  my_model = model_builder()
  my_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
  es = EarlyStopping(monitor="val_loss", mode="min", patience=5,  restore_best_weights=True)   #implement early stop

  my_model.fit(
    features_train_scaled, labels_train,
    epochs=40, batch_size=batch_size,
    verbose=0, validation_split=0.2, callbacks=[es]
  )

  # display the final loss and final metric
  res_mse, res_mae = my_model.evaluate(features_test_scaled, labels_test, verbose=0)
  results[batch_size] = {"RMSE": np.sqrt(res_mse), "MAE": res_mae}
  print(f"Batch size {batch_size} → RMSE: {np.sqrt(res_mse):.4f}, MAE: {res_mae:.4f}")


# Compare optimizers and plot the model loss per epoch
best_model = None
best_rmse = float('inf')


fig, axes = plt.subplots(len(["adam", "SGD", "RMSprop"]), 2, figsize=(12, 12))
for i, optimizer in enumerate(["adam", "SGD", "RMSprop"]):
  my_model = model_builder()
  my_model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
  es = EarlyStopping(monitor="val_loss", mode="min", patience=5,  restore_best_weights=True)   #implement early stop

  fit_model = my_model.fit(
    features_train_scaled, labels_train,
    epochs=40, batch_size=8,
    verbose=0, validation_split=0.2, callbacks=[es]
    )

  # display the final loss and final metric
  res_mse, res_mae = my_model.evaluate(features_test_scaled, labels_test, verbose=0)
  rmse = np.sqrt(res_mse)
  print(f"Optimizer {optimizer}  → RMSE: {np.sqrt(res_mse):.4f}, MAE: {res_mae:.4f}")

  # Save the best performing model
  if rmse < best_rmse:
    best_rmse = rmse
    best_model = my_model
    best_optimizer = optimizer


  # MAE subplot
  axes[i, 0].plot(fit_model.history['mae'], label='train', linewidth=2)
  axes[i, 0].plot(fit_model.history['val_mae'], label='validation', linewidth=2, linestyle='--')
  axes[i, 0].set_title(f'{optimizer} - MAE per Epoch')
  axes[i, 0].set_ylabel('MAE')
  axes[i, 0].legend(loc='upper right')
  axes[i, 0].grid(True, alpha=0.3)

  # Loss subplot
  axes[i, 1].plot(fit_model.history['loss'], label='train', linewidth=2)
  axes[i, 1].plot(fit_model.history['val_loss'], label='validation', linewidth=2, linestyle='--')
  axes[i, 1].set_title(f'{optimizer} - Loss per Epoch')
  axes[i, 1].set_ylabel('Loss (MSE)')
  axes[i, 1].legend(loc='upper right')
  axes[i, 1].grid(True, alpha=0.3)


print(f"\nBest optimizer: {best_optimizer} with RMSE: {best_rmse:.4f}")
fig.tight_layout()
fig.savefig('static/images/my_plots.png')

# ----------------- MODEL PREDICTION PERFORMANCE --------------
predicted_values = best_model.predict(features_test_scaled)
print("R² Score:", r2_score(labels_test, predicted_values))

# Example of a single applicant matching the dataset's feature columns
# [GRE, TOEFL, University Rating, SOP, LOR, CGPA, Research]
applicant = np.array([[320, 110, 4, 4.5, 4.0, 8.5, 1]])

# Scale using the already-fitted ColumnTransformer
applicant_df = pd.DataFrame(applicant, columns=features_train.columns)
applicant_scaled = ct.transform(applicant_df)

# Predict
admission_chance = best_model.predict(applicant_scaled)
print(f"Predicted Admission Chance: {admission_chance[0][0]:.2f}")