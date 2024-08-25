import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('random_forest_model.h5')

print(model.predict([[0, 0, 5, 0, 0, 0, 0]]))  # Use a dummy input
