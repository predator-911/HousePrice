import joblib

model = joblib.load('random_forest_model.h5')
print(model.predict([[0, 0, 5, 0, 0, 0, 0]]))  # Use a dummy input
