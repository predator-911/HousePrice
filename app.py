import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Function to load and check model and scaler
def load_objects():
    model = None
    scaler = None
    train_columns = None
    try:
        model = joblib.load('random_forest_model.h5')
        scaler = joblib.load('scaler.pkl')
        train_columns = joblib.load('train_columns.pkl')
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
    except Exception as e:
        st.error(f"Error loading objects: {e}")
    return model, scaler, train_columns

# Load the trained model, scaler, and column names
model, scaler, train_columns = load_objects()

if model is not None and scaler is not None and train_columns is not None:
    # Streamlit app UI
    st.title('House Price Prediction App')
    st.write('This app predicts the sale prices of houses using a pre-trained RandomForest model.')

    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file with the test data to make predictions.", type=["csv"])

    if uploaded_file is not None:
        try:
            # Load the uploaded file
            test = pd.read_csv(uploaded_file)
            st.write("Uploaded file:", test.head())

            # Preprocess the test data similar to the training data
            for col in test.columns:
                if test[col].dtype == 'object':
                    test[col].fillna(test[col].mode()[0], inplace=True)
                else:
                    test[col].fillna(test[col].median(), inplace=True)

            # Encode categorical variables
            test = pd.get_dummies(test)

            # Align the test data with the training data columns
            test = test.reindex(columns=train_columns, fill_value=0)

            # Scale the features
            test_scaled = scaler.transform(test)

            # Make predictions
            predictions = model.predict(test_scaled)

            # Prepare results
            results = pd.DataFrame({
                'Id': test.index,  # Assuming 'Id' column exists or use index
                'SalePrice': predictions
            })

            # Display results
            st.write('Predictions:')
            st.dataframe(results)

            # Option to download the predictions as a CSV file
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions as CSV", csv, "house_price_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error processing the file: {e}")
else:
    st.error("Model, scaler, or training columns could not be loaded. Check file paths and compatibility.")

st.write("Make sure the model and scaler files are correctly uploaded and compatible with the scikit-learn version.")
