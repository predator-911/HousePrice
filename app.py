import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model and other objects
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

# Streamlit app UI
st.title('House Price Prediction App')
st.write('This app predicts the sale prices of houses using a pre-trained RandomForest model.')

# Load model and objects
model, scaler, train_columns = load_objects()

if model and scaler and train_columns:
    uploaded_file = st.file_uploader("Upload a CSV file with the test data to make predictions.", type=["csv"])

    if uploaded_file is not None:
        try:
            # Load and preprocess the uploaded file
            test = pd.read_csv(uploaded_file)
            for col in test.columns:
                if test[col].dtype == 'object':
                    test[col].fillna(test[col].mode()[0], inplace=True)
                else:
                    test[col].fillna(test[col].median(), inplace=True)
            test = pd.get_dummies(test)
            test = test.reindex(columns=train_columns, fill_value=0)

            # Scale features
            test_scaled = scaler.transform(test)

            # Make predictions
            predictions = model.predict(test_scaled)

            # Prepare results
            results = pd.DataFrame({
                'Id': test.index,  # Or use 'Id' column if available
                'SalePrice': predictions
            })

            # Display results
            st.write('Predictions:')
            st.dataframe(results)

            # Option to download results
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions as CSV", csv, "house_price_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error processing the file: {e}")

else:
    st.error("Model, scaler, or training columns could not be loaded. Check file paths and compatibility.")
