import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Function to load and check model, scaler, and columns
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

    # Input fields for individual features
    st.header("Enter details for prediction:")
    features = {
        'MSSubClass': st.number_input("MSSubClass", min_value=1, max_value=200, value=60),
        'LotFrontage': st.number_input("LotFrontage", min_value=0, max_value=200, value=70),
        'LotArea': st.number_input("LotArea", min_value=0, max_value=100000, value=8000),
        'OverallQual': st.number_input("OverallQual", min_value=1, max_value=10, value=5),
        'OverallCond': st.number_input("OverallCond", min_value=1, max_value=10, value=5),
        'YearBuilt': st.number_input("YearBuilt", min_value=1800, max_value=2024, value=2000),
        'YearRemodAdd': st.number_input("YearRemodAdd", min_value=1800, max_value=2024, value=2000),
        'MasVnrArea': st.number_input("MasVnrArea", min_value=0, max_value=1000, value=0),
        'BsmtFinSF1': st.number_input("BsmtFinSF1", min_value=0, max_value=2000, value=0),
        # Add more input fields as needed
    }

    # Optionally, allow users to upload a CSV file
    uploaded_file = st.file_uploader("Or upload a CSV file with the test data to make predictions.", type=["csv"])

    if uploaded_file is not None:
        try:
            # Load the uploaded file
            test = pd.read_csv(uploaded_file)

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

    # Show user input data prediction
    if st.button('Predict'):
        try:
            # Convert user inputs to a DataFrame
            input_data = pd.DataFrame([features])
            input_data = pd.get_dummies(input_data)
            input_data = input_data.reindex(columns=train_columns, fill_value=0)
            input_scaled = scaler.transform(input_data)
            
            # Predict and display results
            prediction = model.predict(input_scaled)
            st.write(f'Predicted House Price: ${prediction[0]:,.2f}')
        except Exception as e:
            st.error(f"Error making prediction: {e}")

else:
    st.error("Model, scaler, or training columns could not be loaded. Check file paths and compatibility.")
