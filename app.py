import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Function to load and check model, scaler, and column names
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

    # Option for user input
    st.subheader('Enter the details of the house:')
    
    # Assuming you have features like 'LotArea', 'YearBuilt', etc.
    features = {
        'LotArea': st.number_input('Lot Area', value=0),
        'YearBuilt': st.number_input('Year Built', value=0),
        'OverallQual': st.slider('Overall Quality', min_value=1, max_value=10, value=5),
        'TotalBsmtSF': st.number_input('Total Basement Area', value=0),
        'GrLivArea': st.number_input('Above Grade Living Area', value=0),
        'GarageCars': st.slider('Number of Garage Cars', min_value=0, max_value=4, value=0),
        'GarageArea': st.number_input('Garage Area', value=0),
        # Add more features here based on your model
    }

    # Button to make prediction from user input
    if st.button('Predict Price'):
        try:
            # Create a DataFrame from user input
            input_data = pd.DataFrame([features])

            # Encode categorical variables
            input_data = pd.get_dummies(input_data)

            # Align the input data with the training data columns
            input_data = input_data.reindex(columns=train_columns, fill_value=0)

            # Scale the features
            input_data_scaled = scaler.transform(input_data)

            # Make prediction
            prediction = model.predict(input_data_scaled)[0]
            st.write(f'Predicted Sale Price: ${prediction:,.2f}')
        except Exception as e:
            st.error(f"Error making prediction: {e}")

    # Option to upload a CSV file
    st.subheader('Or upload a CSV file with the test data:')
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Load the uploaded file
            test = pd.read_csv(uploaded_file)

            # Preprocess the test data
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
