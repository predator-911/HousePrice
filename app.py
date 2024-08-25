import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Function to load model, scaler, and column names
def load_objects():
    model, scaler, train_columns = None, None, None
    try:
        model = joblib.load('random_forest_model.pkl')
        scaler = joblib.load('scaler.pkl')
        train_columns = joblib.load('train_columns.pkl')
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
    except Exception as e:
        st.error(f"Error loading objects: {e}")
    return model, scaler, train_columns

# Load the model, scaler, and column names
model, scaler, train_columns = load_objects()

if model and scaler and train_columns:
    st.title('House Price Prediction App')
    st.write('This app predicts the sale prices of houses using a pre-trained RandomForest model.')

    st.subheader('Enter the details of the house:')
    
    features = {
        'LotArea': st.number_input('Lot Area', value=0),
        'YearBuilt': st.number_input('Year Built', value=0),
        'OverallQual': st.slider('Overall Quality', min_value=1, max_value=10, value=5),
        'TotalBsmtSF': st.number_input('Total Basement Area', value=0),
        'GrLivArea': st.number_input('Above Grade Living Area', value=0),
        'GarageCars': st.slider('Number of Garage Cars', min_value=0, max_value=4, value=0),
        'GarageArea': st.number_input('Garage Area', value=0),
    }

    if st.button('Predict Price'):
        try:
            input_data = pd.DataFrame([features])
            input_data = pd.get_dummies(input_data)
            input_data = input_data.reindex(columns=train_columns, fill_value=0)
            input_data_scaled = scaler.transform(input_data)
            prediction = model.predict(input_data_scaled)[0]
            st.write(f'Predicted Sale Price: ${prediction:,.2f}')
        except Exception as e:
            st.error(f"Error making prediction: {e}")

    st.subheader('Or upload a CSV file with the test data:')
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        try:
            test = pd.read_csv(uploaded_file)
            for col in test.columns:
                if test[col].dtype == 'object':
                    test[col].fillna(test[col].mode()[0], inplace=True)
                else:
                    test[col].fillna(test[col].median(), inplace=True)
            test = pd.get_dummies(test)
            test = test.reindex(columns=train_columns, fill_value=0)
            test_scaled = scaler.transform(test)
            predictions = model.predict(test_scaled)
            results = pd.DataFrame({
                'Id': test.index,
                'SalePrice': predictions
            })
            st.write('Predictions:')
            st.dataframe(results)
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions as CSV", csv, "house_price_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing the file: {e}")
else:
    st.error("Model, scaler, or training columns could not be loaded. Check file paths and compatibility.")
