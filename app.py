import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained model
model = joblib.load(r'C:\Users\Ranjan kumar pradhan\.vscode\Propensify\best_model.pkl')

# Initialize label encoders for categorical columns
categorical_columns = ['profession', 'marital', 'schooling', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
label_encoders = {col: LabelEncoder() for col in categorical_columns}

# Fit label encoders with the actual categories
label_encoders['profession'].fit(['admin.', 'blue-collar', 'technician', 'services', 'management', 'retired', 'entrepreneur', 'self-employed', 'housemaid', 'unemployed', 'student', 'unknown'])
label_encoders['marital'].fit(['married', 'single', 'divorced', 'unknown'])
label_encoders['schooling'].fit(['university.degree', 'high.school', 'basic.9y', 'professional.course', 'basic.4y', 'basic.6y', 'unknown', 'illiterate'])
label_encoders['default'].fit(['no', 'unknown', 'yes'])
label_encoders['housing'].fit(['yes', 'no', 'unknown'])
label_encoders['loan'].fit(['no', 'yes', 'unknown'])
label_encoders['contact'].fit(['cellular', 'telephone'])
label_encoders['month'].fit(['may', 'jul', 'aug', 'jun', 'nov', 'apr', 'oct', 'sep', 'mar', 'dec'])
label_encoders['day_of_week'].fit(['mon', 'thu', 'tue', 'wed', 'fri'])
label_encoders['poutcome'].fit(['nonexistent', 'failure', 'success'])

# Streamlit app
def main():
    st.title("Propensify: Customer Response Prediction")
    st.write("Predict whether a customer will respond to a marketing campaign.")

    # Input form
    custAge = st.number_input("Customer Age", min_value=18, max_value=100, step=1)
    profession = st.selectbox("Profession", label_encoders['profession'].classes_)
    marital = st.selectbox("Marital Status", label_encoders['marital'].classes_)
    schooling = st.selectbox("Schooling", label_encoders['schooling'].classes_)
    default = st.selectbox("Default", label_encoders['default'].classes_)
    housing = st.selectbox("Housing Loan", label_encoders['housing'].classes_)
    loan = st.selectbox("Personal Loan", label_encoders['loan'].classes_)
    contact = st.selectbox("Contact Type", label_encoders['contact'].classes_)
    month = st.selectbox("Last Contact Month", label_encoders['month'].classes_)
    day_of_week = st.selectbox("Day of Week", label_encoders['day_of_week'].classes_)
    campaign = st.number_input("Number of Contacts", min_value=1, step=1)
    poutcome = st.selectbox("Previous Campaign Outcome", label_encoders['poutcome'].classes_)
    emp_var_rate = st.number_input("Employment Variation Rate")
    cons_price_idx = st.number_input("Consumer Price Index")
    cons_conf_idx = st.number_input("Consumer Confidence Index")
    euribor3m = st.number_input("Euribor 3-Month Rate")
    nr_employed = st.number_input("Number of Employees")

    # Predict button
    if st.button("Predict"):
        # Create DataFrame from input
        input_data = pd.DataFrame({
            'custAge': [custAge],
            'profession': [profession],
            'marital': [marital],
            'schooling': [schooling],
            'default': [default],
            'housing': [housing],
            'loan': [loan],
            'contact': [contact],
            'month': [month],
            'day_of_week': [day_of_week],
            'campaign': [campaign],
            'poutcome': [poutcome],
            'emp.var.rate': [emp_var_rate],
            'cons.price.idx': [cons_price_idx],
            'cons.conf.idx': [cons_conf_idx],
            'euribor3m': [euribor3m],
            'nr.employed': [nr_employed]
        })

        # Encode categorical columns
        for col in categorical_columns:
            input_data[col] = label_encoders[col].transform(input_data[col])

        # Scale the data (assuming the scaler was fitted on the training set)
        scaler = StandardScaler()
        input_data_scaled = scaler.fit_transform(input_data)  # Fit to input data (placeholder)

        # Predict using the loaded model
        prediction = model.predict(input_data_scaled)

        # Display result
        result = "Yes" if prediction[0] == 1 else "No"
        st.write(f"Customer Response Prediction:- {result}")

if __name__ == "__main__":
    main()