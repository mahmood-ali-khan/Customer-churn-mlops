import streamlit as st
import joblib as jb
import numpy as np
import requests
#Page Config setting
st.set_page_config(page_title="Churn Predictor",page_icon='📊',layout='wide')

st.title('Customer Churn Predictor')
st.markdown('Enter customer details to predict churn probability')

#Input Form
st.subheader('Customer Information')

col1,col2=st.columns(2)
with col1:
    tenure_months=st.slider('Tenure in Months',0,72,12)
    monthly_charges=st.slider('Monthly Charges ($)',18.0, 120.0, 65.0)
    cltv = st.slider('Customer Lifetime Value', 2000, 6500, 4000)
    contract=st.selectbox('Contract Type',['Month-to-month','One year','Two years'])
    internet_service = st.selectbox('Internet Service', 
                                    ['Fiber optic', 'DSL', 'No'])
    payment_method = st.selectbox('Payment Method', 
                                ['Electronic check', 'Mailed check',
                                'Bank transfer (automatic)', 
                                'Credit card (automatic)'])
with col2:
    gender = st.radio('Gender', ['Male', 'Female'])
    senior_citizen = st.radio('Senior Citizen', ['Yes', 'No'])
    partner = st.radio('Partner', ['Yes', 'No'])
    dependents = st.radio('Dependents', ['Yes', 'No'])
    phone_service = st.radio('Phone Service', ['Yes', 'No'])
    paperless_billing = st.radio('Paperless Billing', ['Yes', 'No'])

# Second row of inputs
st.subheader('Services')
col3, col4 = st.columns(2)

with col3:
    multiple_lines = st.selectbox('Multiple Lines', 
                                  ['Yes', 'No', 'No phone service'])
    online_security = st.selectbox('Online Security', 
                                   ['Yes', 'No', 'No internet service'])
    online_backup = st.selectbox('Online Backup', 
                                 ['Yes', 'No', 'No internet service'])
    device_protection = st.selectbox('Device Protection', 
                                     ['Yes', 'No', 'No internet service'])

with col4:
    tech_support = st.selectbox('Tech Support', 
                                ['Yes', 'No', 'No internet service'])
    streaming_tv = st.selectbox('Streaming TV', 
                                ['Yes', 'No', 'No internet service'])
    streaming_movies = st.selectbox('Streaming Movies', 
                                    ['Yes', 'No', 'No internet service'])

st.divider()
if st.button('Predict',type='primary',use_container_width=True):
    payload= {
        'tenure_months': tenure_months,
        'monthly_charges': monthly_charges,
        'cltv': cltv,
        'contract': contract,
        'internet_service': internet_service,
        'payment_method': payment_method,
        'gender': gender,
        'senior_citizen': senior_citizen,
        'partner': partner,
        'dependents': dependents,
        'phone_service': phone_service,
        'paperless_billing': paperless_billing,
        'multiple_lines': multiple_lines,
        'online_security': online_security,
        'online_backup': online_backup,
        'device_protection': device_protection,
        'tech_support': tech_support,
        'streaming_tv': streaming_tv,
        'streaming_movies': streaming_movies
        }
    try:
        with st.spinner('Predicting...'):
            response=requests.post('http://localhost:8000/predict',json=payload,timeout=10)
            result=response.json()


 # ── Section 4 — Result display ───────────────────────
        st.subheader('Prediction Result')
        
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            st.metric(
                'Churn Probability',
                f"{result['churn_probability'] * 100:.1f}%"
            )
        
        with col_r2:
            prediction_label = 'Will Churn' if result['churn_prediction'] == 1 else 'Will Not Churn'
            st.metric('Prediction', prediction_label)
        
        with col_r3:
            st.metric('Risk Level', result['risk_level'])
        
        # Probability bar
        st.progress(result['churn_probability'])
        
        # Risk level message
        if result['risk_level'] == 'High Risk':
            st.error('⚠️ High churn risk — immediate retention action recommended')
        elif result['risk_level'] == 'Medium Risk':
            st.warning('⚡ Medium churn risk — consider targeted retention offer')
        else:
            st.success('✅ Low churn risk — customer is likely to stay')
            
    except requests.exceptions.ConnectionError:
        st.error('Cannot connect to API. Make sure the FastAPI server is running on port 8000.')
    except Exception as e:
        st.error(f'Error: {str(e)}')