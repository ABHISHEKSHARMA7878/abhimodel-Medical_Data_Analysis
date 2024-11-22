import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set page configuration
st.set_page_config(page_title='Medical Data Analysis', page_icon='🩺', layout='wide')

# Title and description
st.title('Medical Data Analysis and Prediction')
st.write('Interactive machine learning application for medical data insights')

# Sidebar for navigation
st.sidebar.title('Navigation')
app_mode = st.sidebar.selectbox('Choose a section', 
    ['Home', 'Data Overview', 'Data Visualization', 'Model Training', 'Prediction'])

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("project-data.csv", sep=';')
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please upload a CSV file.")
        return None

# Handle missing values in data
def handle_missing_values(df):
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

# Encode target variable
def encode_target(df, target_col):
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])
    return df, le

# Home page
if app_mode == 'Home':
    st.header('Welcome to Medical Data Analysis App')
    st.write('''
    This application provides:
    - Comprehensive data exploration
    - Interactive visualizations
    - Machine learning model training
    - Predictive modeling
    ''')
    
    if st.button('Load Sample Data'):
        df = load_data()
        if df is not None:
            st.write(df.head())

# Data Overview
elif app_mode == 'Data Overview':
    st.header('Data Overview')
    df = load_data()
    
    if df is not None:
        df = handle_missing_values(df)
        
        # Basic information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Dataset Information')
            st.write(f'Number of rows: {df.shape[0]}')
            st.write(f'Number of columns: {df.shape[1]}')
            st.write('Columns:', list(df.columns))
        
        with col2:
            st.subheader('Column Types')
            st.write(df.dtypes)
        
        # Missing values
        st.subheader('Missing Values')
        st.write(df.isnull().sum())

# Data Visualization
elif app_mode == 'Data Visualization':
    st.header('Data Visualization')
    df = load_data()
    
    if df is not None:
        df = handle_missing_values(df)
        
        # Visualization options
        viz_type = st.selectbox('Choose Visualization', 
            ['Histogram', 'Boxplot', 'Correlation Heatmap', 'Distribution by Sex'])
        
        if viz_type == 'Histogram':
            col = st.selectbox('Select Column', df.select_dtypes(include=['float64', 'int64']).columns)
            fig, ax = plt.subplots()
            df[col].hist(bins=30, ax=ax)
            st.pyplot(fig)
        
        elif viz_type == 'Boxplot':
            col = st.selectbox('Select Column', df.select_dtypes(include=['float64', 'int64']).columns)
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            st.pyplot(fig)
        
        elif viz_type == 'Correlation Heatmap':
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            corr_matrix = df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        
        elif viz_type == 'Distribution by Sex':
            col = st.selectbox('Select Column', df.select_dtypes(include=['float64', 'int64']).columns)
            fig, ax = plt.subplots()
            sns.boxplot(x='sex', y=col, data=df, ax=ax)
            st.pyplot(fig)

# Model Training
elif app_mode == 'Model Training':
    st.header('Model Training')
    df = load_data()
    
    if df is not None:
        df = handle_missing_values(df)
        df, le = encode_target(df, target_col='category')
        
        # Preprocessing
        X = df.drop(columns=['age', 'category'])
        y = df['category']
        
        # Model selection
        model_name = st.selectbox('Select Model', 
            ['Logistic Regression', 'Random Forest', 'XGBoost', 'SVM'])
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Preprocessing
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), numeric_cols),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_cols)
            ])
        
        # Model selection
        if model_name == 'Logistic Regression':
            model = LogisticRegression()
        elif model_name == 'Random Forest':
            model = RandomForestClassifier(random_state=42)
        elif model_name == 'XGBoost':
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        elif model_name == 'SVM':
            model = SVC(probability=True)
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Train model
        if st.button('Train Model'):
            pipeline.fit(X_train, y_train)
            
            # Predictions
            y_pred = pipeline.predict(X_test)
            
            # Metrics
            st.subheader('Model Performance')
            st.write('Accuracy:', accuracy_score(y_test, y_pred))
            
            # Classification Report
            st.subheader('Classification Report')
            st.text(classification_report(y_test, y_pred))
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            st.pyplot(fig)

# Prediction
elif app_mode == 'Prediction':
    st.header('Make Predictions')
    df = load_data()
    
    if df is not None:
        df = handle_missing_values(df)
        df, le = encode_target(df, target_col='category')
        
        # Preprocessing
        X = df.drop(columns=['age', 'category'])
        y = df['category']
        
        # Preprocessor
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = X.select_dtypes(include=['object', 'bool']).columns
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), numeric_cols),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_cols)
            ])
        
        # Train final model
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        full_pipeline.fit(X, y)
        
        # Input features
        st.subheader('Enter Patient Data')
        
        # Dynamic feature inputs
        input_data = {}
        for col in X.columns:
            if col in numeric_cols:
                input_data[col] = st.number_input(f'Enter {col}', 
                    value=float(X[col].mean()), 
                    step=0.1)
            else:
                input_data[col] = st.selectbox(f'Select {col}', 
                    options=X[col].unique())
        
        # Prediction
        if st.button('Predict'):
            input_df = pd.DataFrame([input_data])
            input_transformed = full_pipeline.predict(input_df)
            st.success(f'Predicted Category: {le.inverse_transform(input_transformed)[0]}')
