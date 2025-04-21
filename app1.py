import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Website Traffic Prediction App",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("Website Traffic Prediction App")
st.markdown("""
This app predicts the number of page views a website will receive based on session data.
Upload your website analytics data or use the sample data to make predictions.
""")

# Function to load sample data
@st.cache_data
def load_sample_data():
    data = {
        'Page Views': [5, 4, 4, 5, 5],
        'Session Duration': [11.05, 3.43, 1.62, 3.63, 4.24],
        'Bounce Rate': [0.23, 0.39, 0.40, 0.18, 0.29],
        'Traffic Source': ['Organic', 'Social', 'Organic', 'Organic', 'Paid'],
        'Time on Page': [3.89, 8.48, 9.64, 2.07, 1.96],
        'Previous Visits': [3, 0, 2, 3, 5],
        'Conversion Rate': [1.0, 1.0, 1.0, 1.0, 1.0]
    }
    return pd.DataFrame(data)

# Function to preprocess data
def preprocess_data(df):
    # Convert Traffic Source to categorical
    df['Traffic Source'] = df['Traffic Source'].astype('category')
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['Traffic Source'], drop_first=True)
    
    return df_encoded

# Function to split data into features and target
def split_data(df, target_col='Page Views'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

# Function to train the model
def train_model(X, y):
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, mse, r2, X_train, X_test, y_train, y_test

# Function to make predictions
def predict(model, scaler, input_data):
    # Scale input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    return prediction

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Model Training", "Prediction"])

# Initialize session state for data storage
if 'data' not in st.session_state:
    st.session_state.data = load_sample_data()
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

# Home page
if page == "Home":
    st.header("Welcome to the Website Traffic Prediction App")
    
    st.subheader("Upload Your Data")
    uploaded_file = st.file_uploader("Upload a CSV file with your website analytics data", type=["csv"])
    
    if uploaded_file is not None:
        try:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success("Data uploaded successfully!")
            st.dataframe(st.session_state.data.head())
        except Exception as e:
            st.error(f"Error: {e}")
    
    st.subheader("Or Use Sample Data")
    if st.button("Load Sample Data"):
        st.session_state.data = load_sample_data()
        st.success("Sample data loaded!")
        st.dataframe(st.session_state.data)
    
    st.markdown("""
    ### Instructions:
    1. Upload your website analytics data or use the sample data
    2. Navigate to the Data Analysis tab to explore your data
    3. Train the model in the Model Training tab
    4. Make predictions in the Prediction tab
    """)

# Data Analysis page
elif page == "Data Analysis":
    st.header("Data Analysis")
    
    if st.session_state.data is not None:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.data.head())
        
        st.subheader("Data Statistics")
        st.dataframe(st.session_state.data.describe())
        
        st.subheader("Data Visualization")
        
        # Select columns for visualization
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X-axis", st.session_state.data.columns)
        with col2:
            y_axis = st.selectbox("Y-axis", st.session_state.data.columns, index=0)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if st.session_state.data[x_axis].dtype == 'object' or st.session_state.data[x_axis].dtype.name == 'category':
            sns.barplot(x=x_axis, y=y_axis, data=st.session_state.data, ax=ax)
        else:
            sns.scatterplot(x=x_axis, y=y_axis, data=st.session_state.data, ax=ax)
            
        plt.title(f"{y_axis} vs {x_axis}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        st.subheader("Correlation Matrix")
        numeric_cols = st.session_state.data.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            correlation_matrix = st.session_state.data[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Not enough numerical columns for correlation analysis.")
    else:
        st.warning("No data available. Please upload data or load sample data from the Home page.")

# Model Training page
elif page == "Model Training":
    st.header("Model Training")
    
    if st.session_state.data is not None:
        st.subheader("Select Target Variable")
        target_column = st.selectbox("Select the column to predict", st.session_state.data.columns, index=0)
        
        st.subheader("Select Features")
        feature_cols = st.multiselect("Select features for training", 
                                     [col for col in st.session_state.data.columns if col != target_column],
                                     default=[col for col in st.session_state.data.columns if col != target_column])
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                try:
                    # Filter data to include only selected columns
                    df_selected = st.session_state.data[[target_column] + feature_cols]
                    
                    # Preprocess data
                    df_processed = preprocess_data(df_selected)
                    
                    # Split data
                    X, y = split_data(df_processed, target_col=target_column)
                    
                    # Train model
                    model, scaler, mse, r2, X_train, X_test, y_train, y_test = train_model(X, y)
                    
                    # Store model and metrics in session state
                    st.session_state.model = model
                    st.session_state.scaler = scaler
                    st.session_state.metrics = {'mse': mse, 'r2': r2}
                    st.session_state.X_columns = X.columns.tolist()
                    
                    # Display metrics
                    st.success("Model trained successfully!")
                    st.metric("Mean Squared Error", f"{mse:.4f}")
                    st.metric("R² Score", f"{r2:.4f}")
                    
                    # Feature importance
                    importance = model.feature_importances_
                    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
                    feature_importance = feature_importance.sort_values('Importance', ascending=False)
                    
                    st.subheader("Feature Importance")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
                    plt.title("Feature Importance")
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Predictions vs Actual
                    y_pred = model.predict(scaler.transform(X))
                    pred_df = pd.DataFrame({'Actual': y, 'Predicted': y_pred})
                    
                    st.subheader("Predictions vs Actual")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(x='Actual', y='Predicted', data=pred_df, ax=ax)
                    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
                    plt.xlabel("Actual")
                    plt.ylabel("Predicted")
                    plt.title("Predictions vs Actual")
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error during model training: {e}")
    else:
        st.warning("No data available. Please upload data or load sample data from the Home page.")

# Prediction page
elif page == "Prediction":
    st.header("Traffic Prediction")
    
    if st.session_state.model is None:
        st.warning("No trained model available. Please train a model first.")
    else:
        st.subheader("Enter Values for Prediction")
        
        # Get the required input features based on the trained model
        input_features = {}
        
        # Create input widgets for each feature
        for col in st.session_state.X_columns:
            if 'Traffic Source_' in col:
                # Skip one-hot encoded columns as we'll handle them differently
                continue
                
        # Create a dropdown for Traffic Source
        traffic_sources = ['Organic', 'Social', 'Paid']
        selected_source = st.selectbox("Traffic Source", traffic_sources)
        
        # Create numerical inputs
        col1, col2 = st.columns(2)
        with col1:
            session_duration = st.number_input("Session Duration", min_value=0.0, value=3.0, step=0.1)
            bounce_rate = st.number_input("Bounce Rate", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
            time_on_page = st.number_input("Time on Page", min_value=0.0, value=4.0, step=0.1)
        
        with col2:
            previous_visits = st.number_input("Previous Visits", min_value=0, value=2, step=1)
            conversion_rate = st.number_input("Conversion Rate", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
        
        if st.button("Predict"):
            try:
                # Prepare input data
                input_data = {
                    'Session Duration': session_duration,
                    'Bounce Rate': bounce_rate,
                    'Time on Page': time_on_page,
                    'Previous Visits': previous_visits,
                    'Conversion Rate': conversion_rate,
                    'Traffic Source_Organic': 1 if selected_source == 'Organic' else 0,
                    'Traffic Source_Paid': 1 if selected_source == 'Paid' else 0,
                    'Traffic Source_Social': 1 if selected_source == 'Social' else 0
                }
                
                # Filter out columns not in the model
                input_df = pd.DataFrame([input_data])
                input_df = input_df[[col for col in input_df.columns if col in st.session_state.X_columns]]
                
                # Make prediction
                prediction = predict(st.session_state.model, st.session_state.scaler, input_df)
                
                # Display prediction
                st.success(f"Predicted Page Views: {prediction[0]:.2f}")
                
                # Create a gauge chart for the prediction
                fig, ax = plt.subplots(figsize=(8, 4))
                
                # Determine color based on prediction value
                max_value = 10  # Assuming max page views is 10
                color = 'green' if prediction[0] > max_value * 0.7 else 'orange' if prediction[0] > max_value * 0.4 else 'red'
                
                # Create a simple horizontal bar chart as a gauge
                ax.barh([0], [prediction[0]], color=color)
                ax.barh([0], [max_value], color='lightgray', alpha=0.3)
                
                # Add text
                ax.text(prediction[0]/2, 0, f"{prediction[0]:.2f}", ha='center', va='center', color='white', fontweight='bold')
                
                # Customize appearance
                ax.set_xlim(0, max_value)
                ax.set_yticks([])
                ax.set_xlabel('Page Views')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")

# Footer
st.markdown("---")
st.markdown("Website Traffic Prediction App | Made with Streamlit")