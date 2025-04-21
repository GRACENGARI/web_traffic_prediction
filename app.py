import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
import pickle

# Set page config
st.set_page_config(page_title="Website Analytics Dashboard", layout="wide")

# Create sample data
data = {
    'Date': pd.date_range(start='2023-01-01', periods=100),
    'Page Views': np.random.randint(100, 1000, 100),
    'Session Duration': np.random.uniform(1, 15, 100),
    'Bounce Rate': np.random.uniform(0.1, 0.5, 100),
    'Traffic Source': np.random.choice(['Organic', 'Social', 'Paid', 'Direct', 'Referral'], 100),
    'Time on Page': np.random.uniform(1, 10, 100),
    'Previous Visits': np.random.randint(0, 10, 100),
    'Conversion Rate': np.random.uniform(0.01, 0.1, 100)
}

# Create DataFrame
df = pd.DataFrame(data)

# Function to create and train LSTM model for time series prediction
def create_lstm_model(data, feature='Page Views', train_split=0.8):
    # Extract the feature we want to predict
    dataset = data[feature].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    # Save the scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Create sequences for LSTM
    def create_sequences(data, seq_length=7):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    
    # Use 7 days of data to predict the next day
    seq_length = 7
    X, y = create_sequences(scaled_data, seq_length)
    
    # Split into train and test sets
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Reshape input for LSTM [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Create LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train model
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    
    # Save model
    model.save('web_traffic_model.keras')
    
    return model, scaler, seq_length

# Function to make predictions
def predict_future_traffic(model, scaler, data, feature='Page Views', days=7, seq_length=7):
    # Extract the latest data points
    latest_data = data[feature].values[-seq_length:].reshape(-1, 1)
    
    # Scale the data
    scaled_data = scaler.transform(latest_data)
    
    # Make predictions
    predictions = []
    current_batch = scaled_data.reshape((1, seq_length, 1))
    
    for i in range(days):
        # Get prediction
        current_pred = model.predict(current_batch, verbose=0)[0]
        
        # Store prediction
        predictions.append(current_pred[0])
        
        # Update batch for next prediction
        current_batch = np.append(current_batch[:, 1:, :], 
                                 [[current_pred]], 
                                 axis=1)
    
    # Invert scaling
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    
    # Generate future dates
    last_date = data['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
    
    # Create prediction DataFrame
    pred_df = pd.DataFrame({
        'Date': future_dates,
        feature: predictions.flatten()
    })
    
    return pred_df

# Main dashboard
st.title("Website Analytics Dashboard")

# Check if model exists or create it
model_path = 'web_traffic_model.keras'
scaler_path = 'scaler.pkl'

if not os.path.exists(model_path):
    with st.spinner('Training prediction model... This may take a moment.'):
        model, scaler, seq_length = create_lstm_model(df)
        st.success('Model trained successfully!')
else:
    # Load existing model and scaler
    model = load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    seq_length = 7

# Overview section
st.header("Key Metrics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Page Views", f"{df['Page Views'].sum():,}")
with col2:
    st.metric("Avg. Session Duration", f"{round(df['Session Duration'].mean(), 2)} min")
with col3:
    st.metric("Avg. Bounce Rate", f"{round(df['Bounce Rate'].mean() * 100, 2)}%")
with col4:
    st.metric("Avg. Time on Page", f"{round(df['Time on Page'].mean(), 2)} min")

# Data Explorer
st.header("Data Explorer")
st.dataframe(df)

# Visualizations
st.header("Data Visualizations")

# Create two columns for charts
col1, col2 = st.columns(2)
with col1:
    st.subheader("Traffic Source Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    source_counts = df['Traffic Source'].value_counts()
    ax.pie(source_counts, labels=source_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)
with col2:
    st.subheader("Session Duration vs Time on Page")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x='Session Duration', y='Time on Page', hue='Traffic Source', size='Page Views',
                   sizes=(50, 200), ax=ax)
    st.pyplot(fig)

# Time series of page views
st.subheader("Page Views Over Time")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Date'], df['Page Views'])
ax.set_xlabel('Date')
ax.set_ylabel('Page Views')
ax.grid(True)
st.pyplot(fig)

# Traffic prediction section
st.header("Traffic Prediction")
prediction_days = st.slider("Number of days to predict", min_value=1, max_value=30, value=7)

# Make prediction
with st.spinner('Generating predictions...'):
    predictions = predict_future_traffic(model, scaler, df, days=prediction_days, seq_length=seq_length)

# Display predictions
st.subheader(f"Predicted Page Views for Next {prediction_days} Days")
st.dataframe(predictions)

# Plot predictions
fig, ax = plt.subplots(figsize=(12, 6))

# Plot historical data
ax.plot(df['Date'][-30:], df['Page Views'][-30:], label='Historical')

# Plot predictions
ax.plot(predictions['Date'], predictions['Page Views'], 'r--', label='Predicted')

ax.set_xlabel('Date')
ax.set_ylabel('Page Views')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Additional analysis
st.header("Traffic Source Analysis")

# Group by Traffic Source
source_analysis = df.groupby('Traffic Source').agg({
    'Page Views': 'mean',
    'Session Duration': 'mean',
    'Bounce Rate': 'mean',
    'Time on Page': 'mean',
    'Previous Visits': 'mean'
}).reset_index().round(2)

st.dataframe(source_analysis)

# Interactive filters
st.header("Interactive Filter")
selected_source = st.multiselect("Select Traffic Source", df['Traffic Source'].unique(),
                                default=df['Traffic Source'].unique())
if selected_source:
    filtered_df = df[df['Traffic Source'].isin(selected_source)]
    st.write(f"Showing data for selected traffic sources: {', '.join(selected_source)}")
    st.dataframe(filtered_df)
   
    # Comparative visualization for filtered data
    st.subheader("Metrics by Traffic Source")
    fig, ax = plt.subplots(figsize=(10, 6))
   
    source_metrics = filtered_df.groupby('Traffic Source').agg({
        'Bounce Rate': 'mean',
        'Time on Page': 'mean',
        'Session Duration': 'mean'
    }).reset_index()
   
    # Reshape for visualization
    metrics_melted = pd.melt(source_metrics, id_vars=['Traffic Source'],
                            value_vars=['Bounce Rate', 'Time on Page', 'Session Duration'],
                            var_name='Metric', value_name='Value')
   
    sns.barplot(data=metrics_melted, x='Traffic Source', y='Value', hue='Metric', ax=ax)
    ax.set_title('Comparison of Key Metrics by Traffic Source')
    st.pyplot(fig)
   
# Add correlation analysis
st.header("Correlation Analysis")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr = df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Website Analytics Dashboard - Created with Streamlit")