
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the Thai Government Lottery Results dataset
thai_lottery_dataset = load_dataset("ANTDPU/ThaiGovernmentLotteryResults")

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(thai_lottery_dataset['train'])

# Preprocess the data
# Assuming 'date' is a string column representing the date in format 'YYYY-MM-DD'
# Extract year, month, and day from the 'date' column
df['year'] = pd.to_datetime(df['date']).dt.year
df['month'] = pd.to_datetime(df['date']).dt.month
df['day'] = pd.to_datetime(df['date']).dt.day

# Split the data into features (X) and target (y)
X = df[['year', 'month', 'day']]
y = df['num']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model
mse = mean_squared_error(y_val, y_pred)
print(f"Mean Squared Error: {mse}")

# Predict future lottery numbers
# Assuming future_data is a DataFrame containing future dates
# Remember to preprocess future_data similarly to the training data
future_data = pd.DataFrame({"date": ["2024-04-30", "2024-05-01", "2024-05-02"]})  # Example future dates
future_data['year'] = pd.to_datetime(future_data['date']).dt.year
future_data['month'] = pd.to_datetime(future_data['date']).dt.month
future_data['day'] = pd.to_datetime(future_data['date']).dt.day

future_predictions = model.predict(future_data[['year', 'month', 'day']])

print("Predicted future lottery numbers:")
print(future_predictions)

