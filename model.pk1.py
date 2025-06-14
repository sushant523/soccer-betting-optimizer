import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# --- Load and Clean Data ---
df = pd.read_csv('E0.csv')

# Basic cleanup: drop rows with missing values in key columns
df = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].dropna()

# Create target: 1 if home team wins, 0 otherwise
df['HomeWin'] = df['FTR'].apply(lambda x: 1 if x == 'H' else 0)

# Features: goal difference, total goals, etc.
df['GoalDiff'] = df['FTHG'] - df['FTAG']
df['TotalGoals'] = df['FTHG'] + df['FTAG']

# Encode teams as numeric features
df['HomeTeam'] = df['HomeTeam'].astype('category').cat.codes
df['AwayTeam'] = df['AwayTeam'].astype('category').cat.codes

# Features and labels
X = df[['HomeTeam', 'AwayTeam', 'GoalDiff', 'TotalGoals']]
y = df['HomeWin']

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.pkl')

print("✅ Model trained and saved as model.pkl")
