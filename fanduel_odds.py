import requests
import pandas as pd
import joblib
import numpy as np

API_KEY = '260eabf9df4291276b979adf99e30742'
SPORT = 'soccer_epl'
REGION = 'us'
MARKET = 'h2h'

# Load your trained model
model = joblib.load('model.pkl')

url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds/'
params = {
    'apiKey': API_KEY,
    'regions': REGION,
    'markets': MARKET,
    'bookmakers': 'fanduel'
}

response = requests.get(url, params=params)

if response.status_code != 200:
    print('Error:', response.status_code, response.text)
else:
    odds_data = response.json()
    matches = []

    for game in odds_data:
        home = game['home_team']
        away = game['away_team']
        commence = game['commence_time']
        for bookmaker in game['bookmakers']:
            for market in bookmaker['markets']:
                for outcome in market['outcomes']:
                    team = outcome['name']
                    odds = outcome['price']

                    # ➕ Predict win probability using trained model
                    home_encoded = hash(home) % 1000
                    away_encoded = hash(away) % 1000
                    goal_diff = 0  # Unknown before game
                    total_goals = 0  # Unknown before game
                    features = np.array([[home_encoded, away_encoded, goal_diff, total_goals]])
                    est_prob = round(model.predict_proba(features)[0][1], 3)

                    payout = odds - 1
                    EV = round((est_prob * payout) - ((1 - est_prob) * 1), 3)

                    # ✅ Kelly Calculation
                    b = payout
                    p = est_prob
                    q = 1 - p
                    kelly_fraction = (b * p - q) / b if b != 0 else 0
                    kelly_fraction = max(kelly_fraction, 0)
                    kelly_fraction = round(kelly_fraction * 0.5, 3)

                    matches.append({
                        'Match': f"{home} vs {away}",
                        'Team': team,
                        'Odds': odds,
                        'Est. Prob': est_prob,
                        'EV': EV,
                        'Kelly Stake %': kelly_fraction
                    })

    df = pd.DataFrame(matches)

    # ✅ Filter: Show only bets with positive EV and Kelly suggestion
    df = df[(df['EV'] > 0) & (df['Kelly Stake %'] > 0)]

    # ✅ Sort by highest EV
    df = df.sort_values(by='EV', ascending=False)

    # ✅ Export to CSV
    df.to_csv("profitable_bets.csv", index=False)

    # ✅ Print to terminal
    print(df)
    print("\nSaved profitable bets to: profitable_bets.csv ✅")
