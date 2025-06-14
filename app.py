import requests
import pandas as pd
import streamlit as st
import joblib
import numpy as np

API_KEY = '260eabf9df4291276b979adf99e30742'
SPORT = 'soccer_epl'
REGION = 'us'
MARKET = 'h2h'

# Load trained model
model = joblib.load('model.pkl')

# --- Function to fetch odds and apply predictions ---
def fetch_and_predict(bankroll):
    url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds/'
    params = {
        'apiKey': API_KEY,
        'regions': REGION,
        'markets': MARKET,
        'bookmakers': 'fanduel'
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        st.error(f"Error {response.status_code}: {response.text}")
        return pd.DataFrame()

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

                    # Predict win probability using model
                    home_encoded = hash(home) % 1000
                    away_encoded = hash(away) % 1000
                    goal_diff = 0
                    total_goals = 0
                    features = np.array([[home_encoded, away_encoded, goal_diff, total_goals]])
                    est_prob = round(model.predict_proba(features)[0][1], 3)

                    payout = odds - 1
                    EV = round((est_prob * payout) - ((1 - est_prob) * 1), 3)

                    # Kelly Calculation
                    b = payout
                    p = est_prob
                    q = 1 - p
                    kelly_fraction = (b * p - q) / b if b != 0 else 0
                    kelly_fraction = max(kelly_fraction, 0)
                    kelly_fraction = round(kelly_fraction * 0.5, 3)

                    # Calculate actual bet amount
                    bet_amount = round(kelly_fraction * bankroll, 2)

                    matches.append({
                        'Match': f"{home} vs {away}",
                        'Team': team,
                        'Odds': odds,
                        'Est. Prob': est_prob,
                        'EV': EV,
                        'Kelly Stake %': kelly_fraction,
                        'Bet Amount ($)': bet_amount
                    })

    return pd.DataFrame(matches)

# --- Streamlit UI ---
st.set_page_config(page_title="⚽ Soccer Betting Optimizer", layout="wide")
st.title("⚽ Soccer Betting Optimizer (Model-Based)")

with st.sidebar:
    st.header("Filters & Bankroll")
    bankroll = st.number_input("Your current bankroll ($)", min_value=10, value=100)
    min_ev = st.slider("Minimum EV", 0.0, 5.0, 0.5, step=0.1)
    min_kelly = st.slider("Minimum Kelly %", 0.0, 0.5, 0.05, step=0.01)
    show_all = st.checkbox("Show All Bets (Ignore Filters)", value=False)

st.write("🔄 Fetching EPL odds and calculating smart picks...")

df = fetch_and_predict(bankroll)

if df.empty:
    st.warning("No matches or odds data available.")
else:
    if not show_all:
        df = df[(df['EV'] >= min_ev) & (df['Kelly Stake %'] >= min_kelly)]

    df = df.sort_values(by='EV', ascending=False)

    st.success(f"✅ {len(df)} smart bets found")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📁 Download CSV", csv, "model_based_bets.csv", "text/csv")
