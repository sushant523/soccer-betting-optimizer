import requests
import pandas as pd
import streamlit as st

API_KEY = '260eabf9df4291276b979adf99e30742'
LEAGUES = [
    'soccer_epl',
    'soccer_spain_la_liga',
    'soccer_germany_bundesliga',
    'soccer_italy_serie_a',
    'soccer_uefa_champs_league'
]
REGION = 'us'
MARKET = 'h2h'


# --- Fetch Odds for All Leagues ---
def fetch_all_odds():
    all_matches = []

    for league in LEAGUES:
        url = f'https://api.the-odds-api.com/v4/sports/{league}/odds/'
        params = {
            'apiKey': API_KEY,
            'regions': REGION,
            'markets': MARKET,
            'bookmakers': 'fanduel'
        }

        response = requests.get(url, params=params)
        if response.status_code != 200:
            st.warning(f"[{league}] Error {response.status_code}: {response.text}")
            continue

        odds_data = response.json()

        for game in odds_data:
            home = game['home_team']
            away = game['away_team']
            commence = game['commence_time']
            league_name = league.replace('soccer_', '').replace('_', ' ').title()

            for bookmaker in game['bookmakers']:
                for market in bookmaker['markets']:
                    for outcome in market['outcomes']:
                        team = outcome['name']
                        odds = outcome['price']

                        # ➕ Estimated win probabilities (placeholder logic)
                        if 'Manchester City' in team:
                            est_prob = 0.75
                        elif 'Sheffield' in team:
                            est_prob = 0.2
                        else:
                            est_prob = 0.5

                        payout = odds - 1
                        EV = round((est_prob * payout) - ((1 - est_prob) * 1), 3)

                        b = payout
                        p = est_prob
                        q = 1 - p
                        kelly_fraction = (b * p - q) / b if b != 0 else 0
                        kelly_fraction = max(kelly_fraction, 0)
                        kelly_fraction = round(kelly_fraction * 0.5, 3)

                        all_matches.append({
                            'League': league_name,
                            'Match': f"{home} vs {away}",
                            'Team': team,
                            'Odds': odds,
                            'Est. Prob': est_prob,
                            'EV': EV,
                            'Kelly Stake %': kelly_fraction
                        })

    return pd.DataFrame(all_matches)


# --- Streamlit UI ---
st.set_page_config(page_title="Soccer Betting Optimizer", layout="wide")
st.title("Sushant bet odds (Free API (need to update))")

with st.sidebar:
    st.header("Filters")
    min_ev = st.slider("Minimum EV", 0.0, 5.0, 0.5, step=0.1)
    min_kelly = st.slider("Minimum Kelly %", 0.0, 0.5, 0.05, step=0.01)
    show_all = st.checkbox("Show All Bets (Ignore Filters)", value=False)

st.write("🔄 Fetching odds from 5 top leagues...")
df = fetch_all_odds()

if df.empty:
    st.warning("No betting data found.")
else:
    if not show_all:
        df = df[(df['EV'] >= min_ev) & (df['Kelly Stake %'] >= min_kelly)]

    df = df.sort_values(by='EV', ascending=False)

    st.success(f"✅ {len(df)} recommended bets found")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📁 Download CSV", csv, "multi_league_bets.csv", "text/csv")
