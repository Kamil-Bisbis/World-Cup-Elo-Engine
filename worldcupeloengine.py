import pandas as pd
import numpy as np
import warnings
import itertools

# -----------------------------
# Step 0: Configure Script Parameters
# -----------------------------

# Suppress FutureWarnings related to pandas to maintain clean output
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuration Parameters
CSV_PATH = "/Users/kamilbisbis/Documents/GitHub/World-Cup-Elo-Engine/FIFA World Cup 1930-2022 All Match Dataset.csv"
OUTPUT_MATCHES_CSV = 'worldcup_matches_with_elo.csv'
OUTPUT_PEAK_ELO_CSV = 'teams_peak_elo.csv'
OUTPUT_AVG_ELO_CSV = 'teams_average_elo.csv'
TOP_N_TEAMS = 50  # Specifies the number of top teams to save separately

# Elo Calculation Parameters
INITIAL_ELO = 1500.0  # Initial Elo rating assigned to all teams
BASE_K_FACTOR = 40     # Base K-factor determining the sensitivity of Elo updates

# Define K-Factor based on Tournament Stage with increased emphasis
# Each subsequent stage has a K-factor 1.5 times greater than the previous stage
STAGE_K_FACTOR = {
    'group stage': 40,
    'round of 16': 60,         # 1.5x group stage
    'quarter-finals': 90,      # 1.5x round of 16
    'semi-finals': 135,        # 1.5x quarter-finals
    'final': 202.5,            # 1.5x semi-finals
    'third place': 202.5,      # Same as final
    # Additional stages can be included as needed
}

# Bonus Elo for Tournament Outcomes with increased emphasis
TOURNAMENT_WIN_BONUS = 150          # Additional Elo points awarded to the tournament winner
RUNNER_UP_BONUS = 75                # Additional Elo points awarded to the tournament runner-up
THIRD_PLACE_BONUS = 75              # Additional Elo points awarded to the third-place team

# -----------------------------
# Step 1: Load and Clean the Data
# -----------------------------

try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"Error: The file at path '{CSV_PATH}' was not found.")
    exit(1)

# Remove columns that are not required for Elo calculation to streamline the dataset
columns_to_drop = [
    'Key Id', 'Tournament Id', 'Match Id', 'Match Name',
    'Group Name', 'Group Stage', 'Knockout Stage', 'Replayed', 'Replay',
    'Stadium Id', 'Stadium Name', 'City Name', 'Country Name',
    'Home Team Id', 'Home Team Code', 'Away Team Id', 'Away Team Code',
    'Score', 'Home Team Score Margin', 'Away Team Score Margin',
    'Extra Time', 'Penalty Shootout', 'Score Penalties',
    'Home Team Score Penalties', 'Away Team Score Penalties'
]
df = df.drop(columns=columns_to_drop, axis=1, errors='ignore')

# Rename columns for enhanced clarity and consistency
df = df.rename(columns={
    'Match Date': 'date',
    'Match Time': 'time',
    'Home Team Name': 'team1',
    'Away Team Name': 'team2',
    'Home Team Score': 'team1_score',
    'Away Team Score': 'team2_score',
    'Result': 'result',
    'tournament Name': 'Tournament Name',  # Ensure consistent naming
})

# Standardize team names by merging "West Germany" into "Germany"
df['team1'] = df['team1'].replace('West Germany', 'Germany')
df['team2'] = df['team2'].replace('West Germany', 'Germany')

# Convert the 'date' column to datetime format for accurate chronological sorting
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Extract tournament year from 'Tournament Name'
df['tournament_year'] = df['Tournament Name'].str.extract('(\d{4})').astype(float)
df['tournament_year'].fillna(df['date'].dt.year, inplace=True)
df['tournament_year'] = df['tournament_year'].astype(int)

# Function to sanitize score entries by removing non-standard characters
def clean_score(score):
    if isinstance(score, str):
        # Remove any non-standard characters (e.g., '�', '–', '−') from the score
        score = score.replace('�', '').replace('–', '-').replace('−', '-')
        return score
    return score

# Apply the cleaning function to score columns
df['team1_score'] = df['team1_score'].apply(clean_score)
df['team2_score'] = df['team2_score'].apply(clean_score)

# Convert score columns to integers, replacing non-numeric entries with 0
df['team1_score'] = pd.to_numeric(df['team1_score'], errors='coerce').fillna(0).astype(int)
df['team2_score'] = pd.to_numeric(df['team2_score'], errors='coerce').fillna(0).astype(int)

# Sort matches in chronological order to ensure accurate Elo calculations
df = df.sort_values(by='date').reset_index(drop=True)

# Assign unique match IDs sequentially
df['match_id'] = np.arange(1, len(df) + 1)

# -----------------------------
# Step 2: Initialize Elo Ratings
# -----------------------------

elo_ratings = {}

# -----------------------------
# Step 3: Define Elo Functions
# -----------------------------

def expected_score(elo_a, elo_b):
    """
    Calculate the expected score for Team A against Team B based on their Elo ratings.

    Parameters:
    elo_a (float): Elo rating of Team A.
    elo_b (float): Elo rating of Team B.

    Returns:
    float: Expected probability of Team A winning.
    """
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def update_elo(winner_elo, loser_elo, k, result=1, goal_diff=1):
    """
    Update the Elo ratings of the winning and losing teams based on the match outcome and goal difference.

    Parameters:
    winner_elo (float): Current Elo rating of the winning team.
    loser_elo (float): Current Elo rating of the losing team.
    k (float): K-factor influencing the magnitude of Elo changes.
    result (float): Actual result of the match (1 for win, 0 for loss, 0.5 for draw).
    goal_diff (int): Goal difference in the match.

    Returns:
    tuple: Updated Elo ratings for the winner and loser.
    """
    expected_win = expected_score(winner_elo, loser_elo)
    # Adjust K-factor based on goal difference to reward decisive victories
    adjusted_k = k + (goal_diff * 2)  # Example: Each goal difference adds 2 to K
    new_winner_elo = winner_elo + adjusted_k * (result - expected_win)
    new_loser_elo = loser_elo + adjusted_k * (0 - (1 - expected_win))
    return round(new_winner_elo, 2), round(new_loser_elo, 2)

# -----------------------------
# Step 4: Determine Match Results
# -----------------------------

def determine_result(row):
    """
    Determine the outcome of a match based on the result column or score comparison.

    Parameters:
    row (Series): A row from the DataFrame representing a match.

    Returns:
    str: 'team1_win', 'team2_win', or 'draw' indicating the match outcome.
    """
    if pd.isnull(row['result']):
        # If the result is missing, determine outcome based on scores
        if row['team1_score'] > row['team2_score']:
            return 'team1_win'
        elif row['team1_score'] < row['team2_score']:
            return 'team2_win'
        else:
            return 'draw'
    else:
        # Determine outcome based on the result column
        if row['result'] == 'home team win':
            return 'team1_win'
        elif row['result'] == 'away team win':
            return 'team2_win'
        elif row['result'] == 'draw':
            return 'draw'
        else:
            # Handle unexpected result values by comparing scores
            if row['team1_score'] > row['team2_score']:
                return 'team1_win'
            elif row['team1_score'] < row['team2_score']:
                return 'team2_win'
            else:
                return 'draw'

# -----------------------------
# Step 5: Calculate Elo Ratings
# -----------------------------

# Initialize Elo rating columns as floats to accommodate decimal values
df['team1_elo_start'] = 0.0
df['team2_elo_start'] = 0.0
df['team1_elo_end'] = 0.0
df['team2_elo_end'] = 0.0

# Iterate through each match to update Elo ratings accordingly
for index, row in df.iterrows():
    team1 = row['team1']
    team2 = row['team2']

    # Initialize Elo ratings for new teams
    if team1 not in elo_ratings:
        elo_ratings[team1] = INITIAL_ELO
    if team2 not in elo_ratings:
        elo_ratings[team2] = INITIAL_ELO

    # Retrieve current Elo ratings
    team1_elo_start = elo_ratings[team1]
    team2_elo_start = elo_ratings[team2]

    # Record starting Elo ratings before the match
    df.at[index, 'team1_elo_start'] = team1_elo_start
    df.at[index, 'team2_elo_start'] = team2_elo_start

    # Determine the result of the match
    match_result = determine_result(row)

    # Assign K-factor based on the tournament stage
    stage = row['Stage Name'].lower()
    k = STAGE_K_FACTOR.get(stage, BASE_K_FACTOR)  # Default to BASE_K_FACTOR if stage not found

    # Calculate goal difference to influence Elo adjustments
    goal_diff = abs(row['team1_score'] - row['team2_score'])

    # Identify if the match is a Final or Third Place playoff
    is_final = stage == 'final'
    is_third_place = stage == 'third place'

    # Apply increased K-factor for critical stages
    if is_final or is_third_place:
        k_final = k * 1.5  # 1.5x multiplier for finals and third place
    else:
        k_final = k

    if match_result == 'team1_win':
        # Update Elo ratings for a Team1 victory
        new_team1_elo, new_team2_elo = update_elo(team1_elo_start, team2_elo_start, k_final, result=1, goal_diff=goal_diff)

        # Award bonus Elo to the tournament winner or third place
        if is_final:
            new_team1_elo += TOURNAMENT_WIN_BONUS
        elif is_third_place:
            new_team1_elo += THIRD_PLACE_BONUS
    elif match_result == 'team2_win':
        # Update Elo ratings for a Team2 victory
        new_team2_elo, new_team1_elo = update_elo(team2_elo_start, team1_elo_start, k_final, result=1, goal_diff=goal_diff)

        # Award bonus Elo to the tournament winner or third place
        if is_final:
            new_team2_elo += TOURNAMENT_WIN_BONUS
        elif is_third_place:
            new_team2_elo += THIRD_PLACE_BONUS
    elif match_result == 'draw':
        # Update Elo ratings for a draw outcome
        expected_team1 = expected_score(team1_elo_start, team2_elo_start)
        expected_team2 = expected_score(team2_elo_start, team1_elo_start)
        # Adjust K-factor based on goal difference; for draws, goal_diff is 0
        adjusted_k = k_final + (goal_diff * 2)
        new_team1_elo = team1_elo_start + adjusted_k * (0.5 - expected_team1)
        new_team2_elo = team2_elo_start + adjusted_k * (0.5 - expected_team2)
        new_team1_elo = round(new_team1_elo, 2)
        new_team2_elo = round(new_team2_elo, 2)
    else:
        # Handle any unexpected outcomes by maintaining current Elo ratings
        new_team1_elo, new_team2_elo = team1_elo_start, team2_elo_start

    # Record updated Elo ratings after the match
    df.at[index, 'team1_elo_end'] = new_team1_elo
    df.at[index, 'team2_elo_end'] = new_team2_elo

    # Update the Elo ratings dictionary with new values
    elo_ratings[team1] = new_team1_elo
    elo_ratings[team2] = new_team2_elo

    # Assign Runner-Up Bonus for Final matches
    if is_final:
        # Identify the runner-up and assign bonus
        if match_result == 'team1_win':
            elo_ratings[team2] += RUNNER_UP_BONUS
            df.at[index, 'team2_elo_end'] = elo_ratings[team2]
        elif match_result == 'team2_win':
            elo_ratings[team1] += RUNNER_UP_BONUS
            df.at[index, 'team1_elo_end'] = elo_ratings[team1]

# -----------------------------
# Step 6: Export Match Data with Elo Ratings
# -----------------------------

# Save the DataFrame containing match data along with Elo ratings to a CSV file
df.to_csv(OUTPUT_MATCHES_CSV, index=False)

# -----------------------------
# Step 7: Calculate Peak Elo and Average Elo
# -----------------------------

# Prepare team Elo ratings over time
team1_data = df[['date', 'team1', 'team1_elo_end', 'tournament_year']].rename(columns={'team1': 'team', 'team1_elo_end': 'elo'})
team2_data = df[['date', 'team2', 'team2_elo_end', 'tournament_year']].rename(columns={'team2': 'team', 'team2_elo_end': 'elo'})
team_elos = pd.concat([team1_data, team2_data], ignore_index=True)

# Calculate Peak Elo for each team
peak_elos = team_elos.loc[team_elos.groupby('team')['elo'].idxmax()].reset_index(drop=True)
peak_elos['year'] = peak_elos['date'].dt.year

# Sort teams by Peak Elo in descending order
peak_elos_sorted = peak_elos.sort_values(by='elo', ascending=False).reset_index(drop=True)

# Save Peak Elo data to CSV
peak_elos_sorted.to_csv(OUTPUT_PEAK_ELO_CSV, index=False)

# Display the top 10 greatest teams by Peak Elo
top10_peak_elos = peak_elos_sorted.head(10)
print("\nTop 10 Greatest Teams by Peak Elo:")
for idx, row in top10_peak_elos.iterrows():
    rank = idx + 1
    team = row['team']
    elo = row['elo']
    year = row['year']
    print(f"{rank}. {team} - Peak Elo: {elo} in {year}")

# Calculate Average Elo across all World Cups
teams = team_elos['team'].unique()
tournament_years = df['tournament_year'].unique()

# Create all combinations of teams and tournament years
team_tournament_combinations = pd.DataFrame(list(itertools.product(teams, tournament_years)), columns=['team', 'tournament_year'])

# Get the last Elo rating for each team in each tournament
last_elos = team_elos.sort_values('date').groupby(['team', 'tournament_year']).tail(1)

# Merge to ensure all combinations are present
avg_elos_full = pd.merge(team_tournament_combinations, last_elos[['team', 'tournament_year', 'elo']], on=['team', 'tournament_year'], how='left')

# Fill missing Elo ratings with INITIAL_ELO for tournaments the team did not participate in
avg_elos_full['elo'].fillna(INITIAL_ELO, inplace=True)

# Calculate the average Elo for each team across all 22 World Cups
avg_elos_team = avg_elos_full.groupby('team')['elo'].mean().reset_index()

# Sort teams by Average Elo in descending order
avg_elos_team_sorted = avg_elos_team.sort_values(by='elo', ascending=False).reset_index(drop=True)

# Save Average Elo data to CSV
avg_elos_team_sorted.to_csv(OUTPUT_AVG_ELO_CSV, index=False)

# Display the top N teams by Average Elo
print(f"\nTop {TOP_N_TEAMS} Teams by Average Elo across all World Cups:")
for idx, row in avg_elos_team_sorted.head(TOP_N_TEAMS).iterrows():
    rank = idx + 1
    team = row['team']
    elo = round(row['elo'], 2)
    print(f"{rank}. {team} - Average Elo: {elo}")

# -----------------------------
# End of Script
# -----------------------------
