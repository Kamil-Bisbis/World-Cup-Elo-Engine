import pandas as pd
import numpy as np

# Load the dataset (replace the path with the actual path of your downloaded CSV file)
df = pd.read_csv("/Users/kamilbisbis/Documents/GitHub/Soccer-Elo-Engine/fifa-world-cup-1930-2022-all-match-dataset.csv")

# Preview the data
print(df.head())

# Clean and preprocess the data
# Remove columns you don't need (optional, depending on your analysis)
df = df.drop(['QualifiedTeams', 'MatchesPlayed','GoalScored', 'QualifiedTeams.1', 'MatchesPlayed.1','GoalScored.1'], axis=1)

# Rename columns for better readability (optional)
df = df.rename(columns={'Year': 'year', 'Country': 'host_country', 'Winner': 'winner', 'Runners-Up': 'runner_up', 'Third': 'third_place', 'Fourth': 'fourth_place', 'GoalsScored': 'total_goals', 'QualifiedTeams': 'teams', 'MatchesPlayed': 'matches', 'Attendance': 'attendance', 'GoalsScored.1': 'avg_goals', 'QualifiedTeams.1': 'avg_teams', 'MatchesPlayed.1': 'avg_matches', 'Attendance.1': 'avg_attendance'})

# Further clean the data
df['year'] = df['year'].astype(str)
df['year'] = pd.to_datetime(df['year'])
df['total_goals'] = df['total_goals'].str.replace(',', '').astype(int)
df['winner'] = df['winner'].str.strip()
df['runner_up'] = df['runner_up'].str.strip()
df['third_place'] = df['third_place'].str.strip()
df['fourth_place'] = df['fourth_place'].str.strip()

# Analyze and manipulate the data as needed
# For example, print the total number of goals scored in all World Cups
print(f'Total goals scored in all World Cup matches: {df["total_goals"].sum()}')

# Save the cleaned data into a CSV file (optional)
df.to_csv("cleaned_worldcup_data.csv", index=False)

print("Data cleaning and manipulation complete. Saved to 'cleaned_worldcup_data.csv'.")
