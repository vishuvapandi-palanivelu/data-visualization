from selenium import webdriver
from selenium.webdriver.support.ui import Select
import re, sqlite3, csv
import time

import plotly as py
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import squarify
import plotly.graph_objs as go

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model

# ---------------------------------------------------------------------------------------------------------------->>>>
#Chapter 1 - Web Scrapping

options = webdriver.ChromeOptions()
#options.add_argument('--disable-gpu')  # Disables GPU hardware acceleration
#options.add_argument('--no-sandbox')  # Bypass OS security model, WARNING: not recommended
#options.add_argument('--disable-software-rasterizer')
options.add_argument('--headless')  # Runs Chrome in headless mode.

# Initialize WebDriver with these options
driver = webdriver.Chrome(executable_path='C:/Users/vishu/Downloads/chromedriver_win32 (1)/chromedriver.exe', options=options)

#Data to Scrap (Players, Seasons, Player Data, MVP, Team Wins)
# Initialize the Chrome WebDriver with the path to the chromedriver executable
#driver = webdriver.Chrome(executable_path='C:/Users/vishu/Downloads/chromedriver_win32 (1)/chromedriver.exe')

#Store in a Database File
conn = sqlite3.connect('MSIS615_Team7_db_1(Player_Data).db')
c = conn.cursor()


#1(a). Player Data

c.execute("CREATE TABLE IF NOT EXISTS player_data(\
                name varchar(20),\
                year_start varchar(20),\
                year_end varchar(20),\
                position varchar(20),\
                height varchar(20),\
                weight varchar(20),\
                birth_date varchar(20),\
                college varchar(20))")

for letter in 'abcdefghijklmnopqrstuvwxyz':
    link = 'https://www.basketball-reference.com/players/'+letter+'/'

# Initialize the Chrome WebDriver with the path to the chromedriver executable
#    driver = webdriver.Chrome(executable_path='C:/Users/vishu/Downloads/chromedriver_win32 (1)/chromedriver.exe')

# Navigate to the specified URL
    try:
        driver.get(link)
    except:
        try:
            driver.get(link)
        except:
            driver.get(link)

# Retrieve the page source HTML
    html_source = driver.page_source

#    regex = '<a href="/players/*?">.*?</a>.*?<td.*?>(.*?)</td>.*?<td.*?>(.*?)</td>.*?<td.*?>(.*?)</td>.*?<td.*?>(.*?)</td>.*?<td.*?>(.*?)</td>.*?<a href="/friv/birthdays.*?">(.*?)</a>.*?<a href="/friv/colleges.*?">(.*?)</a>'
    regex = '<th scope="row" class="left " data-append-csv=".*?" data-stat="player">.*?<a href="/players/.*?">(.*?)</a>.*?</th>.*?<td class="right " data-stat="year_min">(.*?)</td>.*?<td class="right " data-stat="year_max">(.*?)</td>.*?<td class="center " data-stat="pos">(.*?)</td>.*?<td class="right " data-stat="height" csk=".*?">(.*?)</td>.*?<td class="right " data-stat="weight">(.*?)</td>.*?<a href="/friv/birthdays..*?">(.*?)</a>.*?<a href="/friv/colleges.*?">(.*?)</a>'
    matches=re.compile(regex, re.S|re.I).findall(html_source)

    for m in matches:
        name, year_start, year_end, position, height, weight, birth_date, college = m
        sqlcmnd = "INSERT INTO player_data VALUES (?,?,?,?,?,?,?,?)"
        c.execute(sqlcmnd ,(name, year_start, year_end, position, height, weight, birth_date, college))

    #time.sleep(5) 

conn.commit()
conn.close()

#1(b). Seasons Stats (1950-2024)

#Store in a Database File
conn = sqlite3.connect('MSIS615_Team7_db_2(Seasons_Stats).db')
c = conn.cursor()

c.execute("CREATE TABLE IF NOT EXISTS Seasons_stats(\
                Year varchar(20),\
                Player varchar(20),\
                Pos varchar(20),\
                Age varchar(20),\
                Tm varchar(20), G varchar(20), GS varchar(20), MP varchar(20), PER varchar(20), TS_PER varchar(20), G_3PAr varchar(20),\
                FTr varchar(20), ORB_PER varchar(20), DRB_PER varchar(20), TRB_PER varchar(20), AST_PER varchar(20), STL_PER varchar(20),\
                BLK_PER varchar(20), TOV_PER varchar(20), USG_PER varchar(20), BLANK1 varchar(20), OWS varchar(20), DWS varchar(20), WS varchar(20),\
                WS48 varchar(20), BLANK2 varchar(20), OBPM varchar(20), DBPM varchar(20), BPM varchar(20), VORP varchar(20), FG varchar(20),\
                FGA varchar(20), FG_PER varchar(20), G_3P varchar(20), G_3PA varchar(20), G_3P_PER varchar(20), G_2P varchar(20),\
                G_2PA varchar(20), G_2P_PER varchar(20), eFG_PER varchar(20), FT varchar(20), FTA varchar(20), FT_PER varchar(20), ORB varchar(20),\
                DRB varchar(20), TRB varchar(20), AST varchar(20), STL varchar(20), BLK varchar(20), TOV varchar(20), PF varchar(20), PTS varchar(20))")

season_advanced = "https://www.basketball-reference.com/leagues/NBA_{}_advanced.html"
season_total = "https://www.basketball-reference.com/leagues/NBA_{}_totals.html"

start_year = 1950
end_year = 2024

for year in range(start_year, end_year+1):

    count = 0
    while count != 2:

        if count == 0:
            link = season_advanced.format(year)
            ind_num = 28
        elif count == 1:
            link = season_total.format(year)
            ind_num = 29

        # Navigate to the specified URL
        try:
            driver.get(link)
        except:
            try:
                driver.get(link)
            except:
                driver.get(link)

        # Retrieve the page source HTML
        html_source = driver.page_source

        links = re.findall(r"<td.*?>(.*?)</td>",html_source)
        lis=[]
        mainlis=[]
        index = 0
        num = 0
        for val in links:
            lis.append(val)
            index+=1
            if index%ind_num==0:
                copy=lis[0]
                lis[0]=re.findall(r'<a href="/players/.*?">(.*?)</a>', copy)[0]
            
                if lis[3].startswith('<a href'):
                    copy = lis[3]
                    lis[3]=re.findall(r'<a href="/teams.*?">(.*?)</a>', copy)[0]

                mainlis.append(lis)
                lis=[]
                num+=1

        if count == 0:
            sqlcmnd = "INSERT INTO Seasons_Stats (Year,Player,Pos,Age,Tm,G,MP,PER,TS_PER,G_3PAr,FTr,ORB_PER,DRB_PER,TRB_PER,AST_PER,STL_PER,BLK_PER,TOV_PER,USG_PER,BLANK1,OWS,DWS,WS,WS48,BLANK2,OBPM,DBPM,BPM,VORP) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
            for m in mainlis:
                c.execute(sqlcmnd ,(year,m[0],m[1],m[2],m[3],m[4],m[5],m[6],m[7],m[8],m[9],m[10],m[11],m[12],m[13],m[14],m[15],m[16],m[17],m[18],m[19],m[20],m[21],m[22],m[23],m[24],m[25],m[26],m[27]))
        elif count == 1:
            sqlcmnd = "UPDATE Seasons_Stats SET FG = ?, FGA = ?, FG_PER = ?, G_3P = ?, G_3PA = ?, G_3P_PER = ?, G_2P = ?, G_2PA = ?, G_2P_PER = ?, eFG_PER = ?, FT = ?, FTA = ?, FT_PER = ?, ORB = ?, DRB = ?, TRB = ?, AST = ?, STL = ?, BLK = ?, TOV = ?, PF = ?, PTS = ? WHERE Year = ? AND Player = ? AND Tm = ?"
            for m in mainlis:
                c.execute(sqlcmnd ,(m[7],m[8],m[9],m[10],m[11],m[12],m[13],m[14],m[15],m[16],m[17],m[18],m[19],m[20],m[21],m[22],m[23],m[24],m[25],m[26],m[27],m[28],year,m[0],m[3]))
    
        count += 1

conn.commit()
conn.close()

#1(c) - MVP Players

#Store in a Database File
conn = sqlite3.connect('MSIS615_Team7_db_3(MVP_Players).db')
c = conn.cursor()

c.execute("CREATE TABLE IF NOT EXISTS mvp_players(\
                Year varchar(20),\
                Player varchar(20))")

link = 'https://www.basketball-reference.com/awards/mvp.html'

# Navigate to the specified URL
try:
    driver.get(link)
except:
    try:
        driver.get(link)
    except:
        driver.get(link)

# Retrieve the page source HTML
html_source = driver.page_source

regex = '<th scope="row".*?data-stat="season"><a href="/leagues/NBA_(.*?).html">.*?</a></th>.*?<a href="/players/.*?.html">(.*?)</a>'
matches=re.compile(regex, re.S|re.I).findall(html_source)

for m in matches:
    year, player = m
    sqlcmnd = "INSERT INTO mvp VALUES (?,?)"
    c.execute(sqlcmnd ,(year, player))

conn.commit()
conn.close()

#1(d) - Team Wins

#Store in a Database File
conn = sqlite3.connect('MSIS615_Team7_db_4(Team_Wins).db')
c = conn.cursor()

c.execute("CREATE TABLE IF NOT EXISTS team_wins(\
                Team varchar(20),\
                Year varchar(20),\
                Wins varchar(20))")

teams = ['ATL','BOS','BRK','CHA','NJN','CHH','CHI','CHO','CLE','DAL','DEN','DET','GSW','HOU','IND','LAC','LAL','MEM','VAN','MIA','MIL','MIN','NOH','NOK','NOP','NYK','OKC','ORL','PHI','PHO','POR','SAC','SAS','SEA','UTA','WAS']
link = 'https://www.basketball-reference.com/teams/'

for team in teams:
    # Construct the full URL by appending the team abbreviation
    url = f"{link}{team}/"

    # Navigate to the specified URL
    try:
        driver.get(url)
    except:
        try:
            driver.get(url)
        except:
            driver.get(url)

    time.sleep(5)

    # Retrieve the page source HTML
    html_source = driver.page_source

    regex = '<th scope="row" class="left " data-stat="season"><a href="/teams/.*?/(.*?).html">.*?</a></th>.*?<td class="right " data-stat="wins">(.*?)</td>'
    matches=re.compile(regex, re.S|re.I).findall(html_source)

    for m in matches:
        year, wins = m
        sqlcmnd = "INSERT INTO team_wins VALUES (?,?,?)"
        c.execute(sqlcmnd ,(team, year, wins))

conn.commit()
conn.close()

driver.close()

# ---------------------------------------------------------------------------------------------------------------->>>>
# DEFINITIONS
#
def fi(clf):
    feature_importance = clf.feature_importances_
    max_importance = feature_importance.max()
    #feature_importance = 100.0 * (feature_importance / feature_importance.max())
    # Check if max_importance is zero or NaN
    if max_importance == 0 or np.isnan(max_importance):
        print("Max importance is zero or NaN, adjusting to avoid division by zero.")
        feature_importance = np.zeros_like(feature_importance)
    else:
        feature_importance = 100.0 * (feature_importance / max_importance)

    sorted_idx = np.argsort(feature_importance)
    sorted_idx = sorted_idx[len(feature_importance) - 30:]
    pos = np.arange(sorted_idx.shape[0]) + .5

    non_zero_indices = feature_importance[sorted_idx] > 0
    filtered_importances = feature_importance[sorted_idx][non_zero_indices]
    filtered_labels = X_train.columns[sorted_idx][non_zero_indices]

    labels = [f"{label}: {value:.2f}_PER" for label, value in zip(filtered_labels, filtered_importances)]

    plt.figure(figsize=(12,8))
    squarify.plot(sizes=filtered_importances, label=labels, alpha=.8 )
    plt.axis('off')
    plt.yticks(pos, X_train.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')

    plt.show()

def check_nan(in_):
    print("Remaining NaN's:" + str(in_.isna().sum().sum()))
    if(in_.isna().sum().sum() > 0): 
        in_.fillna(0, inplace=True) # Avoid NaN in regression

def xy_sets(train, test):
    X_train = train.iloc[ : , 1:train.shape[1] - 1]
    y_train = train.iloc[:, train.shape[1] - 1]

    X_test = test.iloc[ : , 1:test.shape[1] - 1]
    y_test = test.iloc[ : , test.shape[1] - 1]
    return X_train, X_test, y_train, y_test

def db_to_csv(db_path, table_name, csv_file_name):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    # Load the table into a pandas DataFrame
    query = f'SELECT * FROM {table_name}'
    df = pd.read_sql_query(query, conn)
    # Write the DataFrame to CSV
    df.to_csv(csv_file_name, index=False)
    # Close the database connection
    conn.close()
#
# ---------------------------------------------------------------------------------------------------------------->>>>

# DATA EXTRACTION

# File paths to the databases
db_player_data = 'MSIS615_Team7_db_1(Player_Data).db'
db_seasons_stats = 'MSIS615_Team7_db_2(Seasons_Stats).db'
db_players = 'MSIS615_Team7_db_1(Player_Data).db'

# Convert and save to CSV
db_to_csv(db_player_data, 'player_data', 'player_data.csv')  
db_to_csv(db_seasons_stats, 'Seasons_stats', 'seasons_stats.csv')
db_to_csv(db_players, 'player_data', 'players.csv') 

# Example of reading the CSV files into pandas DataFrames
seasons_stats = pd.read_csv('seasons_stats.csv')
player_data = pd.read_csv('player_data.csv')
players = pd.read_csv('players.csv')

players.rename(columns={'name': 'Player'}, inplace=True)
players['born'] = pd.to_datetime(players['birth_date']).dt.year

# Path to the SQLite database file
db_path = 'MSIS615_Team7_db_3(MVP_Players).db'
# Path to the CSV file to be created
csv_path = 'mvp_players.csv'

# Connect to the SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# SQL query to fetch all data from the table
table_name = "mvp_players" 
query = f"SELECT * FROM {table_name}"

# Execute the query
cursor.execute(query)

# Fetch all results
rows = cursor.fetchall()

# Get the column headers
columns = [description[0] for description in cursor.description]

# Open a CSV file to write the data
#with open(csv_path, 'w', newline='') as csv_file:
#    csv_writer = csv.writer(csv_file)   
    # Write the header to the CSV file
#    csv_writer.writerow(columns)
    # Write the rows of data to the CSV file
#    csv_writer.writerows(rows)

query = "SELECT player, year FROM mvp_players ORDER BY player, year"
cursor.execute(query)
rows = cursor.fetchall()
mvp_players = {}

for player, year in rows:
    if player in mvp_players:
        mvp_players[player].append(int(year))
    else:
        mvp_players[player] = [int(year)]

# Close the database connection
conn.close()

db_path = 'MSIS615_Team7_db_4(Team_Wins).db'

# Connecting to the SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
query = "SELECT team, year, wins FROM team_wins"
cursor.execute(query)

# Fetch all records
records = cursor.fetchall()

cursor.close()
conn.close()

# Process rows into the desired dictionary format
teams_wins = {}
for team, year, wins in records:
    year = int(year)  # Convert year to integer
    wins = int(wins)  # Convert wins to integer
    if team not in teams_wins:
        teams_wins[team] = {}
    teams_wins[team][year] = wins

# ---------------------------------------------------------------------------------------------------------------->>>>

# Chapter 2 - Regression (K-Means Clustering)

# DATA CLEANING
seasons_stats = seasons_stats[~seasons_stats.Player.isnull()]
players = players[~players.Player.isnull()]
players = players.rename(columns = {'Unnamed: 0':'id'})
num_players = player_data.groupby('name').count()
num_players =  num_players.iloc[:,:1]
num_players = num_players.reset_index()
num_players.columns = ['Player', 'count']
num_players[num_players['count'] > 1].head()

#seasons_stats = seasons_stats.iloc[:,1:]
seasons_stats = seasons_stats.drop(['BLANK1', 'BLANK2'], axis=1)
player_data['id'] = player_data.index
mj_stats = seasons_stats[seasons_stats.Player == 'Michael Jordan']
mj_stats['Year'].iloc[0] - mj_stats['Age'].iloc[0] 

seasons_stats['born'] = seasons_stats['Year'] - seasons_stats['Age'] - 1
players = players[~players.born.isnull()]

# We will concatenate players and player_data dataframes because none has all players
players_born = players[['Player', 'born']]
player_data = player_data[~player_data.birth_date.isnull()]
for i, row in player_data.iterrows():
    player_data.loc[i, 'born'] = float(row['birth_date'].split(',')[1])
player_data_born = player_data[['name', 'born']]
player_data_born.columns = ['Player', 'born']
born = pd.concat([players_born, player_data_born])
born = born.drop_duplicates()
born = born.reset_index()
born = born.drop('index', axis=1)
born['id'] = born.index

data = seasons_stats.merge(born, on=['Player', 'born'])
data = data[data.Tm != 'TOT']

# Filter players with at least 800 min in a season at played at least half of the matchs
data = data[(data.MP > 800) & (data.G > 40)]
# Per games
data['PPG'] = data['PTS'] / data['G']
data['APG'] = data['AST'] / data['G']
data['RPG'] = data['TRB'] / data['G']
data['SPG'] = data['STL'] / data['G']
data['BPG'] = data['BLK'] / data['G']
data['FPG'] = data['PF'] / data['G']
data['TOVPG'] = data['TOV'] / data['G']

data['MVP'] = 0
for i, row in data.iterrows():  
    for k, v in mvp_players.items():
        for year in v:
            if row['Player'] != k:
                break
            elif(row['Year'] == year) & (row['Player'] == k):
                data.loc[i, 'MVP'] = 1
                break

dataMvpOnly = data.loc[data['MVP'] != 0]            

train = seasons_stats[(seasons_stats['Year'].astype(int) > 1985) & (seasons_stats['Year'].astype(int) < 2006)]
val = seasons_stats[(seasons_stats['Year'].astype(int) > 2005) & (seasons_stats['Year'].astype(int) < 2012)]
test = seasons_stats[(seasons_stats['Year'].astype(int) > 2012)]

data = data[data.Year >= 2000]

# Adding Team Wins since 2000 to show this important paramater
data.sort_values(by='Tm').Tm.unique()
data[data.Tm == 'NOH'].Year.unique()

for i, row in data.iterrows():  
    for k, v in teams_wins.items():
        for year, value in v.items():
            if ((row['Tm'] == k) & (row['Year'] == year)):
                data.loc[i, 'Tm_Wins'] = value

data_mvp = data[['id', 'Player', 'Year', 'PER', 'WS', 'BPM', 'VORP', 'PPG', 'Tm_Wins', 'MVP']]
data_mvp = data_mvp.fillna(0)

data_mvp = data_mvp.drop(['Player'], axis=1)

check_nan(data_mvp)

train, test = train_test_split(data_mvp, test_size=0.25, shuffle=True )
X_train, X_test, y_train, y_test = xy_sets(train, test)

clf = tree.DecisionTreeRegressor(max_depth=5)
clf = clf.fit(X_train, y_train)
pred = clf.predict(X_test)

fi(clf)

# ---------------------------------------------------------------------------------------------------------------->>>>
# Chapter 3 - Data Visualization

season_data = {}
df = pd.read_csv('Seasons_stats.csv')
columns_to_extract = ['Player', 'Pos', 'Age', 'PER', 'USG_PER', 'G','Year']

for year, group in df.groupby('Year'):
        # Ensure the year is a valid integer
        try:
            year = int(year)
        except ValueError:
            continue  # Skip entries with invalid year data
        
        # Filter the data for the specified columns and store it as a DataFrame in the dictionary
        season_data[year] = group[columns_to_extract]

for i in list(range(1952,2017)):
    season_data[i] = season_data[i].groupby('Player').agg({'Pos': 'first', 'Age': 'first', 'PER': 'mean', 'USG_PER': 'mean', 'G': 'sum', 'Year': 'first'}).reset_index()

# Methodology - Identifying each player's peak using PER
# First, creating an empty dataframe for each of the players with data
position_mapping = {
    'C': 'CENTER',
    'C-PF': 'CENTER',
    'PF': 'POWER FORWARD',
    'PF-SF': 'POWER FORWARD',
    'PG': 'POINT GUARD',
    'PG-SG': 'POINT GUARD',
    'SF': 'SMALL FORWARD',
    'SF-SG': 'SMALL FORWARD',
    'SG': 'SHOOTING GUARD',
    'SG-PG': 'SHOOTING GUARD'
}

player_data = {}
for i in list(range(1952,2017)):
    for index, row in season_data[i].iterrows():
        player_data[row['Player']] = pd.DataFrame(columns=['Player', 'Pos', 'Age', 'PER', 'USG_PER', 'G', 'Year'])

# Methodology
# Second, appending each season's data to the dataframe as a row, based on player name
for year in range(1952, 2017):
    # Check if the year exists in the season_data dictionary
    if year in season_data:
        season_data[year]['Pos'] = season_data[year]['Pos'].map(position_mapping).fillna(season_data[year]['Pos'])
        # Iterate through each row in the DataFrame for that year
        for index, row in season_data[year].iterrows():
            player = row['Player']
            # If the player does not exist in the player_data, initialize with an empty DataFrame
            if player not in player_data:
                player_data[player] = pd.DataFrame(columns=row.index)
            # Append the row to the respective player's DataFrame
            player_data[player] = pd.concat([player_data[player], pd.DataFrame([row])], ignore_index=True)

# Data Cleaning
# To ensure proper identification of a player's peak, scoping to players with more than 5 years in the league.
player_data1 = {}
for key,value in player_data.items():
    if len(player_data[key].index) > 5:
        player_data1[key] = player_data[key]
        
# To maintain fairness and reliability of results,
# only looking at seasons in which players played at least half of the full season (41 games).
for key,value in player_data1.items():
    player_data1[key] = player_data1[key][player_data1[key]['G']>=41]

# Methodology
# Third, sorting each season by PER (descending), and keeping the year with the highest PER
# Using the year column to identify which was the player's first year in the league,
# and saving it as the player's draft (first) year.
player_data_topPER = {}
for key,value in player_data1.items():
    player_data_topPER[key] = player_data1[key].sort_values(['PER'], ascending = False).head(1)
    player_data_topPER[key]['draftyr'] = player_data1[key]['Year'].min()

# Methodology
# Fourth, concatenating all the peak years into one dataframe for further analysis

peak_year_players = pd.DataFrame(columns=['Player', 'Pos', 'Age', 'PER', 'USG%', 'G', 'Year', 'draftyr'])
for key,value in player_data_topPER.items():
    peak_year_players = pd.concat([peak_year_players,player_data_topPER[key]])

# Data Cleaning
# To maintain fairness and reliability of results,
# only looking at seasons in which players played at least half of the full season (41 games).
peak_year_players = peak_year_players[peak_year_players['G']>=41]

peak_year_players.to_csv('Peak Year Players.csv', index = False)

df = pd.read_csv('Peak Year Players.csv')

# Assuming the column for player peak age is named 'Age', if not, replace 'Age' with the correct column name
# Calculate age distribution
age_counts = df['Age'].value_counts().sort_index()
total_players = df['Age'].count()

# Convert counts to percentages
age_percentages = (age_counts / total_players) * 100

# Create the histogram
plt.figure(figsize=(10, 7))
bars = plt.bar(age_percentages.index, age_percentages.values, color='salmon')

# Adding percentage labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')

plt.xlabel('Age')
plt.ylabel('Percentage of Players at Peak')
plt.title('Peak Distribution')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Set the aesthetic style of the plots
sns.set(style='whitegrid', context='talk')  # 'talk' context has larger fonts suitable for presentations

# Assuming there's a column 'Year' to determine the decade
df['Decade'] = df['Year'].apply(lambda year: (year // 10) * 10)

# 1. Average Peak Age by Draft Decade - Line Plot
plt.figure(figsize=(10, 7))
decade_avg = df.groupby('Decade')['Age'].mean()
ax1 = sns.lineplot(x=decade_avg.index, y=decade_avg.values, marker='o', linestyle='-', linewidth=2.5, color='dodgerblue')
ax1.set_title('Draft Decade vs Average Peak Age')
ax1.set_xlabel('Draft Decade')
ax1.set_ylabel('Average Peak Age')
ax1.set_ylim(25, decade_avg.max() + 1)  # Adjust y-axis to start from 25
#plt.show()

# 2. Average Peak Age by Position - Area Plot
# Prepare to plot the average peak ages by position using a box plot
avg_ages = df.groupby('Pos')['Age'].mean().reset_index()

# Sort the positions based on the average peak age for better visualization
avg_ages = avg_ages.sort_values(by='Age', ascending=False)

# Prepare to plot using seaborn for better aesthetics
plt.figure(figsize=(10,7))
bar_plot = sns.barplot(x='Pos', y='Age', data=avg_ages, palette='coolwarm')

# Set plot details
plt.title('Average Peak Age by Position')
plt.xlabel('Position')
plt.ylabel('Average Peak Age')
plt.ylim(25, avg_ages['Age'].max() + 1)  # Ensure the y-axis starts at 25 and gives some space above the highest value

# Rotate x-axis labels to prevent overlap
plt.xticks(rotation=45)  # Rotate labels by 45 degrees

# Annotate each bar with the age value
for p in bar_plot.patches:
    bar_plot.annotate(format(p.get_height(), '.1f'), 
        (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha = 'center', va = 'center', 
            xytext = (0, 9), 
            textcoords = 'offset points')

plt.show()

# ---------------------------------------------------------------------------------------------------------------->>>>
# Chapter 4 - Descriptive Analysis

peak_year_data = pd.read_csv('Peak Year Players.csv')
season_stats_data = pd.read_csv('Seasons_Stats.csv')

# Merge the datasets on 'Player' and 'Year' columns
merged_data = pd.merge(peak_year_data, season_stats_data, on=['Player', 'Year'], suffixes=('_peak', '_season'))

# Clean up the data by dropping columns with all null values
cleaned_data = merged_data.dropna(axis=1, how='all')

# Define key metrics for analysis
key_metrics = ['PER_peak', 'PER_season', 'PTS', 'TRB', 'AST']

# Calculate correlation matrix for the key metrics
correlation_matrix = cleaned_data[key_metrics].corr()

# Print the correlation matrix
#print("Correlation Matrix:\n", correlation_matrix)

# creating heatmap
plt.figure(figsize=(20, 15))
sns.heatmap(correlation_matrix,
            annot = True,
            fmt = '.2f',
            cmap='Blues')
plt.title('Correlation between variables of the Team dataset')
plt.show()