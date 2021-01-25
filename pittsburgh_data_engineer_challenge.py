import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import *
from datetime import datetime


print("Part 1: Injesting Raw Data Into application_in")
application_in = pd.read_json("https://data.pa.gov/resource/mcba-yywm.json")
print(application_in.head())


print("\nPart 2: Moving Rows With Null Values Into invalid_data")
print("appplication_in null count and shape before removal:")
print(application_in.isnull().sum())
print(application_in.shape)

mask = application_in.isnull().any(axis=1)
invalid_data = application_in[mask]
print("\ninvalid_data first rows, null count, and shape:")
print(invalid_data.head())

# verifying that invalid_data contains all of the null data from application_in
print(invalid_data.isnull().sum())
print(invalid_data.shape)

# verifying that application_in no longer contains null data
application_in = application_in[~mask]
print("\napplication_in null count and shape after removing null values:")
print(application_in.isnull().sum())
print(application_in.shape)


print("\nPart 3: Converting Senate District Names To Snake Case")
print("Senate District names before conversion:")
print(application_in['senate'].unique())

# changing the senate district names to all lower case
application_in['senate'] = application_in['senate'].str.lower()
#application_in['senate'].unique()

# replacing all spaces with underscores
application_in['senate'] = application_in['senate'].str.replace(' ', '_')
print("Senate District names after setting all letters lowercase and replacing spaces with underscores:")
print(application_in['senate'].unique())


print("\nPart 4: Adding yr_born Column")
# taking a substring containing only year from birth date column
application_in['yr_born'] = application_in['dateofbirth'].str[0:4]
print("application_in after adding new column:")
print(application_in.head())

# reording columns to place yr_born to the right of dateofbirth
application_in = application_in[['appissuedate', 'appreturndate', 'ballotreturneddate', 'ballotsentdate', 'congressional', 'countyname', 'dateofbirth', 'yr_born', 'legislative', 'mailapplicationtype', 'party', 'senate']]
print("application_in after adding reordering columns:")
print(application_in.head())

print("Current datatype of yr_born: " + str(application_in['yr_born'].dtype))

# converting yr_born datatype to integer
application_in['yr_born'] = application_in['yr_born'].apply(pd.to_numeric)
print("Datatype of yr_born after conversion: " + str(application_in['yr_born'].dtype))


print("\nPart 5: Finding Relationship Between Age, Party Affiliation, & Overall Ballot Requests")
application_in['age'] = 0
application_in['requests'] = 1
# calculating (approximate) age by subtracting year of election from birth year
for i, row in application_in.iterrows():
    application_in.at[i, 'age'] = 2020 - row['yr_born']

print("application_in after adding new columns to keep track of voter age (in 2020) and request count:")
print(application_in.head())

# creating new dataframe that groups age and party affiliation, summing up request totals for each combination
age_party_counts = application_in.groupby(['age','party'], as_index=False)['requests'].sum()
print("Dataframe that groups together age and party affiliation to aggregate request total:")
print(age_party_counts)

print("All unique parties included in dataset:")
print(age_party_counts['party'].unique())

parties = ['D', 'R']
# combining all non-democratic or republican party affiliations into a single third party
for i, row in age_party_counts.iterrows():
    if age_party_counts.at[i, 'party'] not in parties:
        age_party_counts.at[i, 'party'] = 'I'
        
print("\nAll unique parties after combining all parties besides Democratic & Republican into a single party:")
print(age_party_counts['party'].unique())

print("\nAge data in new dataframe:")
print(age_party_counts['age'].describe())

# making separate dataframes for each party affiliation to simplify graphing
dem_requests = age_party_counts[age_party_counts.party=='D']
rep_requests = age_party_counts[age_party_counts.party=='R']
third_party_requests = age_party_counts[age_party_counts.party=='I']

print("\nAge data for Democrats only:")
print(dem_requests['age'].describe())
print("\nAge data for Republicans only:")
print(rep_requests['age'].describe())
print("\nAge data for third party voters only:")
print(third_party_requests['age'].describe())

print("\nDemocratic voter ballot request total: " + str(dem_requests['requests'].sum()))
print("Republican voter ballot request total: " + str(rep_requests['requests'].sum()))
print("Third party voter ballot request total: " + str(third_party_requests['requests'].sum()) + "\n")

# generating scatterplot to show ballot request numbers for specific ages grouped by party
fig, ax = plt.subplots(figsize=(18,7))
colors = {'D':'blue', 'R':'red', 'I':'green'}
ax.scatter(dem_requests['age'], dem_requests['requests'], c=dem_requests['party'].apply(lambda x: colors[x]))
ax.scatter(rep_requests['age'], rep_requests['requests'], c=rep_requests['party'].apply(lambda x: colors[x]))
ax.scatter(third_party_requests['age'], third_party_requests['requests'], c=third_party_requests['party'].apply(lambda x: colors[x]))
plt.legend(labels=['Democrats','Republicans','Third Party'])
plt.title('Ballot Requests as a Function of Age & Party', size=24)
plt.xlabel('Age', size=18)
plt.ylabel('Ballot Requests', size=18)
plt.savefig("age_party_counts_scatter.png")
plt.show()

# grouping ages into ~10 year bins to make graph more readable
age_ranges = [0, 29, 39, 49, 59, 69, 79, 89, 100]
age_party_counts['age_bins'] = pd.cut(x=age_party_counts['age'], bins=age_ranges)
print("Dataframe that groups voters by ~10 year age ranges:")
print(age_party_counts.head())

age_group_party = age_party_counts.groupby(['age_bins','party'], as_index=False)['requests'].sum()
print("\nAggregating ballot request totals within each age range by party affiliation:")
print(age_group_party)

dem_data = age_group_party[age_group_party.party=='D']
rep_data = age_group_party[age_group_party.party=='R']
third_party_data = age_group_party[age_group_party.party=='I']

# generating bar chart with separate bars for each party affiliation within each age range
X = np.arange(dem_data['age_bins'].count())
fig, ax = plt.subplots()
ax.bar(X + 0.00, dem_data['requests'], color=dem_data['party'].apply(lambda x: colors[x]), width = 0.25)
ax.bar(X + 0.25, rep_data['requests'], color=rep_data['party'].apply(lambda x: colors[x]), width = 0.25)
ax.bar(X + 0.50, third_party_data['requests'], color=third_party_data['party'].apply(lambda x: colors[x]), width = 0.25)
ax.set_xlabel('Age Ranges')
ax.set_ylabel('Ballot Requests')
ax.set_title('Ballot Requests as a Function of Age Group & Party')
ax.set_xticklabels(['00', '18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90+'])
ax.set_yticks(np.arange(0, 130, 10))
ax.legend(labels=['Democrats', 'Republicans', 'Third Party'])
plt.savefig("age_group_party_counts.png")
plt.show()

print("Ballot request rates were highest for Democrats across all ages, "
      "other than voters aged 90 and up, for which they were level with Republicans at 9 ballot requests. "
      "The difference is most noticeable among voters in their 60s, with 125 Democratic voters requesting ballots "
      "compared to only 32 Republicans. Democratic voters were the oldest on average at 57.05 years, "
      "with third party voters being the youngest on average at 51.13 years. "
      "Ballot request rates for both Republican and Democratic voters dropped off within the age range of 30-49. "
      "For Democratic voters, ballot requests decreased from 81 requests for 18-29 year old voters to 55 requests "
      "for 30-39 year olds and 59 requests for 40-49 year olds, before rising again to 98 requests for voters in their 50s. "
      "This trend was similar for Republican voters, but less severe. Request numbers dropped from 26 ballots for 18-29 year olds "
      "to 11 requests for 30-39 year olds and 15 for 40-49 year olds, before rising to 25 ballot requests for voters in their 50s. "
      "Third party ballot request rates were low for voters of all ages. However, more third party voters requested ballots "
      "between the ages of 30-49 than Republicans within the same age range. A total of 35 third party voters between the age of "
      "30-49 requested ballots, compared to only 26 Republicans.")

print("\nPart 6: Finding Median Latency for Issuing Applications and Returning Ballots for Each Legislative District")
print("Legislative District ballot totals:")
print(application_in['legislative'].value_counts())

# creating latency column that is calculated by subtracting issue date from return date
application_in['latency'] = 0
date_mask = '%Y-%m-%d'
for i, row in application_in.iterrows():
    a = datetime.strptime(str(application_in.at[i, 'ballotreturneddate'])[0:10], date_mask)
    b = datetime.strptime(str(application_in.at[i, 'appissuedate'])[0:10], date_mask)
    application_in.at[i, 'latency'] = a - b
    application_in.at[i, 'latency'] = pd.to_numeric(application_in.at[i, 'latency'].days, downcast='integer')
    
print("\napplication_in after adding new latency column that is calculated by subtracting issue date from return date:")
print(application_in.head())

# grouping new dataframe from latency period for each legislative district
leg_latency = application_in[['legislative', 'latency']]
print("Latency data for each Legislative District:")
print(leg_latency.sort_values(by='legislative'))

leg_latencies = application_in.groupby(['legislative'])['latency'].apply(np.median)
print("Median latency for each Legislative District:")
print(leg_latencies.to_string())


print("\nPart 7: Congressional District With Highest Ballot Request Rate")
district_counts = application_in['congressional'].value_counts()
print("Ballot request data for each Congressional District:")
print(district_counts)

print("\n" + district_counts.index.tolist()[0] +
      " is the district with the highest frequency of ballot requests, with a total of " + 
      str(district_counts[0]) + " requests.")


print("\nPart 8: Republican & Democratic Application Counts in Each County")
# reducing dataframe to only include information for democrats and republicans, then limiting to relevant columns
dem_rep_data = application_in[application_in['party'].isin(parties)]
dem_rep_data = dem_rep_data[['countyname', 'party', 'requests']]
#dem_rep_data

# adapted from https://stackoverflow.com/questions/42854801/including-missing-combinations-of-values-in-a-pandas-groupby-aggregation
# if a given party hasn't received any ballot requests from a county, this autofills that row's request count to be 0
county_party_counts = dem_rep_data.groupby(['countyname', 'party']).requests.sum().unstack(fill_value=0).stack().reset_index(name='requests')
print("Dataframe that groups together ballot requests by Democrats and Republicans in each county:")
print(county_party_counts)

#Adapted from https://www.pythoncharts.com/matplotlib/grouped-bar-charts-matplotlib/
# generating bar chart that plots democratic vs. republican ballot request rates for each county
sns.set_context('talk')
fig, ax = plt.subplots(figsize=(100, 20))
x = np.arange(len(county_party_counts['countyname'].unique()))
bar_width = 0.4
b1 = ax.bar(x, county_party_counts.loc[county_party_counts['party'] == 'D', 'requests'],
            color='blue', width=bar_width, label='Democrats')
b2 = ax.bar(x + bar_width, county_party_counts.loc[county_party_counts['party'] == 'R', 'requests'],
            color='red', width=bar_width, label='Republicans')

ax.set_xticks(x + bar_width / 2)
ax.set_xticklabels(county_party_counts['countyname'].unique())

ax.legend(fontsize=40)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.tick_params(bottom=False, left=False)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color='#EEEEEE')
ax.xaxis.grid(False)

ax.set_xlabel('County', fontsize=40, labelpad=15)
ax.set_ylabel('Ballot Requests', fontsize=40, labelpad=15)
ax.set_title('Ballot Requests For Each County', fontsize=40, pad=15)

fig.tight_layout()
plt.savefig("county_party_counts.png")
plt.show()