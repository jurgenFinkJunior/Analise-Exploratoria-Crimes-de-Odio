import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = 'output/'
MULTIPLE_SEP = ';'

#create dataframe from csv
df = pd.read_csv('hate_crime.csv')

#keep only relevant columns
columns_to_keep = [
    "incident_id", 
    "data_year", 
    "agency_type_name", 
    "state_name", 
    "division_name", 
    "region_name", 
    "population_group_description", 
    "incident_date", 
    "adult_victim_count", 
    "juvenile_victim_count", 
    "total_offender_count", 
    "adult_offender_count", 
    "juvenile_offender_count", 
    "offender_race", 
    "offender_ethnicity", 
    "victim_count", 
    "offense_name", 
    "total_individual_victims", 
    "location_name", 
    "bias_desc", 
    "victim_types", 
    "multiple_offense", 
    "multiple_bias"
]
df = df[columns_to_keep]

presidents = {
    1991: 'Bush Sr',
    1992: 'Bush Sr',
    1993: 'Clinton',
    1994: 'Clinton',
    1995: 'Clinton',
    1996: 'Clinton',
    1997: 'Clinton',
    1998: 'Clinton',
    1999: 'Clinton',
    2000: 'Clinton',
    2001: 'Bush Jr',
    2002: 'Bush Jr',
    2003: 'Bush Jr',
    2004: 'Bush Jr',
    2005: 'Bush Jr',
    2006: 'Bush Jr',
    2007: 'Bush Jr',
    2008: 'Bush Jr',
    2009: 'Obama',
    2010: 'Obama',
    2011: 'Obama',
    2012: 'Obama',
    2013: 'Obama',
    2014: 'Obama',
    2015: 'Obama',
    2016: 'Obama',
    2017: 'Trump',
    2018: 'Trump',
    2019: 'Trump',
    2020: 'Trump',
    2021: 'Biden',
    2022: 'Biden',
    2023: 'Biden',
    2024: 'Biden',
}

parties = {
    'Bush Sr': 'Republican',
    'Clinton': 'Democrat',
    'Bush Jr': 'Republican',
    'Obama': 'Democrat',
    'Trump': 'Republican',
    'Biden': 'Democrat',
}

df['president'] = df['data_year'].map(presidents)
df['party'] = df['president'].map(parties)

print(df.head())

def victims_by_year():
    crime_by_year = df.groupby('data_year')['total_individual_victims'].sum()
    plt.figure(figsize=(10, 6))
    crime_by_year.plot(kind='bar', color='skyblue')
    plt.title('Total Hate Crime Victims by Year')
    plt.xlabel('Year')
    plt.ylabel('Total Victims')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'hate_crime_victims_by_year.png')
    plt.show()

def victims_by_bias(year=None):
    df_year = df
    if year is not None:
        df_year = df[df['data_year'] == year]

    #when multiple biases are listed, split them and count each separately
    df_expanded = df_year.assign(bias_desc=df_year['bias_desc'].str.split(MULTIPLE_SEP)).explode('bias_desc')
    crime_by_bias = df_expanded.groupby('bias_desc')['total_individual_victims'].sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    crime_by_bias.plot(kind='barh', color='salmon')
    plt.title('Top Hate Crime Victims by Bias Description' + (f' in {year}' if year else ''))
    plt.xlabel('Total Victims')
    plt.ylabel('Bias Description')
    #plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + f'hate_crime_victims_by_bias{"_" + str(year) if year else ""}.png')
    plt.show()

def race_on_race(year=None):
    df_year = df
    if year is not None:
        df_year = df[df['data_year'] == year]

    #filter by bias to include only racial related hate crimes
    racial_biases = ['Anti-Black or African American', 'Anti-White', 'Anti-Asian', 
                      'Anti-Native Hawaiian or Other Pacific Islander', 'Anti-American Indian or Alaska Native',]
    df_race = df_year[df_year['bias_desc'].isin(racial_biases)]

    #filter to remove where offender race is not valid
    offender_races_to_exclude = ['Unknown', 'Not Specified', 'Multiple']
    df_race = df_race[~df_race['offender_race'].isin(offender_races_to_exclude)]

    #rename bias to victim race for clarity
    bias_to_victim_race = {
        'Anti-Black or African American': 'Black or African American',
        'Anti-White': 'White',
        'Anti-Asian': 'Asian',
        'Anti-Native Hawaiian or Other Pacific Islander': 'Native Hawaiian or Other Pacific Islander',
        'Anti-American Indian or Alaska Native': 'American Indian or Alaska Native'
    }
    df_race = df_race.replace({'bias_desc': bias_to_victim_race})

    #when multiple races are listed, split them and count each separately
    df_race = df_race.assign(offender_race=df_race['offender_race'].str.split(MULTIPLE_SEP)).explode('offender_race')

    #create crosstab and order rows and columns consistently
    race_crosstab = pd.crosstab(df_race['offender_race'], df_race['bias_desc'])
    
    #define consistent race order for both rows and columns
    race_order = ['White', 'Black or African American', 'Asian', 
                  'American Indian or Alaska Native', 'Native Hawaiian or Other Pacific Islander']
    
    #reindex both rows and columns to have the same order, but reversed
    race_crosstab = race_crosstab.reindex(index=race_order[::-1], columns=race_order, fill_value=0)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(race_crosstab, cmap='viridis', aspect='auto')
    plt.colorbar(label='Number of Victims')
    plt.xticks(ticks=range(len(race_crosstab.columns)), labels=race_crosstab.columns, rotation=45, ha='right')
    plt.yticks(ticks=range(len(race_crosstab.index)), labels=race_crosstab.index)
    plt.title('Hate Crime Victims by Offender and Victim Race' + (f' in {year}' if year else ''))
    plt.xlabel('Victim Race')
    plt.ylabel('Offender Race')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + f'hate_crime_offender_race_vs_victim{"_" + str(year) if year else ""}.png')
    plt.show()

def victims_by_president_and_party():
    crime_by_president = df.groupby('president')['total_individual_victims'].sum()
    crime_by_party = df.groupby('party')['total_individual_victims'].sum()

    plt.figure(figsize=(10, 6))
    crime_by_president.plot(kind='bar', color='lightgreen')
    plt.title('Total Hate Crime Victims by U.S. President')
    plt.xlabel('President')
    plt.ylabel('Total Victims')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'hate_crime_victims_by_president.png')
    plt.show()

    plt.figure(figsize=(8, 6))
    crime_by_party.plot(kind='bar', color='orange')
    plt.title('Total Hate Crime Victims by Political Party')
    plt.xlabel('Political Party')
    plt.ylabel('Total Victims')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'hate_crime_victims_by_party.png')
    plt.show()

def histogram_by_presidential_term_colored_by_party():
    plt.figure(figsize=(12, 6))
    for term_years_president_party, group in df.groupby(['president', 'party']):
        president, party = term_years_president_party
        plt.hist(group['data_year'], bins=range(df['data_year'].min(), df['data_year'].max() + 2), 
                 alpha=0.5, label=f'{president} ({party})', color=((0, 0, 1) if party == 'Democrat' else (1, 0, 0)))

    plt.title('Hate Crime Incidents Over Time by Presidential Party')
    plt.xlabel('Year')
    plt.xticks(range(df['data_year'].min(), df['data_year'].max() + 1))
    plt.ylabel('Number of Incidents')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'hate_crime_victims_by_party_histogram.png')
    plt.show()

if __name__ == "__main__":
    #victims_by_year()
    #victims_by_bias()
    #race_on_race()
    #victims_by_bias(2024)
    #race_on_race(2024)
    #victims_by_president_and_party()
    histogram_by_presidential_term_colored_by_party()

    #TODO: histogram by date separated by transfer of power date of presidency (round to year, its close anyway)

    #TODO: download population by race by year to do per capita
