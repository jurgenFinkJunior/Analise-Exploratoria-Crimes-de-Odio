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

if __name__ == "__main__":
    victims_by_year()
    victims_by_bias()
    race_on_race()
    victims_by_bias(2024)
    race_on_race(2024)