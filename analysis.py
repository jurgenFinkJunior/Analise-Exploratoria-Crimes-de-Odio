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

print("Distinct Offender Race Counts:")
print(df['offender_race'].value_counts())

print("Distinct Bias Counts:")
df_expl = df.assign(bias_desc=df['bias_desc'].str.split(MULTIPLE_SEP)).explode('bias_desc')
#keep only the ones with more than one occurrence to remove strange values with typos and such
df_expl = df_expl[df_expl['bias_desc'].map(df_expl['bias_desc'].value_counts()) > 1]
print(df_expl['bias_desc'].value_counts())

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
        'Anti-American Indian or Alaska Native': 'American Indian or Alaska Native',
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

def victims_by_presidential_terms():
    # removed bush sr because data starts in the middle of his term
    presidential_bins = [1993, 1997, 2001, 2005, 2009, 2013, 2017, 2021, 2025]
    
    term_labels = [
        '1993-1996\nClinton\n(Dem)', 
        '1997-2000\nClinton\n(Dem)', 
        '2001-2004\nBush Jr\n(Rep)',
        '2005-2008\nBush Jr\n(Rep)',
        '2009-2012\nObama\n(Dem)',
        '2013-2016\nObama\n(Dem)',
        '2017-2020\nTrump\n(Rep)',
        '2021-2024\nBiden\n(Dem)'
    ]
    
    # Create the histogram
    plt.figure(figsize=(14, 8))
    
    # Filter out rows with missing victim data to avoid NaN issues
    df_clean = df.dropna(subset=['total_individual_victims'])
    
    years = df_clean['data_year'].tolist()
    weights = df_clean['total_individual_victims'].tolist()
    n, bins, patches = plt.hist(years, bins=presidential_bins, weights=weights, 
                               edgecolor='black', alpha=0.7)
    
    # Color patches by party
    party_colors = ['red' if 'Rep' in label else 'blue' for label in term_labels]
    for patch, color in zip(patches, party_colors):
        patch.set_facecolor(color)
    
    plt.title('Distribution of Hate Crime Victims by Presidential Terms', fontsize=16)
    plt.xlabel('Presidential Terms', fontsize=12)
    plt.ylabel('Total Victims', fontsize=12)
    
    # Set x-tick positions to the center of each bin
    bin_centers = [(presidential_bins[i] + presidential_bins[i+1]) / 2 for i in range(len(presidential_bins)-1)]
    plt.xticks(bin_centers, term_labels, rotation=0, ha='center')
    
    # Add value labels on top of bars
    for i, value in enumerate(n):
        plt.text(bin_centers[i], value + max(n)*0.01, 
                str(int(value)), ha='center', va='bottom', fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Democrat'),
                      Patch(facecolor='red', alpha=0.7, label='Republican')]
    plt.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'hate_crime_victims_by_presidential_terms_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()

def boxplot_of_victims_per_crime():
    df_clean = df.dropna(subset=['total_individual_victims'])
    #keep only rows with 10 or more victims
    df_clean = df_clean[df_clean['total_individual_victims'] > 9]
    #split offense_name by MULTIPLE_SEP and explode to have one offense per row
    df_expanded = df_clean.assign(offense_name=df['offense_name'].str.split(MULTIPLE_SEP)).explode('offense_name')
    #keep only selected offense names to avoid clutter
    top_offenses = df_expanded['offense_name'].value_counts().nlargest(6).index
    df_expanded = df_expanded[df_expanded['offense_name'].isin(top_offenses)]
    
    plt.figure(figsize=(12, 8))
    
    # Create horizontal boxplot using matplotlib
    offense_names = df_expanded['offense_name'].unique()
    data_by_offense = [df_expanded[df_expanded['offense_name'] == offense]['total_individual_victims'].values 
                      for offense in offense_names]
    
    # Create horizontal boxplot
    box_plot = plt.boxplot(data_by_offense, vert=False, patch_artist=True, labels=offense_names)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
    for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
        patch.set_facecolor(color)
    
    plt.title('Number of Victims in Crimes with 10 or more Victims by Offense Name', fontsize=14)
    plt.xlabel('Number of Victims')
    plt.ylabel('Offense Name')
    plt.xscale('log')  # Use logarithmic scale for better visibility
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'boxplot_victims_per_crime_horizontal.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    #victims_by_year()
    #victims_by_bias()
    #race_on_race()
    #victims_by_bias(2024)
    #race_on_race(2024)
    #victims_by_presidential_terms()
    boxplot_of_victims_per_crime()

    #TODO: download population by race by year to do per capita