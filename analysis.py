import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import OrRd
import geopandas as gpd
import numpy as np

OUTPUT_DIR = 'output/'
MULTIPLE_SEP = ';'

#create dataframe from csv
df = pd.read_csv('hate_crime.csv')

#keep only relevant columns
columns_to_keep = [
    "incident_id", 
    "data_year", 
    "agency_type_name", 
    "state_abbr",
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
df_bias = df.assign(bias_desc=df['bias_desc'].str.split(MULTIPLE_SEP)).explode('bias_desc')
#keep only the ones with more than one occurrence to remove strange values with typos and such
df_bias = df_bias[df_bias['bias_desc'].map(df_bias['bias_desc'].value_counts()) > 1]
print(df_bias['bias_desc'].value_counts())

def victims_by_year():
    crime_by_year = df.groupby('data_year')['total_individual_victims'].sum()
    plt.figure(figsize=(10, 6))
    crime_by_year.plot(kind='bar', color='skyblue')
    plt.title('Total Hate Crime Victims by Year')
    plt.xlabel('Year')
    plt.ylabel('Total Victims')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'bar_victims_by_year.png')
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
    plt.savefig(OUTPUT_DIR + f'barh_victims_by_bias{"_" + str(year) if year else ""}.png')
    plt.show()

def race_on_race(year=None):
    df_year = df
    if year is not None:
        df_year = df[df['data_year'] == year]

    #filter by bias to include only racial related hate crimes
    racial_biases = [
        'Anti-Black or African American', 'Anti-White', 'Anti-Asian', 
        'Anti-Native Hawaiian or Other Pacific Islander', 'Anti-American Indian or Alaska Native',
        'Anti-Hispanic or Latino', 'Anti-Arab', 'Anti-Other Race/Ethnicity/Ancestry'
    ]
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
        'Anti-Hispanic or Latino': 'Hispanic or Latino',
        'Anti-Arab': 'Arab',
        'Anti-Other Race/Ethnicity/Ancestry': 'Other Race/Ethnicity/Ancestry'
    }
    df_race = df_race.replace({'bias_desc': bias_to_victim_race})

    #when multiple races are listed, split them and count each separately
    df_race = df_race.assign(offender_race=df_race['offender_race'].str.split(MULTIPLE_SEP)).explode('offender_race')

    #create crosstab and order rows and columns consistently
    race_crosstab = pd.crosstab(df_race['offender_race'], df_race['bias_desc'])

    lines = ['White', 'Black or African American', 'Asian', 
             'American Indian or Alaska Native', 'Native Hawaiian or Other Pacific Islander']
    columns = ['White', 'Black or African American', 'Asian', 
               'American Indian or Alaska Native', 'Native Hawaiian or Other Pacific Islander',
               'Hispanic or Latino', 'Arab', 'Other Race/Ethnicity/Ancestry']

    #reindex both rows and columns to have the same order, but reversed
    race_crosstab = race_crosstab.reindex(index=lines[::-1], columns=columns, fill_value=0)

    plt.figure(figsize=(12, 8))
    plt.imshow(race_crosstab, cmap='viridis', aspect='auto', norm=LogNorm())
    plt.colorbar(label='Number of Victims')
    plt.xticks(ticks=range(len(race_crosstab.columns)), labels=race_crosstab.columns, rotation=45, ha='right')
    plt.yticks(ticks=range(len(race_crosstab.index)), labels=race_crosstab.index)
    plt.title('Hate Crime Victims by Offender and Victim Race' + (f' in {year}' if year else ''))
    plt.xlabel('Victim Race')
    plt.ylabel('Offender Race')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + f'heatmap_offender_race_vs_victim{"_" + str(year) if year else ""}.png')
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
    plt.savefig(OUTPUT_DIR + 'histogram_presidential_terms.png', dpi=300, bbox_inches='tight')
    plt.show()

def boxplot_of_victims_per_crime(min_victims=12):
    df_clean = df.dropna(subset=['total_individual_victims'])
    
    #split offense_name by MULTIPLE_SEP and explode to have one offense per row
    df_clean = df_clean.assign(offense_name=df['offense_name'].str.split(MULTIPLE_SEP)).explode('offense_name')
    
    #keep only rows with min_victims or more victims
    df_clean = df_clean[df_clean['total_individual_victims'] >= min_victims]
    
    #keep only selected offense names to avoid clutter
    top_offenses = df_clean['offense_name'].value_counts().nlargest(6).index
    df_clean = df_clean[df_clean['offense_name'].isin(top_offenses)]
    
    plt.figure(figsize=(12, 8))
    
    # Create horizontal boxplot using matplotlib
    offense_names = df_clean['offense_name'].unique()
    data_by_offense = [df_clean[df_clean['offense_name'] == offense]['total_individual_victims'].values 
                      for offense in offense_names]
    
    # Create horizontal boxplot
    box_plot = plt.boxplot(data_by_offense, vert=False, patch_artist=True, labels=offense_names, medianprops=dict(color='black'))
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
    for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
        patch.set_facecolor(color)

    plt.title(f'Number of Victims in Crimes with {min_victims} or more Victims by Offense Name', fontsize=14)
    plt.xlabel('Number of Victims')
    plt.ylabel('Offense Name')
    plt.xscale('log')  # Use logarithmic scale for better visibility
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'boxplot_victims_per_crime_horizontal.png', dpi=300, bbox_inches='tight')
    plt.show()

def geomap_of_victims_by_state():
    """Create a choropleth map of hate crime victims by state"""
    try:
        # Use a reliable, simple data source - US Census Bureau's Cartographic Boundary Files
        # This is a stable URL that should work consistently
        states_url = "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip"
        
        print("Loading US states geographic data...")
        states = gpd.read_file(states_url)
        
        # Filter out territories, keep only continental US + Alaska + Hawaii
        states = states[~states['STUSPS'].isin(['PR', 'VI', 'MP', 'GU', 'AS'])]
        
    except Exception as e:
        print(f"Could not load Census data: {e}")
        print("Trying alternative approach with state abbreviations...")
        
        # If Census data fails, create a simple approach without maps
        # Just show the data in a bar chart format
        victims_by_state = df.groupby('state_abbr')['total_individual_victims'].sum().sort_values(ascending=False)
        
        plt.figure(figsize=(15, 10))
        victims_by_state.head(20).plot(kind='barh', color='darkred', alpha=0.7)
        plt.title('Top 20 States by Hate Crime Victims', fontsize=16)
        plt.xlabel('Total Victims')
        plt.ylabel('State')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR + 'hate_crimes_by_state_bar.png', dpi=300, bbox_inches='tight')
        plt.show()
        return
    
    # Aggregate victims by state
    print("Aggregating hate crime data by state...")
    victims_by_state = df.groupby('state_abbr')['total_individual_victims'].sum().reset_index()
    victims_by_state.columns = ['state_abbr', 'total_victims']
    
    # Merge with geographic data using state abbreviations
    print("Merging data with geographic boundaries...")
    states = states.merge(victims_by_state, left_on='STUSPS', right_on='state_abbr', how='left')
    states['total_victims'] = states['total_victims'].fillna(0)
    
    # Create the map
    print("Creating choropleth map...")
    fig, ax = plt.subplots(1, 1, figsize=(24, 16))
    
    # Prepare data for logarithmic scale (avoid log(0) by adding small value)
    states['total_victims_log'] = states['total_victims'].replace(0, np.nan)
    min_victims = states['total_victims_log'].min()
    max_victims = states['total_victims_log'].max()
    
    # Use logarithmic normalization
    norm = LogNorm(vmin=max(min_victims, 1), vmax=max_victims)
    
    # Plot the map with logarithmic normalization
    states.plot(column='total_victims_log', 
                cmap='OrRd', 
                linewidth=0.8, 
                ax=ax, 
                edgecolor='black',
                alpha=0.8,
                norm=norm)
    
    # Create a colorbar with logarithmic scale
    sm = plt.cm.ScalarMappable(cmap=OrRd, norm=norm)
    sm.set_array([])    # Position the colorbar at the bottom
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.046, pad=0.08)
    cbar.set_label('Total Hate Crime Victims (Log Scale)', fontsize=14, labelpad=10)
    cbar.ax.tick_params(labelsize=12)
    
    # Add state labels (abbreviations)
    for idx, row in states.iterrows():
        if row['geometry'] is not None and row['total_victims'] > 0:
            # Get the centroid for label placement
            centroid = row['geometry'].centroid
            ax.annotate(text=row['STUSPS'], 
                       xy=(centroid.x, centroid.y),
                       ha='center', va='center',
                       fontsize=10, fontweight='bold',
                       color='white', 
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7))
    
    # Set the map extent to focus on continental US
    ax.set_xlim(-130, -65)  # Longitude limits
    ax.set_ylim(20, 50)     # Latitude limits
    
    ax.set_title('Hate Crime Victims by State (1991-2024)', fontsize=20, pad=30, fontweight='bold')
    ax.axis('off')
    
    # Remove extra whitespace
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    plt.savefig(OUTPUT_DIR + 'geomap_victims_by_state.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    # Also create a summary table
    print("\nTop 10 states by total hate crime victims:")
    top_states = victims_by_state.nlargest(10, 'total_victims')
    for _, row in top_states.iterrows():
        print(f"{row['state_abbr']}: {int(row['total_victims']):,} victims")
    
    print(f"\nMap saved as: {OUTPUT_DIR}geomap_victims_by_state.png")

def geomap_evolution_by_decade():
    """Create a 4-panel subplot showing hate crime evolution across 1994, 2004, 2014, and 2024"""
    try:
        # Load US states shapefile
        states_url = "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip"
        print("Loading US states geographic data for evolution map...")
        states_base = gpd.read_file(states_url)
        states_base = states_base[~states_base['STUSPS'].isin(['PR', 'VI', 'MP', 'GU', 'AS'])]
        
    except Exception as e:
        print(f"Could not load geographic data for evolution map: {e}")
        return
    
    # Years to analyze
    years = [1994, 2004, 2014, 2024]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    # State name to abbreviation mapping
    state_name_to_abbr = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
        'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
        'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
        'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
        'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
        'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
        'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
        'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
        'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
        'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
        'District of Columbia': 'DC'
    }
    
    # Calculate global min/max for consistent color scale across all years
    all_year_data = []
    for year in years:
        year_data = df[df['data_year'] == year].groupby('state_abbr')['total_individual_victims'].sum()
        all_year_data.extend(year_data.values)
    
    global_min = max(1, min([x for x in all_year_data if x > 0]))  # Avoid log(0)
    global_max = max(all_year_data)
    norm = LogNorm(vmin=global_min, vmax=global_max)
    
    for i, year in enumerate(years):
        ax = axes[i]
        
        # Filter data for the specific year
        year_df = df[df['data_year'] == year]
        victims_by_state = year_df.groupby('state_abbr')['total_individual_victims'].sum().reset_index()
        victims_by_state.columns = ['state_abbr', 'total_victims']
        
        # Create a copy of the base states geodataframe
        states = states_base.copy()
        
        # Merge with year-specific data
        states = states.merge(victims_by_state, left_on='STUSPS', right_on='state_abbr', how='left')
        states['total_victims'] = states['total_victims'].fillna(0)
        states['total_victims_log'] = states['total_victims'].replace(0, np.nan)
        
        # Plot the map
        states.plot(column='total_victims_log', 
                   cmap='OrRd', 
                   linewidth=0.5, 
                   ax=ax, 
                   edgecolor='black',
                   alpha=0.8,
                   norm=norm,
                   missing_kwds={"color": "lightgrey", "alpha": 0.5})
        
        # Set map extent to focus on continental US
        ax.set_xlim(-130, -65)
        ax.set_ylim(20, 50)
        
        # Add title for each subplot
        total_victims = victims_by_state['total_victims'].sum()
        ax.set_title(f'{year}\nTotal Victims: {int(total_victims):,}', 
                    fontsize=16, fontweight='bold', pad=15)
        ax.axis('off')
    
    # Add a single colorbar for all subplots
    sm = plt.cm.ScalarMappable(cmap=OrRd, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', 
                       fraction=0.05, pad=0.08, shrink=0.8)
    cbar.set_label('Total Hate Crime Victims (Log Scale)', fontsize=14, labelpad=10)
    cbar.ax.tick_params(labelsize=12)
    
    # Overall title
    fig.suptitle('Evolution of Hate Crime Victims by State\n1994 • 2004 • 2014 • 2024', 
                fontsize=20, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.12)
    plt.savefig(OUTPUT_DIR + 'geomap_evolution_by_decade.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    # Print summary statistics
    print("\nHate Crime Evolution Summary:")
    print("-" * 40)
    for year in years:
        year_total = df[df['data_year'] == year]['total_individual_victims'].sum()
        print(f"{year}: {int(year_total):,} total victims")
    
    print(f"\nEvolution map saved as: {OUTPUT_DIR}geomap_evolution_by_decade.png")

if __name__ == "__main__":
    #victims_by_year()
    #victims_by_bias()
    #victims_by_bias(2024)
    #race_on_race()
    #race_on_race(2024)
    #victims_by_presidential_terms()
    #boxplot_of_victims_per_crime()
    #geomap_of_victims_by_state()
    geomap_evolution_by_decade()