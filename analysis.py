import os
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

def bias_trends_over_time():
    """Create a sophisticated timeline showing how different bias categories have evolved"""
    # Categorize bias types into major groups
    bias_categories = {
        'Racial': ['Anti-Black or African American', 'Anti-White', 'Anti-Asian', 
                  'Anti-Native Hawaiian or Other Pacific Islander', 'Anti-American Indian or Alaska Native',
                  'Anti-Multiple Races, Group', 'Anti-Arab', 'Anti-Hispanic or Latino', 'Anti-Not Hispanic or Latino'],
        'Religious': ['Anti-Jewish', 'Anti-Islamic (Muslim)', 'Anti-Catholic', 'Anti-Protestant',
                     'Anti-Other Religion', 'Anti-Multiple Religions, Group', 'Anti-Atheism/Agnosticism',
                     'Anti-Orthodox (Russian, Greek, Other)', 'Anti-Other Christian', 'Anti-Mormon',
                     'Anti-Jehovah\'s Witness', 'Anti-Eastern Orthodox (Russian, Greek, Other)',
                     'Anti-Hindu', 'Anti-Buddhist', 'Anti-Sikh'],
        'Sexuality and Gender': ['Anti-Gay (Male)', 'Anti-Lesbian', 'Anti-Lesbian, Gay, Bisexual, or Transgender (Mixed Group)',
                              'Anti-Heterosexual', 'Anti-Bisexual', 'Anti-Transgender', 'Anti-Gender Non-Conforming'],
        'Disability': ['Anti-Physical Disability', 'Anti-Mental Disability']
    }
    
    # Create a figure with 2x2 subplots for 4 categories
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Color palette for different categories
    colors = ['#e74c3c', '#3498db', '#9b59b6', '#f39c12']
    
    for idx, (category, bias_list) in enumerate(bias_categories.items()):
        ax = axes[idx]
        
        # Filter and expand data for this category
        category_data = df[df['bias_desc'].isin(bias_list)]
        category_expanded = category_data.assign(bias_desc=category_data['bias_desc'].str.split(MULTIPLE_SEP)).explode('bias_desc')
        category_expanded = category_expanded[category_expanded['bias_desc'].isin(bias_list)]
        
        # Group by year and bias type
        yearly_data = category_expanded.groupby(['data_year', 'bias_desc'])['total_individual_victims'].sum().unstack(fill_value=0)
        
        # Create a mapping for long labels to shorter ones
        label_mapping = {
            'Anti-Lesbian, Gay, Bisexual, or Transgender (Mixed Group)': 'Anti-LGBT (Mixed)',
            'Anti-Eastern Orthodox (Russian, Greek, Other)': 'Anti-Eastern Orthodox',
            'Anti-Native Hawaiian or Other Pacific Islander': 'Anti-Pacific Islander',
            'Anti-American Indian or Alaska Native': 'Anti-Native American'
        }
        
        # Rename columns with shorter labels
        yearly_data.columns = [label_mapping.get(col, col) for col in yearly_data.columns]
        
        # Create stacked area plot
        yearly_data.plot(kind='area', ax=ax, alpha=0.7, color=plt.cm.Set3(np.linspace(0, 1, len(yearly_data.columns))))
        
        ax.set_title(f'{category} Bias Crimes Over Time', fontsize=13, fontweight='bold', pad=20)
        ax.set_xlabel('Year', fontsize=11)
        ax.set_ylabel('Victims', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        
        # Add trend line
        total_by_year = yearly_data.sum(axis=1)
        z = np.polyfit(total_by_year.index, total_by_year.values, 1)
        p = np.poly1d(z)
        ax.plot(total_by_year.index, p(total_by_year.index), "r--", alpha=0.8, linewidth=2)
    
    plt.suptitle('Evolution of Hate Crime Bias Categories (1991-2024)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(OUTPUT_DIR + 'bias_trends_over_time.png', dpi=300, bbox_inches='tight')
    plt.show()

def hate_crime_seasonality_heatmap():
    """Create a calendar heatmap showing seasonal patterns in hate crimes"""
    # Convert incident_date to datetime
    df_clean = df.dropna(subset=['incident_date'])
    df_clean['incident_date'] = pd.to_datetime(df_clean['incident_date'], errors='coerce')
    df_clean = df_clean.dropna(subset=['incident_date'])
    
    # Extract month and day of year
    df_clean['month'] = df_clean['incident_date'].dt.month
    df_clean['day_of_year'] = df_clean['incident_date'].dt.dayofyear
    df_clean['week_of_year'] = df_clean['incident_date'].dt.isocalendar().week
    
    # Create monthly heatmap
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Monthly aggregation
    monthly_crimes = df_clean.groupby(['data_year', 'month'])['total_individual_victims'].sum().unstack(fill_value=0)
    
    # Create heatmap
    im1 = ax1.imshow(monthly_crimes.values, cmap='Reds', aspect='auto')
    ax1.set_title('Hate Crimes by Month and Year', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Year', fontsize=12)
    ax1.set_xticks(range(12))
    ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax1.set_yticks(range(0, len(monthly_crimes), 5))
    ax1.set_yticklabels(monthly_crimes.index[::5])
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Total Victims', fontsize=12)
    
    # Weekly pattern within year
    weekly_pattern = df_clean.groupby('week_of_year')['total_individual_victims'].sum()
    ax2.plot(weekly_pattern.index, weekly_pattern.values, linewidth=3, color='darkred', alpha=0.8)
    ax2.fill_between(weekly_pattern.index, weekly_pattern.values, alpha=0.3, color='red')
    ax2.set_title('Hate Crimes by Week of Year (All Years Combined)', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Week of Year', fontsize=12)
    ax2.set_ylabel('Total Victims', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add annotations for notable peaks
    max_week = weekly_pattern.idxmax()
    max_value = weekly_pattern.max()
    ax2.annotate(f'Peak: Week {max_week}', 
                xy=(max_week, max_value), xytext=(max_week+5, max_value+max_value*0.1),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'hate_crime_seasonality.png', dpi=300, bbox_inches='tight')
    plt.show()

def regional_radar_comparison():
    """Create radar charts comparing hate crime patterns across US regions"""
    # Define US regions
    regions = {
        'Northeast': ['Connecticut', 'Maine', 'Massachusetts', 'New Hampshire', 'Rhode Island', 
                     'Vermont', 'New Jersey', 'New York', 'Pennsylvania'],
        'South': ['Delaware', 'Florida', 'Georgia', 'Maryland', 'North Carolina', 'South Carolina', 
                 'Virginia', 'District of Columbia', 'West Virginia', 'Alabama', 'Kentucky', 
                 'Mississippi', 'Tennessee', 'Arkansas', 'Louisiana', 'Oklahoma', 'Texas'],
        'Midwest': ['Illinois', 'Indiana', 'Michigan', 'Ohio', 'Wisconsin', 'Iowa', 'Kansas', 
                   'Minnesota', 'Missouri', 'Nebraska', 'North Dakota', 'South Dakota'],
        'West': ['Arizona', 'Colorado', 'Idaho', 'Montana', 'Nevada', 'New Mexico', 'Utah', 
                'Wyoming', 'Alaska', 'California', 'Hawaii', 'Oregon', 'Washington']
    }
    
    # Create bias categories for radar chart
    bias_categories = ['Anti-Black or African American', 'Anti-White', 'Anti-Hispanic or Latino', 'Anti-Asian',
                       'Anti-Catholic', 'Anti-Jewish', 'Anti-Islamic (Muslim)',
                       'Anti-Gay (Male)']
    
    # Calculate regional statistics
    regional_stats = {}
    
    for region, states in regions.items():
        region_df = df[df['state_name'].isin(states)]
        region_expanded = region_df.assign(bias_desc=region_df['bias_desc'].str.split(MULTIPLE_SEP)).explode('bias_desc')
        
        stats = []
        for bias in bias_categories:
            bias_count = region_expanded[region_expanded['bias_desc'] == bias]['total_individual_victims'].sum()
            stats.append(bias_count)
        regional_stats[region] = stats
    
    # Normalize data (0-1 scale for each bias type across regions)
    normalized_stats = {}
    for i, bias in enumerate(bias_categories):
        max_val = max([regional_stats[region][i] for region in regions.keys()])
        if max_val > 0:
            for region in regions.keys():
                if region not in normalized_stats:
                    normalized_stats[region] = []
                normalized_stats[region].append(regional_stats[region][i] / max_val)
        else:
            for region in regions.keys():
                if region not in normalized_stats:
                    normalized_stats[region] = []
                normalized_stats[region].append(0)
    
    # Create radar chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    for idx, (region, stats) in enumerate(normalized_stats.items()):
        ax = axes[idx]
        
        # Number of variables
        N = len(bias_categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add values
        values = stats + [stats[0]]  # Complete the circle
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, label=region, color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([bias.replace('Anti-', '').replace(' or ', '/') for bias in bias_categories], fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title(f'{region} Region\nHate Crime Bias Pattern', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True)
    
    plt.suptitle('Regional Comparison of Hate Crime Bias Types\n(Normalized Scale)', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'regional_radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def offender_victim_flow_analysis():
    """Create a sophisticated analysis of offender-victim demographic relationships"""
    # Filter data with both offender and victim information
    flow_df = df.dropna(subset=['offender_race']).copy()
    flow_df = flow_df[~flow_df['offender_race'].isin(['Unknown', 'Not Specified', 'Multiple'])]
    
    # Create bias-to-victim mapping
    bias_to_victim = {
        'Anti-Black or African American': 'Black/African American',
        'Anti-White': 'White',
        'Anti-Asian': 'Asian',
        'Anti-Hispanic or Latino': 'Hispanic/Latino',
        'Anti-Jewish': 'Jewish',
        'Anti-Islamic (Muslim)': 'Muslim'
    }
    
    # Filter for main bias categories
    flow_df = flow_df[flow_df['bias_desc'].isin(bias_to_victim.keys())].copy()
    flow_df['victim_group'] = flow_df['bias_desc'].map(bias_to_victim)
    
    # Expand offender race (handle multiple races)
    flow_expanded = flow_df.assign(offender_race=flow_df['offender_race'].str.split(MULTIPLE_SEP)).explode('offender_race')
    
    # Create cross-tabulation
    cross_tab = pd.crosstab(flow_expanded['offender_race'], flow_expanded['victim_group'], 
                           values=flow_expanded['total_individual_victims'], aggfunc='sum').fillna(0)
    
    # Create two visualizations: heatmap and chord-style plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Heatmap
    im = ax1.imshow(cross_tab.values, cmap='Reds', aspect='auto')
    ax1.set_title('Offender-Victim Demographic Matrix\n(Total Victims)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Victim Group', fontsize=12)
    ax1.set_ylabel('Offender Race', fontsize=12)
    ax1.set_xticks(range(len(cross_tab.columns)))
    ax1.set_xticklabels(cross_tab.columns, rotation=45, ha='right')
    ax1.set_yticks(range(len(cross_tab.index)))
    ax1.set_yticklabels(cross_tab.index)
    
    # Add text annotations
    for i in range(len(cross_tab.index)):
        for j in range(len(cross_tab.columns)):
            value = cross_tab.iloc[i, j]
            if value > 0:
                ax1.text(j, i, f'{int(value)}', ha='center', va='center', 
                        color='white' if value > cross_tab.values.max()/2 else 'black',
                        fontweight='bold', fontsize=10)
    
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Number of Victims', fontsize=12)
    
    # Chord-style visualization (simplified)
    # Calculate percentages for each offender group
    offender_totals = cross_tab.sum(axis=1)
    percentages = cross_tab.div(offender_totals, axis=0).fillna(0) * 100
    
    # Create stacked bar chart as chord alternative
    bottom = np.zeros(len(percentages))
    colors_chord = plt.cm.Set3(np.linspace(0, 1, len(cross_tab.columns)))
    
    for i, victim_group in enumerate(cross_tab.columns):
        ax2.bar(range(len(percentages)), percentages.iloc[:, i], bottom=bottom, 
               label=victim_group, color=colors_chord[i], alpha=0.8)
        bottom += percentages.iloc[:, i]
    
    ax2.set_title('Offender Group Targeting Patterns\n(Percentage Distribution)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Offender Race', fontsize=12)
    ax2.set_ylabel('Percentage of Victims by Group', fontsize=12)
    ax2.set_xticks(range(len(percentages)))
    ax2.set_xticklabels(percentages.index, rotation=45, ha='right')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'offender_victim_flow_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def hate_crime_story_timeline():
    """Create an innovative timeline that tells the story of hate crimes with key events"""
    # Key historical events that might correlate with hate crime spikes
    historical_events = {
        2001: "September 11 Attacks",
        2008: "Financial Crisis",
        2012: "Trayvon Martin Case",
        2015: "Charleston Church Shooting",
        2016: "Presidential Election",
        2017: "Charlottesville Rally",
        2019: "El Paso Shooting",
        2020: "COVID-19 & George Floyd",
        2021: "Capitol Riot"
    }
    
    # Calculate yearly totals and bias breakdowns
    yearly_totals = df.groupby('data_year')['total_individual_victims'].sum()
    
    # Get top bias categories by year
    df_expanded = df.assign(bias_desc=df['bias_desc'].str.split(MULTIPLE_SEP)).explode('bias_desc')
    yearly_bias = df_expanded.groupby(['data_year', 'bias_desc'])['total_individual_victims'].sum().unstack(fill_value=0)
    
    # Create the story timeline
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 16))
    
    # Main timeline
    ax1.plot(yearly_totals.index, yearly_totals.values, linewidth=4, color='darkred', alpha=0.8)
    ax1.fill_between(yearly_totals.index, yearly_totals.values, alpha=0.3, color='red')
    
    # Add event annotations
    for year, event in historical_events.items():
        if year in yearly_totals.index:
            y_val = yearly_totals[year]
            ax1.annotate(f'{year}: {event}', 
                        xy=(year, y_val), xytext=(year, y_val + yearly_totals.max() * 0.1),
                        arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                        fontsize=10, ha='center', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    ax1.set_title('The Story of Hate Crimes in America (1991-2024)\nTotal Victims Over Time with Historical Context', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Total Victims', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Bias evolution stacked area
    top_biases = ['Anti-Black or African American', 'Anti-White', 'Anti-Jewish', 'Anti-Islamic (Muslim)', 'Anti-Gay (Male)']
    bias_subset = yearly_bias[top_biases].fillna(0)
    
    ax2.stackplot(bias_subset.index, bias_subset.T, 
                 labels=top_biases, alpha=0.8,
                 colors=['#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#2ecc71'])
    
    ax2.set_title('Evolution of Major Bias Categories', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Victims', fontsize=12)
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax2.grid(True, alpha=0.3)
    
    # Moving average and volatility
    window = 3
    moving_avg = yearly_totals.rolling(window=window).mean()
    volatility = yearly_totals.rolling(window=window).std()
    
    ax3.plot(yearly_totals.index, yearly_totals.values, 'o-', alpha=0.6, label='Annual Totals', color='gray')
    ax3.plot(moving_avg.index, moving_avg.values, linewidth=3, label=f'{window}-Year Moving Average', color='darkblue')
    ax3.fill_between(moving_avg.index, 
                    moving_avg - volatility, moving_avg + volatility, 
                    alpha=0.2, label='Volatility Band', color='blue')
    
    ax3.set_title('Trend Analysis: Moving Average and Volatility', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Year', fontsize=12)
    ax3.set_ylabel('Victims', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'hate_crime_story_timeline.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    victims_by_year()
    victims_by_bias()
    victims_by_bias(2024)
    #race_on_race()
    #race_on_race(2024)
    victims_by_presidential_terms()
    boxplot_of_victims_per_crime()
    geomap_of_victims_by_state()
    geomap_evolution_by_decade()
    bias_trends_over_time()
    hate_crime_seasonality_heatmap()
    regional_radar_comparison()
    offender_victim_flow_analysis()
    hate_crime_story_timeline()