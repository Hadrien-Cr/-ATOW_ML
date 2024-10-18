import airportsdata
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

def compute_lon_lat(df_train, df_test):
    '''
    Computes for each airport, the corresponding lon, lat

    Adds 4 new columns to df_train and df_test:
        lon_adep: longitude of departure airport 
        lat_adep: latitude of departure airport 
        lon_ades: longitude of destination airport 
        lat_ades: latitude of destination airport 
    '''

    try:
        df_train[['adep', 'ades']]
        df_test[['adep', 'ades']]
        assert "lat" not in df_train.columns
        assert "lon" not in df_train.columns
    except:
        raise Exception('Column adep or ades not found in dataframe df_train')
    try:
        import airportsdata
    except:
        raise Exception('Module airportsdata not found')
    
    # Load the airports data as a dictionary
    airports_dict = airportsdata.load()
    airports = pd.DataFrame.from_dict(airports_dict, orient='index')

    airports_counts = pd.concat([
                                df_train.groupby('adep').size().reset_index(name = 'total_flights').set_index('adep'),
                                df_train.groupby('ades').size().reset_index(name = 'total_flights').set_index('ades'),
                                df_test.groupby('adep').size().reset_index(name = 'total_flights').set_index('adep'),
                                df_test.groupby('ades').size().reset_index(name = 'total_flights').set_index('ades'),

                                 ])
    airports_counts = airports_counts[~airports_counts.index.duplicated(keep='first')]

    airports_counts['lon'], airports_counts['lat'] = np.zeros(len(airports_counts)), np.zeros(len(airports_counts))

    for index, row in airports.iterrows():
        # Ensure lat/lon values are valid
        airport = row['icao']
        for (i, a) in enumerate(list(airports_counts.index)):
            if airport == a:
                lat = row['lat']
                lon = row['lon']
                airports_counts.iloc[i, airports_counts.columns.get_loc('lat')] = lat
                airports_counts.iloc[i, airports_counts.columns.get_loc('lon')] = lon


    # Add new columns
    df_train['lon_adep'] = df_train['adep'].map(airports_counts['lon'])
    df_train['lat_adep'] = df_train['adep'].map(airports_counts['lat'])
    df_train['lon_ades'] = df_train['ades'].map(airports_counts['lon'])
    df_train['lat_ades'] = df_train['ades'].map(airports_counts['lat'])

    df_test['lon_adep'] = df_test['adep'].map(airports_counts['lon'])
    df_test['lat_adep'] = df_test['adep'].map(airports_counts['lat'])
    df_test['lon_ades'] = df_test['ades'].map(airports_counts['lon'])
    df_test['lat_ades'] = df_test['ades'].map(airports_counts['lat'])


    print("-"*100)
    print("Columns for lon & lat: ['lon_adep', 'lat_adep', 'lon_ades', 'lat_ades'] successfully created !")
    print("-"*100)
      


def group_and_rename_airports(df_train, df_test):
    '''
    Performs a grouping of airport codes based on the number of flights

    If the airport is big enough, its code is unchanged
    For small airports, the airports are grouped by proximity and renamed to 30 different small airports (20 airports, SA0, SA1, ..., SA29)
    Same for tiny airports (renamed to AT0, AT1, ..., AT29)
    
    After this process, the columns adep and ades are modified accordingly
    '''

    try:
        df_train[['adep', 'ades']]
        df_test[['adep', 'ades']] 
        df_train[['lat_adep', 'lon_adep', 'lat_ades', 'lon_ades']]
        df_test[['lat_adep', 'lon_adep', 'lat_ades', 'lon_ades']]

    except:
        raise Exception('Column for adep/ades, or lon/lat not found in dataframe df_train')


    # Load the airports data 
    airports_dict = airportsdata.load()
    airports = pd.DataFrame.from_dict(airports_dict, orient='index')

    # Computes for each airport, the corresponding lon, lat
    airports_counts = df_train.groupby('adep').size().reset_index(name = 'total_flights').set_index('adep')
    airports_counts['lon'] = airports['lon']
    airports_counts['lat'] = airports['lat']
    airports_counts['lat'] = airports_counts['lat'].fillna(airports_counts['lat'].mean())
    airports_counts['lon'] = airports_counts['lon'].fillna(airports_counts['lon'].mean())

    # Separate airports into 3 groups
    tiny_airports = airports_counts[airports_counts['total_flights'] < 100]
    small_airports = airports_counts[(airports_counts['total_flights'] < 500) & (airports_counts['total_flights'] > 100)]
    big_medium_airports = airports_counts[airports_counts['total_flights'] > 500]
    
    # Grouping big/medium
    renaming_big_medium_airports = {name:name for name in big_medium_airports.index.unique()} # dont rename big and medium airports

    # handle tiny airports
    k_tiny_airports = 30
    names_tiny_airports = list(tiny_airports.index.unique())
    renaming_tiny_airports = {name:name for name in names_tiny_airports}
    
    X = np.array(tiny_airports[['lat', 'lon']])
    kmeans = KMeans(n_clusters=k_tiny_airports, random_state=0, n_init="auto").fit(X)

    for i in range(len(names_tiny_airports)):
        cluster_idx =  kmeans.labels_[i]
        renaming_tiny_airports[names_tiny_airports[i]] = f'TA{cluster_idx}'

    # handle small airports
    k_small_airports = 30
    name_small_airports = list(small_airports.index.unique())
    renaming_small_airports = {name:name for name in name_small_airports}

    X = np.array(small_airports[['lat', 'lon']])
    kmeans = KMeans(n_clusters=k_small_airports, random_state=0, n_init="auto").fit(X)

    for i in range(len(name_small_airports)):
        cluster_idx =  kmeans.labels_[i]
        renaming_small_airports[name_small_airports[i]] = f'SA{cluster_idx}'

    
    # Perform Renaming
    renaming = {}
    renaming.update(renaming_big_medium_airports)
    renaming.update(renaming_tiny_airports)
    renaming.update(renaming_small_airports)

    df_train['adep'] = df_train['adep'].replace(renaming)
    df_train['ades'] = df_train['ades'].replace(renaming)
    df_test['adep'] = df_test['adep'].replace(renaming)
    df_test['ades'] = df_test['ades'].replace(renaming)

    print("-"*100)
    print(f"Airports codes successfully grouped ! Different codes left : {len(df_train['adep'].unique())}")
    print("-"*100)

def group_and_rename_countries(df_train, df_test):
    '''
    Performs a grouping of country codes based on the number of flights

    If the country is big enough, its country code is unchanged
    For small countries, the countries are grounped by proximity and renamed to 10 different small countries (renamed to SC0, SC1, ..., SC9)
    Same for tiny countries (renamed to ST0, ST1, ..., ST9)
    
    After this process, the columns country_code_adep and country_code_ades are modified accordingly
    '''

    try:
        df_train[['country_code_adep', 'country_code_ades']]
        df_test[['country_code_adep', 'country_code_ades']] 
        df_train[['lat_adep', 'lon_adep', 'lat_ades', 'lon_ades']]
        df_test[['lat_adep', 'lon_adep', 'lat_ades', 'lon_ades']]

    except:
        raise Exception('Column for adep/ades, or lon/lat not found in dataframe df_train')

    # Load the airports data 
    airports_dict = airportsdata.load()
    airports = pd.DataFrame.from_dict(airports_dict, orient='index')
    countries_localisation = airports.groupby('country')[['lat','lon']].mean()
    
    # Computes for each airport, the corresponding lon, lat
    country_counts = df_train.groupby('country_code_adep').size().reset_index(name = 'total_flights').set_index('country_code_adep')
    country_counts['lon'] = countries_localisation['lon']
    country_counts['lat'] = countries_localisation['lat']
    country_counts = country_counts.fillna(0)

    tiny_countries = country_counts[country_counts['total_flights'] < 200]
    small_countries = country_counts[(country_counts['total_flights'] < 750) & (country_counts['total_flights'] > 200)]
    big_medium_countries = country_counts[country_counts['total_flights'] > 750]


    # Grouping big/medium
    renaming_big_medium_countries = {name:name for name in big_medium_countries.index.unique()} # dont rename big and medium airports

    # Grouping tiny airports to 10 countries
    from sklearn.cluster import KMeans

    k_tiny_countries = 10
    names_tiny_countries = list(tiny_countries.index.unique())
    renaming_tiny_countries = {name:name for name in names_tiny_countries}

    X = np.array(tiny_countries[['lat', 'lon']])
    kmeans = KMeans(n_clusters=k_tiny_countries, random_state=0, n_init="auto").fit(X)

    for i in range(len(names_tiny_countries)):
        cluster_idx =  kmeans.labels_[i]
        renaming_tiny_countries[names_tiny_countries[i]] = f'TC{cluster_idx}'

    # Grouping small countries to 10 countries
    k_small_countries = 10

    name_small_countries = list(small_countries.index.unique())
    renaming_small_countries = {name:name for name in name_small_countries}

    X = np.array(small_countries[['lat', 'lon']])
    kmeans = KMeans(n_clusters=k_small_countries, random_state=0, n_init="auto").fit(X)

    for i in range(len(name_small_countries)):
        cluster_idx =  kmeans.labels_[i]
        renaming_small_countries[name_small_countries[i]] = f'SC{cluster_idx}'

    # Perform Renaming
    renaming = {}
    renaming.update(renaming_big_medium_countries)
    renaming.update(renaming_tiny_countries)
    renaming.update(renaming_small_countries)

    df_train['country_code_adep'] = df_train['country_code_adep'].replace(renaming)
    df_train['country_code_ades'] = df_train['country_code_ades'].replace(renaming)
    df_test['country_code_adep'] = df_test['country_code_adep'].replace(renaming)
    df_test['country_code_ades'] = df_test['country_code_ades'].replace(renaming)

    print("-"*100)
    print(f"Country codes successfully grouped ! Different codes left : {len(df_train['country_code_adep'].unique())}")
    print("-"*100)


def test_lon_lat():
    
    """ 
    Test if all the columns are properly filled
    """
    
    train_df = pd.read_csv('data/challenge_set.csv')
    test_df = pd.read_csv('data/submission_set.csv')
    compute_lon_lat(train_df, test_df)
    assert train_df['lon_adep'].isnull().sum() == 0, train_df['lon_adep'].isnull().sum()
    assert train_df['lat_adep'].isnull().sum() == 0, train_df['lat_adep'].isnull().sum()
    assert train_df['lon_ades'].isnull().sum() == 0, train_df['lon_ades'].isnull().sum()
    assert train_df['lat_ades'].isnull().sum() == 0, train_df['lon_ades'].isnull().sum()
    assert test_df['lon_adep'].isnull().sum() == 0, test_df['lon_adep'].isnull().sum()
    assert test_df['lat_adep'].isnull().sum() == 0, test_df['lat_adep'].isnull().sum()
    assert test_df['lon_ades'].isnull().sum() == 0, test_df['lon_ades'].isnull().sum()
    assert test_df['lat_ades'].isnull().sum() == 0, test_df['lat_ades'].isnull().sum()


def test_codes():
    train_df = pd.read_csv('data/challenge_set.csv')
    test_df = pd.read_csv('data/submission_set.csv')
    compute_lon_lat(train_df, test_df)
    group_and_rename_countries(train_df, test_df)
    group_and_rename_airports(train_df, test_df)

    assert len(train_df['country_code_adep'].unique()) < 200
    assert len(train_df['country_code_ades'].unique()) < 200
    assert len(test_df['country_code_adep'].unique()) < 200
    assert len(test_df['country_code_ades'].unique()) < 200
    assert len(train_df['adep'].unique()) < 500
    assert len(train_df['ades'].unique()) < 500
    assert len(test_df['adep'].unique()) < 500
    assert len(test_df['ades'].unique()) < 500

if __name__ == '__main__':
    test_lon_lat()
    test_codes()