# Prediction of US Forest Fire Severity

Forest fire severity in the United States is a pervasive issue with a potential cost to businesses, individuals, and municipalities of billions of dollars each year. In 2020, CoreLogic estimates that the combined reconstruction cost value of homes at risk in the United States is over 638 billion dollars. Due to the recent availability of each geographically referenced forest fire in the US in [this dataset](https://www.kaggle.com/rtatman/188-million-us-wildfires), it is feasible to search for and uncover the variables that determine the severity of those fire observations. 

In this project, python code is used to demonstrate how large amounts of sparsely related data may be used to infer correlation between certain variables and forest fire size in acres. With recursive addition of new data and refinement, these variables along with others may even be used to predict forest fire severity with reasonable accuracy.  

In a practical sense, the techniques outlined below demonstrate how a data analysis process is used to gather and clean immense volumes of data from seemingly unrelated sources to test a predictive model.

### Part 1: The Data Gathering Process

When contemplating which variables might explain forest fire size, one quickly discovers the intuitive question: Does the weather on the date a fire was observed influence its severity?  Fortunately, vast amounts of historical weather observations around the world are readily available to the public via [this NOAA dataset]( https://www.kaggle.com/noaa/gsod).

To make this data available for download in an efficient way, the National Oceanic and Atmospheric Administration hosts it with the Google Big Query API. With the following code, weather observations between 1992 and 2015 with a station reference table are programmatically queried and downloaded.

``` python
#define station query
query = ('SELECT * FROM bigquery-public-data.noaa_gsod.stations')
#use google big query client to retreive data and create noaa_df dataframe
noaa_df = bqclient.query(query).result().to_dataframe()
noaa_df.to_csv('weatherdata/stations.csv', index = False)

#creates master dataframe with all weather data
for year in tqdm_notebook(range(1992, 2016)):
    query = "SELECT * FROM `bigquery-public-data.noaa_gsod.gsod{}`".format(year)
    df = bqclient.query(query).result().to_dataframe()
    filename = 'weatherdata/US_weather_{}.csv'.format(year)
    df.to_csv(filename, index = False)
```

Next, [1.88 million US forest fire observations]( https://www.kaggle.com/rtatman/188-million-us-wildfires) are easily queried from an SQLite database with Python’s `sqlite3` package.

``` python
#create dataframe by querying Fires table from *.sqlite database
connection = sqlite3.connect('fire_database.sqlite')
firedf = pd.read_sql_query('''SELECT FIRE_YEAR,
                                     DISCOVERY_DOY,
                                     FIRE_SIZE,
                                     LATITUDE,
                                     LONGITUDE,
                                     STAT_CAUSE_DESCR,
                                     STATE
                              FROM Fires''', connection)
```

In both the fire observation dataset and weather station table, there are latitude and longitude reference columns. Reasonably, it is possible to find the nearest weather station to each fire observation, then join the weather event for the date each fire was observed. 

However, one issue that quickly arises is how to calculate each nearest weather station distance. Although it may be easy to write a function that finds the difference in each latitude and longitude coordinates, then finds a distance with a Pythagorean theorem calculation, the earth is round. Therefore, this approach may lead to errors in finding the actual nearest station. In turn, the “Haversine Distance” Formula is used to account for this.

![haversine distance formula](haversine_distance_formula.png)

With Python’s `math` package, trigonometric modules easily replicate this equation in the `dist()` function below. Since the `dist()` function simply calculates Haversine distances, it must be nested in another function to find the nearest weather station from all calculations. The `find_nearest()` function does this easily by finding the minimum value calculated in a table column.

``` python
#use Haversine Distance fromula to find distance from fires to stations
from math import radians, cos, sin, asin, sqrt
def dist(lat1, long1, lat2, long2):
    # convert decimal degrees to radians 
    lat1, long1, lat2, long2 = map(radians, [lat1, long1, lat2, long2])
    # haversine formula 
    dlon = long2 - long1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

#to match nearest stations to fires
def find_nearest(lat, long):
    distances = weatherwith_stations.apply(
        lambda row: dist(lat, long, row['lat'], row['lon']), 
        axis=1)
    return (weatherwith_stations.loc[distances.idxmin(), 'ref_key'], distances.min())
```

Once the two queries for data gathering and two functions for data joining are established, they are applied to scripts that loop through all the fire observations and weather data between 1992 and 2015. Even with the help of processing parallelization, this takes upwards of eight hours with the following three annotated code notebooks.

[First Notebook: NOAA Weather Data Gathering](https://github.com/Jeff-VA/Sample-Projects/blob/gh-pages/forest_fire_project/Step_1_NOAA_weather_data_gathering_notebook.ipynb)

[Second Notebook: Find Nearest Stations with the Haversine Distance Formula](https://github.com/Jeff-VA/Sample-Projects/blob/gh-pages/forest_fire_project/Step_2_Find_nearest_stations.ipynb)

[Third Notebook: Joining Weather Station Data](https://github.com/Jeff-VA/Sample-Projects/blob/gh-pages/forest_fire_project/Step_3_Join_stations_and_weather_data.ipynb)

Finally, after much preparation, a single table containing most proximate weather data for each of 1.88 million fire observations from between 1992 and 2015 in the United States is created.

![precleaning table output](precleaning_table.png)

### Part 2: The Data Cleaning Process

Even though all the data needed to build a predictive model for forest fire size prediction are readily available in one table, a few preliminary cleaning steps must be addressed.  Below, the cleaning steps are outlined briefly. To see the source code, [this link](‘https://github.com/Jeff-VA/Sample-Projects/blob/gh-pages/forest_fire_project/Step%204%20Data%20Cleaning%20and%20Preparation.ipynb’) navigates to an annotated Jupyter Notebook.

[Fifth Notebook: Analysis](https://github.com/Jeff-VA/Sample-Projects/blob/gh-pages/forest_fire_project/Step%205%20Analysis.ipynb)

First, to effectively interpret these data, an issue of null indicator values is addressed. With close inspection of the unclean table previewed a few lines above, one quickly notices that there are aberrant indicators for missing values in the weather data. For example, for the `dewp` and `slp` variables, `9999.9` represent missing data. In the following code block, these indicator values are all removed, then imputed with a median value for that variable. 

``` python
#Replace Null indicators with actual Null values
df['dewp'].replace({9999.9: None}, inplace=True)
df['slp'].replace({9999.9: None}, inplace=True)
df['visib'].replace({999.9: None}, inplace=True)
df['wdsp'].replace({999.9: None}, inplace=True)
df['mxpsd'].replace({999.9: None}, inplace=True)
df['gust'].replace({999.9: None}, inplace=True)
df['max'].replace({9999.9: None}, inplace=True)
df['min'].replace({9999.9: None}, inplace=True)
df['prcp'].replace({99.99: None}, inplace=True)
#create list of null variables
null_vars = ['dewp','slp','visib','wdsp','mxpsd','gust','max','min','prcp']
#ensure datatypes are float for variables with null values
for var in null_vars:
    df[var] = pd.to_numeric(df[var])
    
#loop through null variables and impute medians
for var in null_vars:
    df[var].fillna(df[var].median(), inplace = True)
#ensure success
```

``` python

```
