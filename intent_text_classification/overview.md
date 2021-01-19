## Prediction of US Forest FIre Severity

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
