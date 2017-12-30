from cdo_api_py import Client
import sqlite3
from _datetime import datetime
import pandas as pd
from pprint import pprint


#Connect to DB
conn = sqlite3.connect('countries.db')
cursr = conn.cursor()

#Get FIPS codes
query = "SELECT * FROM noaa_ids"
cursr.execute(query)
data = cursr.fetchall()
fips_codes = [row[1] for row in data]

#Get CDO client
token = 'IwcNfdENuPAzCSYdQBXoxjkhqWiBxJpM'
client = Client(token, default_units='metric')

#Get weather data by country aggregate and push into single DataFrame
final_df = pd.DataFrame()
startdate = datetime(1961, 1, 1)
enddate = datetime(2017, 1, 1)
datatypeid=['TAVG', 'PRCP']
for idx, (country, code) in enumerate(data):
    print("\n\n", country, code)
    responses = list(client.get(
        'data',
        datasetid='GSOY',
        datatypeid=datatypeid,
        locationid=code,
        startdate=startdate,
        enddate=enddate
    ))

    results = client.squash_results(responses)
    # return_dataframe or not, the best way to grapple this data is with a dataframe
    df = client.results_to_dataframe(results).reset_index()
    if not df.empty and 'TAVG' in df.columns and 'PRCP' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = pd.DatetimeIndex(df['date']).year
        agg = df.groupby('year').mean().reset_index()
        agg['country'] = country
        agg = agg[['country', 'year', 'TAVG', 'PRCP']]
        pprint(agg)

        final_df = pd.concat([final_df, agg])


# Print to csv
final_df.to_csv('global_weather.csv', sep=',', index=False)