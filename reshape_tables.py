# Kyle Hart
# Project: Commodities vs. Weather
#
# Description: Meant to reshape the tables from the World Bank and UN data.
#           Note, the original csv's were all loaded into the sqlite database 'countries.db'
#           The new tables are saved back into the database to be manually exported as csv's later

import sqlite3

#Connect to DB
conn = sqlite3.connect('data/countries.db')
cursr = conn.cursor()

# Get list of years, needed to select column values
years = [k for k in range(1961,2015)]

#CREATE crops schema
statement = "DROP TABLE IF EXISTS crops"
cursr.execute(statement)
statement = "CREATE TABLE crops( " \
            "   country TEXT, " \
            "   year INT, " \
            "   item TEXT, " \
            "   production DOUBLE, " \
            "   PRIMARY KEY (country, year, item) " \
            ")"
cursr.execute(statement)
#Load values into crops
for k in years:
    statement = "INSERT OR IGNORE INTO crops(country, year, item, production) " \
                "   SELECT country, {}, item, \"{}\" FROM crops_raw WHERE element=\"Production\"".format(k,k)
    cursr.execute(statement)
conn.commit()


#CREATE livestock schema
statement = "DROP TABLE IF EXISTS livestock"
cursr.execute(statement)
statement = "CREATE TABLE livestock( " \
            "   country TEXT, " \
            "   year INT, " \
            "   item TEXT, " \
            "   production DOUBLE, " \
            "   PRIMARY KEY (country, year, item) " \
            ")"
cursr.execute(statement)
#Load values into livestock
for k in years:
    statement = "INSERT OR IGNORE INTO livestock(country, year, item, production) " \
                "   SELECT country, {}, item, \"{}\" FROM livestock_raw WHERE element=\"Production\"".format(k,k)
    cursr.execute(statement)
conn.commit()


#Reset Years
years = [k for k in range(1968,2017)]
#CREATE gdp schema
statement = "DROP TABLE IF EXISTS gdp"
cursr.execute(statement)
statement = "CREATE TABLE gdp( " \
            "   country TEXT, " \
            "   year INT, " \
            "   amount DOUBLE " \
            ")"
cursr.execute(statement)
#Load values into gdp
for k in years:
    statement = "INSERT INTO gdp(country, year, amount) " \
                "   SELECT country, {}, \"{}\" FROM gdp_country_raw".format(k,k)
    cursr.execute(statement)

conn.commit()
conn.close()
