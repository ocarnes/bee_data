# README
## Intro ##
This project was completed to provide recommendations for analyzing temperature data at specific hive locations throughout the country.

### Contents ###
1. Data cleaning
2. Modeling
3. Analysis
4. Recommendations

## Data Cleaning ##
Data cleaning involved importing a csv file in addition to scraping some info from [NOAA](https://www.ncdc.noaa.gov/cdo-web/webservices/v2#datasets) and [Weather Underground](https://www.wunderground.com/history/) for additional climate data based on the zip codes provided. (Note: The [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) python package was used to parse text scrapped
from these sites).
It was assumed that hive data was recorded either a single location within a zip code or with multiple locations spread across a single zip code. The graph below reflects all of the different hives recording data throughout the year long test period.
[Hive Data by Zip](/img/Hive_Data_by_Zip.png).
Once Nan columns were eliminated and the appropriate additional weather data was merged, preliminary modeling could begin.
