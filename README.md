# *[Experiment]* Bogor PM2.5 Air Quality Forecast

A very simplified prediction of today and tomorrow’s PM2.5 air quality in Bogor's city center. Bogor is located ~60km from Jakarta, Indonesia’s capital.

View dashboard here: **:bar_chart: [Bogor PM2.5 Air Quality Forecast](https://bogorairquality.streamlit.app)**
<br><br>
## Project motivation
Is the air quality in Bogor healthy for the upcoming day?

This experimental project explores that question using publicly available data from OpenAQ. The goal is to build a predictive model that forecasts the air quality based only on recent patterns, and compare it against the WHO (World Health Organization) PM2.5 air quality threshold.
<br><br>
## Potential value to users
- Help Bogor's residents decide if it is safe to engage in outdoor activities.
- Support Bogor's policymakers to make data-driven decisions to improve health and the environment.

## :warning: Disclaimer
This project is simply the author's initial exploration of building a machine learning (ML) model. The outputs are prone to bias and error. It is not intended to be advice.  
For your assurance, please read the full disclaimer displayed on the [Bogor PM2.5 Air Quality Forecast](https://bogorairquality.streamlit.app) dashboard.
<br><br>
## Data source
- OpenAQ PM2.5 sensor readings (aggregated daily)
[OpenAQ](http://OpenAQ.org) provides air quality data across the world using 1) reference-grade government monitors and 2) air sensors. Air quality is monitored real-time and the data is available in the OpenAQ Explorer as well as via API.  
In Indonesia, active and real-time sensors with plenty of historical data are very limited. One of the few air sensors that match this criteria is located in Bogor.
> city: Bogor Selatan  
location_id: 5599584  
sensor_id (PM2.5): 13986083  
data source: OpenAQ v3  
data period: 2015-08-29 - present (updates every hour)
> 
Using the [Sensors API](https://docs.openaq.org/resources/sensors), data on air pollutant readings and their concentration over time are retrieved. The API also provides endpoints on many other related data and measurements.
<br><br>
## Steps to reproduce
1. Find location_id of the air sensor station (location_id = 5599584)
2. Find sensor_if of the PM2.5 air pollutant measure (sensor_id = 13986083)
3. Decide which endpoint provides daily average data by an air sensor: https://docs.openaq.org/api/operations/sensor_daily_get_v3_sensors__sensors_id__days_get#200
4. Run a script to call the endpoint, and save the raw JSON response to `data/raw/` for traceability.
`python src/pull_openaq_days.py`
    - Convert the raw JSON into a simple time-series (date-value table), and save as CSV to `data/processed/`  for modeling later.
5. Create time-based features for the model/supervised learning table and save as CSV:
`python src/build_features.py`
    - `date`
    - `pm25` = the value to predict
    - `lag_1` = yesterday’s PM2.5
    - `lag_7` = pm25 7 days ago (same day last week)
    - `roll_mean_7` = mean(PM2.5 of last 7 days, excluding today) to reduce noise & stabilize predictions.
    - `day_of_week` (0–6) for human behavioral patterns/factors
    - `month` (1–12) for seasonality factors
6. Train model with Ridge Regression or RandomForestRegressor
`python src/train.py`
7. Evaluate model
`python src/evaluate.py`
8. Visualize the forecast using a Streamlit dashboard page.
9. Set up a simple warning if either of the predictions are above WHO’s PM2.5 threshold.
<br><br>
## Learnings & future work
### What I learned
- The importance of ensuring a robust dataset & model, especially through data cleaning & sanity checks (missingness, leakage, use only past values). Otherwise the analysis & prediction would be invalid.
- Even a forecast that may look “simple” require complex pythons.
### What I would improve on
- Firstly, ensure that the project will be impactful.
    - Problem framing: Involve citizens & understand their problems and find root cause.
    - Solution: What are citizens’ ideal conditions, explore several AI solutions then focus on delivering one, and understand what AI concerns citizens have.
- Secondly, ensure that the execution is feasible, through a more thorough feasibility check at the start.
- Include a baseline for the ML model’s minimum performance standard.
- Analyze the prediction score and improve the ML model used if necessary.
