# *[Experiment]* Bogor PM2.5 Air Quality Forecast

A rather simple 01-14 February prediction of PM2.5 air quality in Bogor's city center. Bogor is located ~60km from Jakarta, Indonesia’s capital.

View dashboard here: **:bar_chart: [Bogor PM2.5 Air Quality Forecast](https://bogorairquality.streamlit.app)**
<br><br>
## Project motivation
Is the air quality in Bogor healthy for the upcoming days?

This experimental project explores that question using publicly available data from OpenAQ. The goal is to build a predictive model that forecasts the air quality based on recent patterns, and also educate the WHO (World Health Organization) PM2.5 air quality threshold.
<br><br>
## Potential value to users
- Help Bogor's residents decide if it is safe to engage in outdoor activities.
- Support Bogor's policymakers to make data-driven decisions to improve health and the environment.
<br><br>
## :warning: Disclaimer
This project is simply the author's initial exploration of building a machine learning (ML) model. The outputs are prone to bias and error.  
The author used AI-assisted tools to generate parts of the code as part of his learning.  
For further assurance, please read the full disclaimer displayed on [Bogor PM2.5 Air Quality Forecast](https://bogorairquality.streamlit.app).
<br><br>
## Data source
- OpenAQ PM2.5 sensor readings (aggregated daily)

[OpenAQ](http://OpenAQ.org) provides air quality data across the world using air sensors and reference-grade government monitors. Air pollution is monitored real-time and accessible through the OpenAQ Explorer and retrievable via API.  
In Indonesia, active and real-time sensors with plenty of historical data are very limited. One of the few air sensors that match this criteria is located in Bogor.
> city: Bogor Selatan  
location_id: 5599584  
sensor_id (PM2.5): 13986083  
data source: OpenAQ v3  
> 
Using the [Sensors API](https://docs.openaq.org/resources/sensors), data on air pollutant readings and their concentration over time are retrieved.
<br><br>

## Learnings & future work
### What I learned when programming this
- Ensuring a robust dataset & model is important, especially through data cleaning & sanity checks (missingness, leakage, use only past values). Otherwise the analysis & prediction would be invalid.
- Even a forecast that may look simple require plenty of code.
### What I would improve on
- Firstly, ensure that the project will be impactful.
    - Problem framing: Involve citizens & understand their problems and find root cause.
    - Solution: What are citizens’ ideal conditions, explore several AI solutions then focus on delivering one, and understand what AI concerns citizens have.
- Secondly, ensure that the execution is feasible, through a more thorough feasibility check at the beginning.
- Include a baseline for the ML model’s minimum performance standard.
- Analyze the prediction score and improve the ML model used if necessary.
