Live,Interactable dashboard : https://droneflighteriksuzans.netlify.app

In drone operations, remaining battery life is a critical safety and operational constraint. Unexpected battery depletion can lead to aborted missions, lost equipment, or safety incidents. Currently, operators may rely on manufacturer estimates or general rules of thumb, but these do not account for the complex, real-world interplay of environmental conditions and mission-specific parameters. 

## Research Objective
The goal of this analysis is to quantify the relationship between remaining battery percentage and six key operational factors:
1. Flight Duration
2. Distance Flown
3. Actual Payload/Carry Weight
4. Altitude
5. Wind Speed
6. Drone Model
## Methodology
Using a dataset of historical drone flight logs (`DroneLog.csv`), the analysis employed:
*   **Data Cleaning:** Handling missing values, removing duplicates, and filtering out invalid sensor readings (e.g., negative weights, extreme altitude outliers).
*   **Exploratory Data Analysis (EDA):** Correlation heatmaps and scatter plots with linear regression lines to visualize relationships.

### Correlation Analysis
Before building any predictive models, we compute pairwise Pearson correlations 
between remaining battery and the five primary numeric flight parameters: flight 
duration, distance flown, actual carry weight, altitude, and wind speed. This 
correlation matrix serves two purposes: 
(1) it gives a first-pass indication of 
which factors are most strongly associated with battery depletion, and 
(2) it reveals multicollinearity among the predictors themselves — a critical issue that affects how we interpret feature importance in later modeling steps.
<img width="835" height="709" alt="image" src="https://github.com/user-attachments/assets/806bf2f7-4f0c-4c5b-9d55-c356cee932a5" />
<img width="357" height="215" alt="image" src="https://github.com/user-attachments/assets/ed18a988-1dcd-4f51-9aea-ee6453966c49" />
<img width="705" height="646" alt="image" src="https://github.com/user-attachments/assets/a50be55f-c6d9-4e17-b980-ad723ecdcbd9" />

## Key Findings & Conclusions
*   **Wind Speed is the Dominant Factor:** The Random Forest model identified wind speed as the overwhelming predictor of remaining battery. Scatter plots confirmed a steep negative linear relationship; higher wind speeds force the drone to expend significantly more energy to maintain stability.
*   **Distance and Duration are Strong Negative Drivers:** Longer flight times and greater distances consistently correlate with lower remaining battery. (Note: Their importance in the Random Forest was masked by multicollinearity with wind speed, but scatter plots confirm their independent impact).
*   **Payload Weight Reduces Battery Life:** Heavier payloads (e.g., 15-20 kg agricultural tanks) result in significantly lower battery reserves upon landing compared to light payloads.



## Optimal payload capacity

**Payload Utilization Analysis**

- **Battery Impact:** Drones operating at high payload utilization (>75% of max capacity) consistently finish flights with significantly lower battery reserves (~40-60% remaining) compared to lightly loaded drones (~95% remaining).

- **Mission Confounding:** Flight duration and drone model are heavily confounded with payload utilization. High-utilization flights are almost exclusively long-duration agricultural spraying missions (CropMaster), while low-utilization flights are short photography missions (SnapShot Mini).
<img width="1384" height="583" alt="image" src="https://github.com/user-attachments/assets/c9d44e65-c4b1-489f-81cc-d8fcc359ae56" />


- **Failure Rate:** Exceeding 100% payload utilization results in a 100% failure rate (3/3 flights aborted or landed unexpectedly). However, the sample size for overweight flights is extremely small, so this should be interpreted as a hard physical limit rather than a statistical trend.
<img width="602" height="535" alt="image" src="https://github.com/user-attachments/assets/74515ba4-a178-42f8-876b-3a7bfd0649b6" />


**Recommended Payload Policy:**

- **Utilization < 50%** — Safe for all missions, including wind and long duration.
- **Utilization 50-75%** — Normal operating range. Plan for shorter flights or higher wind.
- **Utilization 75-90%** — Only for short, low-wind, well-planned missions. Reserve 30% battery as landing margin.
- **Utilization 90-100%** — Emergency or one-off use only. Consider splitting the payload across two flights.
- **Utilization > 100%** — Never permitted. Guaranteed mission failure.


## **Predicted battery depletion curves by drone model.** 

Linear regression was fit to remaining battery versus flight duration for each drone model using completed flights with valid duration and battery readings. Extrapolating each trend to 0% yields predicted crash times ranging from **105 minutes (TrafficEye)** to **872 minutes (ViewMax 500)**, with VoltGuard (223 min), FlyHigh 300 (265 min), and EventFlyer (300 min) falling in between. 
The dashed orange line marks the 10% battery reserve threshold — the operational limit at which most drones initiate forced return-to-home or landing. Markers (X) indicate the predicted point of total battery depletion.

<img width="1382" height="784" alt="image" src="https://github.com/user-attachments/assets/5f772029-bcad-4548-9059-7c97673f1fa9" />

**Note:** Linear extrapolation of battery drain becomes unreliable beyond ~2× observed flight duration. Predictions beyond this horizon (e.g., CropMaster at 836 min) are statistical artifacts of the regression and do not reflect physical drone endurance. Real battery discharge is non-linear and accelerates below 20%.
<img width="1384" height="684" alt="image" src="https://github.com/user-attachments/assets/82641d3c-4983-4efd-a8d1-7ca4d0dd67eb" />



##  Limitations
###  Data Quality
- Duplicate drone IDs with conflicting records (D249 appears 6 times)
- Missing fields: Flight Duration (D030, D086), Actual Carry Weight (D027, D082, D240)
- Sensor errors: negative payloads (D029, D085), altitude 8444m (D516), wind 12.9 m/s (D819)
###  Analytical Limitations
- **Observational data**: Correlations ≠ causation.
- **Confounding**: Payload utilization is entangled with drone model and mission type.
- **Multicollinearity**: Wind, duration, and distance are correlated, complicating feature importance.
- **Linear extrapolation risk**: The crash-prediction graph extrapolates drain rates up to 20× beyond observed data. Predictions beyond ~2× observed duration should not be treated as physically meaningful.
- **Bounded target**: Battery Remaining ∈ [0, 100], so linear models are approximations.
- **Missing variables**: Battery age, temperature, humidity, time-of-day, and pilot behavior are not captured.
###  Sample Size Caveats
- Overweight flights: n = 3 → failure rate conclusion is directional, not statistical
- Several drone models: n < 10 → per-model drain rates are noisy


##  Conclusions & Recommendations
###  Summary of Findings 

1. Wind is the strongest predictor of remaining battery.
    
2. Duration and distance are strong negative drivers.
    
3. Payload utilization > 75% significantly reduces landing battery.
    
4. Exceeding 100% max carry weight results in guaranteed failure.
    
5. Random Forest predicts remaining battery with R² = [X].
    
6. Drone model does not independently predict battery once mission parameters are known.
    
### Operational Recommendations

- **Weather policy**: Postpone missions when wind > 6 m/s
    
- **Payload policy**: Target ≤75% utilization; 90%+ only for emergencies
    
- **Battery policy**: Reserve 20% minimum, 30% in adverse conditions
    
- **Hard limit**: Never exceed max carry weight

    
## Cleaning Deep Dive
The following subsections document every data cleaning decision made to the raw DroneLog.csv file, with the code used and the rationale behind each step. Each transformation is applied before any analysis, and the effect on the row count is tracked.


### Removing incomplete rows
Dropped any record missing critical identifying fields (Drone ID, Flight Date, or Battery Remaining (%)) — these fields are required for grouping and target-variable analysis.
```python
# Remove fully blank rows (the CSV contained stray empty lines)
df = df.dropna(how='all').copy()

# Remove rows missing critical fields
df = df.dropna(subset=['Drone ID', 'Flight Date', 'Battery Remaining (%)'])

# Strip whitespace from string columns
text_cols = ['Drone ID', 'Application', 'Drone Size', 'Drone Model',
             'Manufacturer', 'Payload Type', 'Operator ID',
             'Flight Status', 'Regulatory Approval ID', 'Notes']
for c in text_cols:
    df[c] = df[c].astype(str).str.strip()
```
Effect: Nullified 1 altitude outlier, 2 negative payloads, 1 wind outlier, 0 battery violations.


### Removing outliers and invalid sensor readings
Dropped or nullified physically impossible values that indicate sensor error or data entry mistakes.

```python
# Altitude > 1000m is beyond consumer drone legal limits (D516 = 8444m)
df.loc[df['Altitude (meters)'] > 1000, 'Altitude (meters)'] = np.nan

# Negative payload weight (D029, D085) — sensor error
df.loc[df['Actual Carry Weight (kg)'] < 0, 'Actual Carry Weight (kg)'] = np.nan

# Wind speed > 20 m/s exceeds safe drone operating limits (D819 = 12.9)
df.loc[df['Wind Speed (m/s)'] > 20, 'Wind Speed (m/s)'] = np.nan

# Battery must be in [0, 100]
df = df[df['Battery Remaining (%)'].between(0, 100)]
```
Effect: Nullified 1 altitude outlier, 2 negative payloads, 1 wind outlier, 0 battery violations.

### Removing duplicates
```python
# Key = all columns except Drone ID
key_cols = [c for c in df.columns if c != 'Drone ID']
df = df.drop_duplicates(subset=key_cols, keep='first')
```
Effect: Removed approximately 15 duplicate flight records (including the 6 repeat rows of D249 and the D058/D092/D093 etc. re-logged pairs).

### Flagging overweight flights instead of deleting them

```python
df['Overweight'] = df['Actual Carry Weight (kg)'] > df['Max Carry Weight (kg)']
print(df['Overweight'].value_counts())
```
Effect: Flagged 3 overweight flights (D026, D033, D084) for use in Section 3.



### Final row count

Raw file	819

Final dataset	~803 rows
