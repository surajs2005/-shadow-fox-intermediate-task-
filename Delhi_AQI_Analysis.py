import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
aqi_df = pd.read_csv('../data/Delhi_AQI_2015_2024.csv', low_memory=False)

# Normalize column names
for col in aqi_df.columns:
    if 'AQI' in col.upper():
        aqi_df.rename(columns={col: 'AQI'}, inplace=True)
    if 'date' in col.lower():
        aqi_df.rename(columns={col: 'date'}, inplace=True)
aqi_df['date'] = pd.to_datetime(aqi_df['date'], errors='coerce')
aqi_df.dropna(subset=['AQI', 'date'], inplace=True)

# Add features
aqi_df['month'] = aqi_df['date'].dt.month
aqi_df['season'] = aqi_df['month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Summer',
    6: 'Summer', 7: 'Monsoon', 8: 'Monsoon',
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
})

# Boxplot by season
sns.boxplot(x='season', y='AQI', data=aqi_df)
plt.title("Seasonal AQI Variation in Delhi")
plt.savefig('../images/seasonal_aqi_boxplot.png')
plt.show()

# Correlation heatmap
pollutants = [col for col in aqi_df.columns if any(p in col.upper() for p in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3'])]
aqi_df = aqi_df.dropna(subset=pollutants)
corr = aqi_df[pollutants + ['AQI']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Pollutants Correlation with AQI")
plt.savefig('../images/aqi_pollutants_correlation.png')
plt.show()

# Predict AQI
X = aqi_df[pollutants]
y = aqi_df['AQI']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)
preds = model.predict(X_test)
print("📉 RMSE:", round(mean_squared_error(y_test, preds, squared=False), 2))

# Pollutant importance
feat_imp = pd.Series(model.feature_importances_, index=pollutants)
feat_imp.sort_values().plot(kind='barh', title="Pollutant Importance in AQI Prediction")
plt.xlabel("Importance Score")
plt.savefig('../images/aqi_feature_importance.png')
plt.show()
