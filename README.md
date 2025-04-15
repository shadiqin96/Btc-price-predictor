# Bitcoin Price Prediction with Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Dummy Bitcoin price data (yearly average)
data = {
    'Year': [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    'Price': [189, 523, 272, 567, 4340, 7092, 7203, 11192, 47153, 28945]
}
df = pd.DataFrame(data)

# Prepare data
X = df[['Year']]
y = df['Price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
future_years = pd.DataFrame({'Year': list(range(2023, 2031))})
predicted_prices = model.predict(future_years)

# Plotting
plt.figure(figsize=(10,6))
plt.plot(df['Year'], df['Price'], 'o-', label='Historical Prices')
plt.plot(future_years, predicted_prices, 'x--', label='Predicted Prices (2023-2030)')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Year')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Show prediction table
for year, price in zip(future_years['Year'], predicted_prices):
    print(f"Prediksi Harga Bitcoin Tahun {year}: ${price:,.2f}")

"""
README.md
---------
# Bitcoin Price Predictor

Skrip Python ini menggunakan regresi linier untuk memprediksi harga Bitcoin dari tahun 2023 hingga 2030 berdasarkan data rata-rata tahunan sebelumnya. Cocok sebagai proyek AI/ML pemula dan untuk penggemar cryptocurrency.

## Fitur:
- Dataset harga Bitcoin sederhana
- Visualisasi historis dan prediksi
- Model Linear Regression

## Cara Menjalankan:
1. Install dependensi:
```bash
pip install pandas numpy matplotlib scikit-learn
```
2. Jalankan script:
```bash
python bitcoin_predictor.py
```

## Output:
- Grafik harga historis dan prediksi
- Tabel harga prediksi tahun 2023-2030

## Lisensi:
MIT License

.gitignore
-----------
__pycache__/
*.pyc
.env
.DS_Store
.ipynb_checkpoints/
*.log
