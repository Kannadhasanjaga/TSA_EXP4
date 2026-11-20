# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 3.11.26

NAME:KANNADHASAN J

REG NO:212224240071

DATASET: BMW SALES(2010-2024)



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

data = pd.read_csv("/content/BMW sales data (2010-2024).csv")

data.columns = data.columns.str.strip()

yearly = data.groupby('Year')['Sales_Volume'].sum()

X = yearly.values

years = yearly.index.values

N = 1000

plt.rcParams['figure.figsize'] = (12, 6)

# ============================
# Original Time Series
# ============================

plt.plot(years, X, marker='o')

plt.title('BMW Sales Volume')
plt.xlabel('Year')
plt.ylabel('Sales Volume')

plt.grid(True)
plt.show()


# ============================
# ACF & PACF
# ============================

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)

plot_acf(
    X,
    lags=min(10, len(X)//2 - 1),
    ax=plt.gca()
)

plt.title('ACF of BMW Sales')


plt.subplot(2, 1, 2)

plot_pacf(
    X,
    lags=min(5, len(X)//2 - 1),
    ax=plt.gca(),
    method='ywm'
)

plt.title('PACF of BMW Sales')

plt.tight_layout()
plt.show()


# ============================
# ARMA(1,1)
# ============================

arma11_model = ARIMA(
    X,
    order=(1, 0, 1)
).fit()

phi1 = arma11_model.arparams[0]
theta1 = arma11_model.maparams[0]

print("AR(1) Coefficient :", phi1)
print("MA(1) Coefficient :", theta1)


# Simulate ARMA(1,1)

ar1 = np.array([1, -phi1])
ma1 = np.array([1, theta1])

ARMA_1 = ArmaProcess(
    ar1,
    ma1
).generate_sample(
    nsample=N
)

plt.figure(figsize=(12, 5))

plt.plot(ARMA_1)

plt.title("Simulated ARMA(1,1) Process")

plt.xlim([0, 500])

plt.grid(True)

plt.show()


plot_acf(
    ARMA_1,
    lags=30
)

plt.title("ACF - Simulated ARMA(1,1)")

plt.show()


plot_pacf(
    ARMA_1,
    lags=30
)

plt.title("PACF - Simulated ARMA(1,1)")

plt.show()


# ============================
# ARMA(2,2)
# ============================

arma22_model = ARIMA(
    X,
    order=(2, 0, 2)
).fit()

phi1 = arma22_model.arparams[0]
phi2 = arma22_model.arparams[1]

theta1 = arma22_model.maparams[0]
theta2 = arma22_model.maparams[1]

print("\nAR(2) Coefficients :", phi1, phi2)
print("MA(2) Coefficients :", theta1, theta2)


# Simulate ARMA(2,2)

ar2 = np.array([
    1,
    -phi1,
    -phi2
])

ma2 = np.array([
    1,
    theta1,
    theta2
])

ARMA_2 = ArmaProcess(
    ar2,
    ma2
).generate_sample(
    nsample=N
)

plt.figure(figsize=(12, 5))

plt.plot(ARMA_2)

plt.title("Simulated ARMA(2,2) Process")

plt.xlim([0, 500])

plt.grid(True)

plt.show()


plot_acf(
    ARMA_2,
    lags=30
)

plt.title("ACF - Simulated ARMA(2,2)")

plt.show()


plot_pacf(
    ARMA_2,
    lags=30
)

plt.title("PACF - Simulated ARMA(2,2)")

plt.show()
```
OUTPUT:
SIMULATED ARMA(1,1) PROCESS:

<img width="992" height="448" alt="image" src="https://github.com/user-attachments/assets/ac81df0b-d2b8-4c3b-905f-1185afe950f2" />


Partial Autocorrelation

<img width="998" height="531" alt="image" src="https://github.com/user-attachments/assets/7dfb34a4-1e67-4430-bb8c-31258920f61b" />


Autocorrelation

<img width="1001" height="529" alt="image" src="https://github.com/user-attachments/assets/bb15540a-3171-40ba-bc16-59b0230b9141" />


SIMULATED ARMA(2,2) PROCESS:

<img width="991" height="450" alt="image" src="https://github.com/user-attachments/assets/f7b8d9bd-98e1-4f7c-a5e5-dad91e05b1eb" />


Partial Autocorrelation

<img width="998" height="530" alt="image" src="https://github.com/user-attachments/assets/2a4c37f8-0793-4d2b-9df6-d0cf2a39d927" />


Autocorrelation

<img width="1000" height="533" alt="image" src="https://github.com/user-attachments/assets/111d95f0-eb90-4c8d-ac34-17e0bd7abb37" />


RESULT:
Thus, a python program is created to fir ARMA Model successfully.
