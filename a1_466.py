# Copyright Shin Ren, 2019
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np


TIME = [i / 2 for i in range(1, 11)]
TODAY = pd.to_datetime("2019-Jan-07")
df = pd.read_csv("a1_data.csv")
selected_bonds = df[(df["Maturity"] == "2019-Mar-01") | 
					(df["Maturity"] == "2019-Sep-01") | 
					(df["Maturity"] == "2020-Mar-01") |
					(df["Maturity"] == "2020-Sep-01") | 
					(df["Maturity"] == "2021-Mar-01") | 
					(df["Maturity"] == "2021-Sep-01") |
					(df["Maturity"] == "2022-Mar-01") | 
					(df["Maturity"] == "2022-Sep-01") | 
					(df["Maturity"] == "2023-Mar-01") |
					(df["Maturity"] == "2023-Sep-01")].drop_duplicates(subset=["Maturity", "Day"]).reset_index(drop=True)


# Calculate Accured interest
selected_bonds["Maturity"] = pd.to_datetime(selected_bonds["Maturity"])
selected_bonds['Accured interest'] = 0

#if Maturity is Mar or Sep, n = 128 
selected_bonds['Accured interest'] = 128 / 365 * selected_bonds['Coupon']

# Calculate Dirty price
selected_bonds['Dirty price'] = selected_bonds['Accured interest'] + selected_bonds['Price']

# Using Bootstrap to calculate spot rate.
pre_sum = 0
pre_t = 0
r1 = 0
spot_rates = []
zcb_price = []
for index, row in selected_bonds.iterrows():
	if index % 10 == 0:
		pre_sum = 0
		pre_t = (row["Maturity"] - TODAY).days / 365
		# if index != 0:
		# 	TODAY += pd.DateOffset(1)

	t = (row["Maturity"] - TODAY).days / 365
	r = -math.log((row["Dirty price"] - row["Coupon"] * pre_sum / 2)/(100 + row["Coupon"] / 2)) / t
	spot_rates.append(r)
	zcb_price.append(math.exp(-r * t))
	pre_sum += math.exp(-r * pre_t)
	pre_t = t

selected_bonds["Spot rate"] = spot_rates
selected_bonds["ZCB price"] = zcb_price


# Calculate forward rate
TODAY = pd.to_datetime("2019-Jan-07")
forward_df = selected_bonds[selected_bonds.index.values % 2 == 1]
price1 = 0
time1 = 0
forward_rates = []
for index, row in forward_df.iterrows():
	if index % 10 == 1:
		price1 = row["ZCB price"]
		time1 = (row["Maturity"] - TODAY).days / 365
		# if index != 1:
		# 	TODAY += pd.DateOffset(1)
	else:
		forward_rate = -math.log(row["ZCB price"]/price1) / ((row["Maturity"] - TODAY).days / 365 - time1)
		forward_rates.append(forward_rate)

# Calculate Cov(Xi) where Xi,j = log(ri,j+1/ri,j), j = 1,...,9
cov_mat1 = np.empty([9, 5])
cov_mat2 = np.empty([1, 36])
for i in range(1, 10):
	all_ratio = np.log(selected_bonds.loc[selected_bonds['Day'] == i + 1, ['Yield']].values.reshape(1, -1)[0] / selected_bonds.loc[selected_bonds['Day'] == i, ['Yield']].values.reshape(1, -1)[0])
	cov_mat1[i - 1] = all_ratio[1::2]

ytm_cov = np.cov(cov_mat1.T)
eig_val1, eig_vec1 = np.linalg.eig(ytm_cov)
for i in range(36):
	cov_mat2[0][i] = forward_rates[i + 4] / forward_rates[i]

forward_cov = np.cov(cov_mat2.reshape(9, 4).T)
eig_val2, eig_vec2 = np.linalg.eig(forward_cov)

# Plot YTM
plt.figure()
for i in range(1, 11):
	day = selected_bonds[selected_bonds["Day"] == i]
	plt.plot(TIME, day["Yield"], marker='*')

plt.legend(["Day " + str(i) for i in range(1, 11)])
plt.xlabel("Time")
plt.ylabel("% Yield")
plt.title('YTM Curve')

# Plot spot rate
plt.figure()
for i in range(1, 11):
	day = selected_bonds[selected_bonds["Day"] == i]
	plt.plot(TIME, day["Spot rate"], marker='o')

plt.legend(["Day " + str(i) for i in range(1, 11)], loc="lower right", framealpha=0.3)
plt.xlabel("Time")
plt.ylabel("Rate")
plt.title('Spot Curve')

# Plot forward rate
plt.figure()
TIME1 = [i for i in range(1, 5)]
for i in range(0, 40, 4):
	plt.plot(TIME1, forward_rates[i:i+4])

axes = plt.gca()
axes.set_ylim([0.016,0.02])
plt.legend(["Day " + str(i) for i in range(1, 11)], loc="lower right", framealpha=0.3)
plt.xlabel("Time")
plt.ylabel("Rate")
plt.title('Forward Curve')
# plt.show()

selected_bonds.to_csv("Output.csv", encoding='utf-8', index=False)