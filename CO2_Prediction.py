import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import CO2_Model
import itertools

# Train the model over the entire data set using quadratic and periodic regression.
# Predict CO2 concentration over the next n years

extrapolate_to_date = 2050  # will extrapolate until February of the selected year
start_year = 1958  # Do not change this variable

# Step 1: Load data, create training features and labels (until 2019), and create test features (future dates)
co2_df = CO2_Model.load_data()
train_f = co2_df["Date Index"].to_numpy().reshape(-1, 1)
train_l = co2_df["CO2"].to_numpy().reshape(-1, 1)
test_f = np.arange(train_f[-1] + 1/12, (extrapolate_to_date - start_year) + 1/12, 1/12).reshape(-1, 1)  # start year is 1958

# Step 2: Perform quadratic regression to obtain coefficients & intercept and predicted CO2
(
    coef_quadratic, intercept_quadratic,  # Coefficient(s) and y intercept
    predicted_co2_train_quadratic, predicted_co2_test_quadratic,
) = CO2_Model.poly_regression(train_f, train_l, test_f, test_y=None, degree=2, error=False)

# Step 3: Calculate the residuals based on training data and labels
train_resid_quadratic, _, _ = CO2_Model.calc_error(predicted_co2_train_quadratic, train_l)

# Step 4: Use residuals to calculated monthly averaged periodic trend. Merge back with original df
co2_df["resid"] = -train_resid_quadratic
cyclic_avg = (
    co2_df.groupby('Month')
    .mean()["resid"]
    .to_frame()\
    .reset_index()\
    .rename(columns={"resid":"Periodic Residual"}))

# Step 5: Create CO2 test dataframe with appropriate values for columns
co2_test_df = pd.DataFrame(test_f.reshape(-1, ), columns=["Date Index"])
co2_test_df["Month"] = [(i + 8) % 12 + 1 for i in range(1, len(co2_test_df) + 1)]  # Creates month labels for co2_test.
co2_test_df["CO2"] = np.NaN
co2_test_df["Year"] = [2019] * 3 + \
                      list(itertools.chain(*[list(itertools.repeat(i, 12)) for i in range(2020, extrapolate_to_date)])) + \
                      [extrapolate_to_date] * 2  # Creates year labels for co2_test

# Step 6: Merge CO2 & CO2 test dataframes
co2_df = co2_df.merge(cyclic_avg, on="Month", how='left').drop(labels='resid', axis=1)
co2_test_df = co2_test_df.merge(cyclic_avg, on="Month", how='left')
co2_df = co2_df.append(co2_test_df, ignore_index=True)


# Step 7: Calculate predicted CO2 based on quadratic & periodic regression
coef = coef_quadratic.reshape(3)
first_order, second_order = coef[1], coef[2]
co2_df["Predicted CO2 Concentration"] = co2_df["Date Index"] * first_order + \
                                        co2_df["Date Index"] ** 2 * second_order + \
                                        intercept_quadratic + \
                                        co2_df["Periodic Residual"]


# Step 8: Plot data
fig, ax = plt.subplots()
ax.scatter(co2_df["Date Index"] + start_year + 3/12, co2_df["CO2"], s=5, label="Training Data, Raw")
ax.plot(co2_df["Date Index"] + start_year + 3/12, co2_df["Predicted CO2 Concentration"], label="Predicted Data", c="seagreen", linewidth=.65)
ax.set_title(f"Extrapolated CO2 Concentration to {extrapolate_to_date}")
ax.set_ylabel("CO2 (ppm)")
ax.set_xticks(np.arange(1950, extrapolate_to_date+10, 10))
ax.legend()
# plt.savefig("CO2_Plots/Extrapolated_CO2_Concentration.png")
plt.show()
