import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from scipy.interpolate import interp1d
import calendar

plt.style.use("seaborn")


# The purpose of this program is to determine the optimal way to model CO2 concentration increases over time

def load_data():
    """Loads data from .csv file into pandas dataframe, creates a reference, and drops unused columns
    Reference is March 1958 because that is the first data point from the observatory
    """
    column_names = ["Year", "Month", "Date Excel", "Date", "CO2", "Seasonally Adjusted CO2 (ppm)", "Fit (ppm)",
                    "Seasonally Adjusted Fit (ppm)", "CO2 Filled (ppm)", "Seasonally Adjusted Filled (ppm)"]
    co2_df_load = pd.read_csv("CO2_copy.csv", names=column_names)
    reference = co2_df_load["Date"].iloc[0]  # Reference point is March 1958
    co2_df_load["Date Index"] = co2_df_load["Date"] - reference
    drop_cols = ["Seasonally Adjusted CO2 (ppm)", "Fit (ppm)", 'Date Excel', 'Date', "Seasonally Adjusted Fit (ppm)",
                 'CO2 Filled (ppm)', "Seasonally Adjusted Filled (ppm)"]
    co2_df_load = (co2_df_load.drop(labels=drop_cols, axis=1)
                   [co2_df_load.CO2 > 0] \
                   .reset_index(drop=True))
    return co2_df_load


def split_data(features, labels, split_frac=.8):
    """Converts the feature and label Pandas series to numpy then splits into train and test sets"""
    split_num = int(split_frac * len(features))
    features = features.to_numpy().reshape(-1, 1)
    labels = labels.to_numpy().reshape(-1, 1)
    train_f = features[0:split_num]
    test_f = features[split_num:]
    train_l = labels[0:split_num]
    test_l = labels[split_num:]
    return train_f, test_f, train_l, test_l


def calc_error(y, predicted_y):
    ''' Computes the residuals, root mean squared error, and mean absolute percentage error'''
    # TODO Add error_dict and return
    resid = np.zeros([len(y), 1])
    for i, (y_i, yp_i) in enumerate(zip(y, predicted_y)):
        resid[i] = y_i - yp_i
    rmse = metrics.mean_squared_error(y, predicted_y) ** .5
    mape = metrics.mean_absolute_percentage_error(y, predicted_y)
    return resid, rmse, mape * 100


def interpolate1d_function(x, y, x_new):
    '''Interpolates between x and y and returns the y values associated with x_new'''
    f = interp1d(x, y)
    return f(x_new)


def poly_regression(x, y, test_x, test_y=None, degree=2, error=False):
    '''Linearizes the values according to the degree of the polynomial and then calls linear regression
    This will call the error function and return the error_dictionary if error=True
    '''
    pr = PolynomialFeatures(degree=degree)
    x_poly = pr.fit_transform(x)
    test_x_poly = pr.fit_transform(test_x)
    coef, intercept, predicted_y, predict_test_y = linear_regression(x_poly, y, test_x_poly, test_y)
    # TODO remove if statement and put outputs in a dictionary
    # error_dict = calc_error(test_y, predict_test_y) if error else None
    # return coef, intercept, predicted_y, predict_test_y, error_dict
    if error:
        resid, rmse, mape = calc_error(test_y, predict_test_y)
        return coef, intercept, predicted_y, predict_test_y, resid, rmse, mape
    else:
        return coef, intercept, predicted_y, predict_test_y


def linear_regression(x, y, test_x, test_y=None, error=False):
    '''Performs linear regression fitted on x and y and returns the model parameters and predicted y
    This will call the error function and return the error_dictionary if error=True
    '''
    lr = linear_model.LinearRegression()
    lr.fit(x, y)
    predicted_y = lr.predict(x)
    predict_test_y = lr.predict(test_x)
    # TODO remove if statement and put outputs in a dictionary
    # error_dict = calc_error(test_y, predict_test_y) if error else None
    # return lr.coef_, lr.intercept_, predicted_y, predict_test_y, error_dict
    if error:
        resid, rmse, mape = calc_error(test_y, predict_test_y)
        return lr.coef_, lr.intercept_, predicted_y, predict_test_y, resid, rmse, mape
    else:
        return lr.coef_, lr.intercept_, predicted_y, predict_test_y


def plot_data(train_x, train_y, test_x, test_y, pred_y_train, pred_y_test, title, error):
    '''Plots the training and test ground truth data and the predicted data'''
    # TODO put input paramters in a dictionary
    # TODO remove title and error and savefig. Figures arent changing
    fig, ax = plt.subplots()
    ax.scatter(train_x + 1958 + 2 / 12, train_y, s=5, label="Training Data, Raw")
    ax.scatter(test_x + 1958 + 2 / 12, test_y, s=5, label="Test Data, Raw")
    ax.plot(train_x + 1958 + 2 / 12, pred_y_train, label="Predicted Training Data", c="royalblue", linewidth=.65)
    ax.plot(test_x + 1958 + 2 / 12, pred_y_test, label="Predicted Test Data", c="seagreen", linewidth=.65)
    ax.set_title(title + " - " + error)
    ax.set_xlabel("Year")
    ax.set_ylabel("CO2 (ppm)")
    ax.legend()
    # plt.savefig("CO2_Plots/" + title.replace(" ", "_") + ".png")
    plt.show()


def plot_cyclic(x, y):
    '''Plots the continuous monthly periodic trend by calling the interpolation function'''
    title = "Monthly Periodic Trend"
    fig, ax = plt.subplots()
    plt.xticks(rotation=45)
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Cyclic Residual CO2 (ppm)")
    ax.scatter(list(calendar.month_name), [None] + list(y), c='maroon')
    x_interp = np.arange(1, 12, .1)
    y_interp = interpolate1d_function(x, y, x_interp)
    ax.plot(x_interp, y_interp)
    fig.tight_layout()
    # plt.savefig("CO2_Plots/" + title.replace(" ", "_") + ".png")
    plt.show()


def sub_plot_data(test_x, resid_quadratic, resid_final):
    '''Plots the residuals from after trend removal and after trend and periodic residual removal'''
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)
    ax[0].scatter(test_x, resid_quadratic, c="seagreen")
    ax[1].scatter(test_x, resid_final, c="seagreen")
    ax[0].set_title("Residuals After Trend Residual Removal")
    ax[1].set_title("Residuals After Trend & Periodic Residuals Removal")
    # plt.savefig("CO2_Plots/Residuals.png")
    plt.show()


def main():
    plot = True  # set to False to hide plots
    co2_df = load_data()
    train_f, test_f, train_l, test_l = split_data(co2_df["Date Index"], co2_df.CO2)
    # date = feature (f), co2 = labels (l)

    ### Linear Regression ###
    (
        coef_linear, intercept_linear,  # Coefficient(s) and y intercept
        predicted_co2_train_linear, predicted_co2_test_linear,
        resid_linear, rmse_linear, mape_linear  # Error metrics on test data
    ) = linear_regression(train_f, train_l, test_f, test_l, error=True)
    title_linear = "CO2 Linear Regression"
    error_linear = f"RMSE: {round(rmse_linear, 2)}"
    plot_data(train_f, train_l, test_f, test_l, predicted_co2_train_linear, predicted_co2_test_linear,
              title_linear, error_linear) \
        if plot is True else None

    ### Quadratic Regression ###
    (
        coef_quadratic, intercept_quadratic,  # Coefficient(s) and y intercept
        predicted_co2_train_quadratic, predicted_co2_test_quadratic,
        resid_quadratic, rmse_quadratic, mape_quadratic  # Error metrics on test data
    ) = poly_regression(train_f, train_l, test_f, test_l, degree=2, error=True)
    title_quadratic = "CO2 Quadratic Regression"
    error_quadratic = f"RMSE: {round(rmse_quadratic, 2)}"
    plot_data(train_f, train_l, test_f, test_l, predicted_co2_train_quadratic, predicted_co2_test_quadratic,
              title_quadratic, error_quadratic) \
        if plot is True else None

    ### Cubic Regression ###
    (
        coef_cubic, intercept_cubic,  # Coefficient(s) and y intercept
        predicted_co2_train_cubic, predicted_co2_test_cubic,
        resid_cubic, rmse_cubic, mape_cubic  # Error metrics on test data
    ) = poly_regression(train_f, train_l, test_f, test_l, degree=3, error=True)
    title_cubic = "CO2 Cubic Regression"
    error_cubic = f"RMSE: {round(rmse_cubic, 2)}"
    plot_data(train_f, train_l, test_f, test_l, predicted_co2_train_cubic, predicted_co2_test_cubic,
              title_cubic, error_cubic) \
        if plot is True else None

    ### Fit Periodic Signal ###
    train_resid_quadratic, _, _ = calc_error(predicted_co2_train_quadratic, train_l)
    zero_resid = np.zeros([len(co2_df) - len(train_resid_quadratic), 1])
    resid_joined = np.concatenate([-train_resid_quadratic, zero_resid])
    co2_df["resid"] = resid_joined
    cyclic_avg = (
        (co2_df[co2_df["resid"] != 0])
        .groupby('Month') \
        .mean())["resid"] \
        .to_frame() \
        .rename(columns={"resid": "Periodic Residual"}) \
        .reset_index()
    co2_df = co2_df.merge(cyclic_avg, on="Month", how='left')
    plot_cyclic(cyclic_avg["Month"], cyclic_avg["Periodic Residual"]) if plot is True else None

    ### Final Model - Quadratic & Periodic Signals ###
    coef = coef_quadratic.reshape(3)
    first_order, second_order = coef[1], coef[2]
    co2_df["predicted_y"] = co2_df["Date Index"] * first_order + \
                            co2_df["Date Index"] ** 2 * second_order + \
                            intercept_quadratic + \
                            co2_df["Periodic Residual"]
    _, _, predicted_co2_train, predicted_co2_test = split_data(co2_df["Date Index"], co2_df["predicted_y"])
    resid_test_final, rmse_test_final, mape_test_final = calc_error(test_l, predicted_co2_test)
    title_final = "CO2 Periodic & Quadratic Regression"
    error_final = f"RMSE: {round(rmse_test_final, 2)}"
    plot_data(train_f, train_l, test_f, test_l, predicted_co2_train, predicted_co2_test, title_final, error_final) \
        if plot is True else None

    ### Residual plots from before and after periodicc signal fitting ###
    sub_plot_data(test_f, resid_quadratic, resid_test_final) if plot is True else None


if __name__ == "__main__":
    main()
