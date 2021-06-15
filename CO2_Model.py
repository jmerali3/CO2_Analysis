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
    """Converts the feature (dates) and label (CO2) Pandas series to numpy then splits into train and test sets"""
    split_num = int(split_frac * len(features))
    features = features.to_numpy().reshape(-1, 1)
    labels = labels.to_numpy().reshape(-1, 1)
    train_date = features[0:split_num]
    test_date = features[split_num:]
    train_y = labels[0:split_num]
    test_y = labels[split_num:]
    return train_date, test_date, train_y, test_y


def calc_error(y, predicted_y):
    ''' Computes the residuals, root mean squared error, and mean absolute percentage error'''
    error_dict = {}
    resid = np.zeros([len(y), 1])
    for i, (y_i, yp_i) in enumerate(zip(y, predicted_y)):
        resid[i] = y_i - yp_i
    error_dict["resid"] = resid
    error_dict["rmse"] = metrics.mean_squared_error(y, predicted_y) ** .5
    error_dict["mape"] = 100 * metrics.mean_absolute_percentage_error(y, predicted_y)
    return error_dict


def interpolate1d_function(x, y, x_new):
    '''Interpolates between x and y and returns the y values associated with x_new'''
    f = interp1d(x, y)
    return f(x_new)


def linear_regression(x, test_x, y, test_y=None, error=False):
    '''Performs linear regression fitted on x and y and returns the model parameters and predicted y
    This will call the error function and return the error_dictionary if error=True
    '''
    lr = linear_model.LinearRegression()
    lr.fit(x, y)
    predicted_train_y = lr.predict(x)
    predict_test_y = lr.predict(test_x)
    linear_dict = {"coef":lr.coef_, "intercept": lr.intercept_,
                   "predicted_train_y": predicted_train_y, "predicted_test_y": predict_test_y}
    error_dict = calc_error(test_y, predict_test_y) if error else None
    return linear_dict, error_dict


def poly_regression(x, test_x, y, test_y=None, degree=2, error=False):
    '''Linearizes the values according to the degree of the polynomial and then calls linear regression
    This will call the error function and return the error_dictionary if error=True
    '''
    pr = PolynomialFeatures(degree=degree)
    x_poly = pr.fit_transform(x)
    test_x_poly = pr.fit_transform(test_x)
    poly_dict, _ = linear_regression(x_poly, test_x_poly, y, test_y)
    error_dict = calc_error(test_y, poly_dict["predicted_test_y"]) if error else None
    return poly_dict, error_dict


def plot_data(split_data_tuple, predictions, title):
    '''Plots the training and test ground truth data and the predicted data'''
    # TODO put input parameters in a tuple
    train_x, test_x, train_y, test_y = split_data_tuple
    pred_y_train, pred_y_test = predictions
    fig, ax = plt.subplots()
    ax.scatter(train_x + 1958 + 2 / 12, train_y, s=5, label="Training Data, Raw")
    ax.scatter(test_x + 1958 + 2 / 12, test_y, s=5, label="Test Data, Raw")
    ax.plot(train_x + 1958 + 2 / 12, pred_y_train, label="Predicted Training Data", c="royalblue", linewidth=.65)
    ax.plot(test_x + 1958 + 2 / 12, pred_y_test, label="Predicted Test Data", c="seagreen", linewidth=.65)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("CO2 (ppm)")
    ax.legend()
    save_title = title.split()[0:3]
    plt.savefig("CO2_Plots/" + "_".join(save_title))
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
    plt.savefig(f"CO2_Plots/{title.replace(' ', '_')}.png")
    plt.show()


def sub_plot_data(test_x, resid_quadratic, resid_final):
    '''Plots the residuals from after trend removal and after trend and periodic residual removal'''
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)
    ax[0].scatter(test_x, resid_quadratic, c="seagreen")
    ax[1].scatter(test_x, resid_final, c="seagreen")
    ax[0].set_title("Residuals After Trend Residual Removal")
    ax[1].set_title("Residuals After Trend & Periodic Residuals Removal")
    plt.savefig("CO2_Plots/Residuals.png")
    plt.tight_layout()
    plt.show()


def main():
    plot = True  # set to False to hide plots
    co2_df = load_data()
    split_data_tuple = split_data(co2_df["Date Index"], co2_df.CO2)
    train_date_index, test_date_index, train_y, test_y = split_data_tuple
    # date = feature (f), co2 = labels (l)

    ### Linear Regression ###
    linear_dict, linear_error_dict = linear_regression(*split_data_tuple, error=True)
    predictions_linear = linear_dict["predicted_train_y"], linear_dict["predicted_test_y"]
    title_linear = f"CO2 Linear Regression - RMSE:{round(linear_error_dict['rmse'],2)}"
    plot_data(split_data_tuple, predictions_linear, title_linear) if plot is True else None

    ### Quadratic Regression ###
    quadratic_dict, quadratic_error_dict = poly_regression(*split_data_tuple, degree=2, error=True)
    predictions_quadratic = quadratic_dict["predicted_train_y"], quadratic_dict["predicted_test_y"]
    title_quadratic = f"CO2 Quadratic Regression - RMSE: {round(quadratic_error_dict['rmse'],2)}"
    plot_data(split_data_tuple, predictions_quadratic, title_quadratic) if plot is True else None

    ### Cubic Regression ###
    cubic_dict, cubic_error_dict = poly_regression(*split_data_tuple, degree=3, error=True)
    predictions_cubic = cubic_dict["predicted_train_y"], cubic_dict["predicted_test_y"]
    title_cubic = f"CO2 Cubic Regression - RMSE: {round(cubic_error_dict['rmse'],2)}"
    plot_data(split_data_tuple, predictions_cubic, title_cubic) if plot is True else None

    ### Fit Periodic Signal ###
    periodic_error_dict = calc_error(quadratic_dict["predicted_train_y"], train_y)
    train_resid_quadratic = periodic_error_dict["resid"]
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
    coef = quadratic_dict["coef"].reshape(3)
    first_order, second_order = coef[1], coef[2]
    co2_df["predicted_y"] = co2_df["Date Index"] * first_order + \
                            co2_df["Date Index"] ** 2 * second_order + \
                            quadratic_dict["intercept"] + \
                            co2_df["Periodic Residual"]
    final_predictions = split_data(co2_df["Date Index"], co2_df["predicted_y"])
    final_error_dict = calc_error(test_y, final_predictions[-1])
    title_final = f"CO2 Complete Model - RMSE: {round(final_error_dict['rmse'], 2)}"
    plot_data(split_data_tuple, final_predictions[-2:], title_final) if plot is True else None

    ### Residual plots from before and after periodicc signal fitting ###
    sub_plot_data(test_date_index, quadratic_error_dict["resid"], final_error_dict["resid"]) if plot is True else None


if __name__ == "__main__":
    main()
