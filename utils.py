import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple
sns.set_context('talk', font_scale=1)
sns.set_palette('Set1')
import warnings
warnings.filterwarnings('ignore')

def process_missing_and_duplicate_timestamps(filepath: str, verbose: bool=False)->pd.DataFrame:
    # This gist was created for the Kaggle dataset "Hourly Energy Consumption" which can be found at https://www.kaggle.com/robikscube/hourly-energy-consumption
    # Taking a look at the datasets, one can see that they are sorted by they are sorted by: year asc -> month desc -> day desc -> hour asc
    # There are also missing/duplicate values, which lead to offset by up to a day
    # This method sorts them properly and deals with missing/duplicate values using the averages (of the energy consumption)
    # Returns the processed dataframe

    df = pd.read_csv(filepath)
    df.sort_values('Datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)

    indices_to_remove = []
    series_to_add = []
    hour_counter = 1
    prev_date = ''

    if verbose:
        print(filepath)

    for index, row in df.iterrows():
        date_str = row['Datetime']

        year_str = date_str[0:4]
        month_str = date_str[5:7]
        day_str = date_str[8:10]
        hour_str = date_str[11:13]
        tail_str = date_str[14:]

        def date_to_str():
            return '-'.join([year_str, month_str, day_str]) + ' ' + ':'.join([hour_str, tail_str])

        def date_with_hour(hour):
            hour = '0' + str(hour) if hour < 10 else str(hour)
            return '-'.join([year_str, month_str, day_str]) + ' ' + ':'.join([hour, tail_str])

        if hour_counter != int(hour_str):
            if prev_date == date_to_str():
                # Duplicate datetime, we'll calculate the average and keep only one
                # Get the average
                average = int((df.iat[index, 1]+df.iat[index-1, 1])/2)
                df.iat[index, 1] = average
                # Dropping here will offset the index, so we do it after the for-loop
                indices_to_remove.append(index-1)
                if verbose:
                    print('Duplicate ' + date_to_str() +
                          ' with average ' + str(average))
            elif hour_counter < 23:
                # Missing datetime, we'll add it using the average of the previous and next for the consumption (MWs)
                average = int((df.iat[index, 1]+df.iat[index-1, 1])/2)

                # Adding here will offset the index, so we do it after the for-loop
                series_to_add.append(
                    pd.Series([date_with_hour(hour_counter), average], index=df.columns))
                if verbose:
                    print('Missing ' + date_with_hour(hour_counter) +
                          ' with average ' + str(average))
            else:
                # Didn't find any such cases in the Hourly Energy Consumption (PJM) (Kaggle) dataset
                # Leaving it for other datasets
                print(date_to_str() + ' and hour_counter ' +
                      str(hour_counter) + " with previous: " + prev_date)

            # Adjust for the missing/duplicate value
            if prev_date < date_to_str():
                hour_counter = (hour_counter + 1) % 24
            else:
                hour_counter = (
                    hour_counter - 1) if hour_counter - 1 > 0 else 0

        # Increment the hour
        hour_counter = (hour_counter + 1) % 24
        prev_date = date_str

    df.drop(indices_to_remove, inplace=True)
    #df = df.append(series_to_add)
    df = pd.concat(
        [df, pd.DataFrame(series_to_add, columns=df.columns)], ignore_index=True)

    # New rows are added at the end, sort them and also recalculate the indices
    df.sort_values('Datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def make_features(data: pd.DataFrame, max_lag: Optional[int] = None, rolling_mean_size: Optional[int] = None):
    """
    Creates features based on the previous values of the target variable.
    Adds rolling mean.

    Args:
        data (pd.DataFrame): Dataframe with the target variable.
        max_lag (int): Maximum lag. If None, no lag features are created.
        rolling_mean_size (int): Size of the rolling mean window in hours. If None, no rolling mean features are created.
    """    
    data = data.copy()
    data['dayofweek'] = data.index.dayofweek
    data['hour'] = data.index.hour
    data['doy'] = data.index.dayofyear

    if max_lag is not None:
        for lag in range(1, max_lag + 1):
            data['lag_{}'.format(lag)] = data['MW'].shift(lag)

    if rolling_mean_size is not None:
        data['rolling_mean'] = data['MW'].shift().rolling(rolling_mean_size).mean()
    return data


def load_data(filepath: str, max_lag: Optional[int] = 1, rolling_mean_size: Optional[int] = None) -> pd.DataFrame:
    """
    Loads the data and creates features.

    Args:
        filepath (str): Path to the csv file.
        max_lag (int): Maximum lag. If None, no lag features are created.
        rolling_mean_size (int): Size of the rolling mean window in hours. If None, no rolling mean features are created.
    """

    df = process_missing_and_duplicate_timestamps(filepath, verbose=False)
    df['dt'] = pd.to_datetime(df['Datetime'])
    df.set_index('dt', inplace=True)
    df.drop('Datetime', axis=1, inplace=True)
    df = df.resample('1H').sum() 
    df.columns = ['MW']

    df = make_features(df, max_lag, rolling_mean_size)
    df = df.dropna()
    return df


def mape(y_true: np.ndarray, y_pred: np.ndarray)->np.ndarray:
    """
    Calculates the mean absolute percentage error.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
    """
    return np.mean(np.abs((y_true - y_pred) / y_true))


def model_fit(data: pd.DataFrame, model: RandomForestRegressor , target_col: str = 'MW', test_size: float = 0.3, visualize: bool = True, diff_frac: float = 0.4)->Tuple[RandomForestRegressor, pd.DateOffset]:
    """
    Fits the model and calculates the MAPE.

    Args:
        data (pd.DataFrame): Dataframe with the target variable.
        model (RandomForestRegressor): Model to fit.
        target_col (str): Name of the target variable.
        test_size (float): Size of the test set.
        visualize (bool): Whether to visualize the results.
        diff_frac (float): Fraction of the data to use as a definition of outliers (for plotting)
    """




    train, test = train_test_split(
            data, shuffle=False, test_size=test_size )

    X_train = train.drop(target_col, axis=1)
    y_train = train[target_col]

    X_test = test.drop(target_col, axis=1)
    y_test = test[target_col]

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_score = mape(y_train, y_train_pred)
    test_score = mape(y_test, y_test_pred)

    print('Train MAPE: {}'.format(train_score))
    print('Test MAPE: {}'.format(test_score))



    train['pred'] = y_train_pred
    train['train'] = True
    test['pred'] = y_test_pred
    test['train'] = False


    df_res = pd.concat([train, test], axis=0)

    df_res['relative_error'] = (df_res['MW'] - df_res['pred']) / df_res['pred']

    if visualize:
        plt.figure(figsize=(9, 9))
        plt.plot(df_res.query('train')['MW'], df_res.query('train')['pred'], 'o', label='train', alpha = 0.5, color = 'C1')
        plt.plot(df_res.query('not train')['MW'], df_res.query('not train')['pred'], 'o', label='test', alpha = 0.5, color = 'C2')

        max_mw, min_mw = df_res['MW'].max(), df_res['MW'].min()
        ideal_line = np.linspace(min_mw, max_mw, 100)
        ideal_line_14 = ideal_line * (diff_frac + 1)
        ideal_line_06 = ideal_line * (1 - diff_frac)

        plt.plot(ideal_line, ideal_line, 'k:', alpha = 0.5, label = f'ideal +- {diff_frac*100}%')
        plt.plot(ideal_line, ideal_line_14, 'k--', alpha = 0.2)
        plt.plot(ideal_line, ideal_line_06, 'k--', alpha = 0.2)

        plt.xlabel('actual consumption, MW')
        plt.ylabel('predicted consumption, MW')
        plt.xlim(min_mw, max_mw)
        plt.ylim(min_mw, max_mw)
        
        plt.legend()
        plt.show()


        plt.figure(figsize=(10, 5))
        plt.plot(df_res.query('train')['MW'], label='train', alpha = 0.5, color = 'C1')
        plt.plot(df_res.query('train')['pred'], label='train pred', alpha = 0.5, color = 'C1', linestyle = '--')

        plt.plot(df_res.query('not train')['MW'], label='test', alpha = 0.5, color = 'C2')
        plt.plot(df_res.query('not train')['pred'], label='test pred', alpha = 0.5, color = 'C2', linestyle = '--')

        plt.xlabel('time')
        plt.ylabel('consumption, MW')
        plt.legend()
        plt.title('Actual and predicted values')
        plt.xticks(rotation=45)
        plt.show()


        #plot last week of train
        plt.figure(figsize=(10, 5))
        plt.plot(df_res.query('train').tail(7*24)['MW'], label='train', alpha = 0.5, color = 'C1')
        plt.plot(df_res.query('train').tail(7*24)['pred'], label='train pred', alpha = 0.5, color = 'C0', linestyle = '--')

        plt.xlabel('time')
        plt.ylabel('consumption, MW')
        plt.legend()
        plt.title('Last week of train')
        plt.xticks(rotation=45)


        #plot last week of test
        plt.figure(figsize=(10, 5))
        plt.plot(df_res.query('not train').tail(7*24)['MW'], label='test', alpha = 0.5, color = 'C2')
        plt.plot(df_res.query('not train').tail(7*24)['pred'], label='test pred', alpha = 0.5, color = 'C3', linestyle = '--')

        plt.xlabel('time')
        plt.ylabel('consumption, MW')
        plt.legend()
        plt.title('Last week of test')
        plt.xticks(rotation=45)


        plt.show()



    return model, df_res


default_model = RandomForestRegressor(n_estimators=700, max_depth=10, n_jobs=-1,bootstrap = False)



def find_outliers(df_res: pd.DataFrame,  visualize: bool = True, threshold: float = 0.5)->pd.DataFrame:
    """
    Finds outliers in the data.

    Args:
        df_res (pd.DataFrame): Dataframe with the target variable.
        visualize (bool): Whether to visualize the results.
        threshold (float): Threshold for outliers (in MAPE terms).
    """

    outliers = df_res.query(f'abs(relative_error) > {threshold}')
    outliers_dates = outliers.index.map(lambda t: t.date()).unique()

    if visualize:
        for date in outliers_dates:
            plt.figure(figsize=(10, 7))
            #idxs = df_res.index.map(lambda t: t.date()) == date
            #idxs are within 2 days
            idxs = (df_res.index.map(lambda t: t.date()) >= date - pd.Timedelta(days=2)) & (df_res.index.map(lambda t: t.date()) <= date + pd.Timedelta(days=2))

            actual = df_res[idxs]['MW']
            pred = df_res[idxs]['pred']

            outliers = df_res[idxs].query(f'abs(relative_error) > {threshold}')

            pred_low_th = pred - pred*threshold
            pred_high_th = pred + pred*threshold


            plt.plot(actual, label='actual', alpha = 0.5, color = 'C1')
            plt.plot(pred, label='predicted', alpha = 0.5, color = 'C0', linestyle = '--')
            plt.fill_between(df_res[idxs].index, pred_high_th, pred_low_th, alpha = 0.05, color = 'C0', label = f'{threshold*100}% outlier threshold')

            plt.plot(outliers['MW'], 'X', label='outliers', alpha = 0.5, color = 'r')

            plt.xlabel('time')
            plt.ylabel('MW')
            plt.legend()
            plt.title(f'Outliers on {date}')
            plt.xticks(rotation=45)
            plt.show()

    return outliers


def plot_date(df_res: pd.DataFrame, date_str: str, half_width_days: int = 3):
    """
    Plots the data for a given date.

    Args:
        df_res (pd.DataFrame): Dataframe with the target variable.
        date_str (str): Date to plot.
        half_width_days (int): Half-width of the plot in days.
    """
    
    date = pd.to_datetime(date_str)
    plt.figure(figsize=(10, 5))

    idxs = (df_res.index.map(lambda t: t.date()) >= date - pd.Timedelta(days=half_width_days)) & (df_res.index.map(lambda t: t.date()) <= date + pd.Timedelta(days=half_width_days))

    actual = df_res[idxs]['MW']
    pred = df_res[idxs]['pred']

    plt.plot(actual, label='actual', alpha = 0.5, color = 'C1')
    plt.plot(pred, label='predicted', alpha = 0.5, color = 'C0', linestyle = '--')

    plt.xlabel('time')
    plt.ylabel('MW')
    plt.legend()
    plt.title(f'{date} +- {half_width_days} days')
    plt.xticks(rotation=45)
    plt.show()
