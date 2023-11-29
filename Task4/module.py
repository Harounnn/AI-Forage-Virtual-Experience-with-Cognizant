from sklearn import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd

class Module():
    def __init__(self, n_splits: int = 5):
        # Defining constants
        self.n_splits = n_splits

    
    def load_data(path: str = '/path/to/csv_file') -> pd.DataFrame :
        '''
        This method is used to load the data into a dataframe

        :param      path : str (optional)
        :return     df : pd.DataFrame
        '''
        data = pd.read_csv(f'{path}')
        data.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
        return data

    def preprocess_data(data: pd.DataFrame = None, target: str = 'estimated_stock_pct') -> (pd.DataFrame, pd.Series):
        '''
        This method preprocess the data by splitting it to features and target then scaling it

        :param      data : pd.DataFrame
        :param      target : str (optional)
        :return     X : pd.DataFrame
                    y : pd.Series
        '''
        X, y = data.drop(columns=[target]), data[target]
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        return X,y

    def train_evaluate_model(X: pd.DataFrame = None, y: pd.Series = None) -> list:
        '''
        This method train our model and returns the training and validation scores in a list

        :param      X : pd.DataFrame
        :param      y : pd.Series
        :return     scores : list
        '''
        mse_train_scores = []
        mse_test_scores = []

        model = LinearRegression()

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train)

            train_predictions = model.predict(X_train)
            mse = mean_squared_error(y_train, train_predictions)
            mse_train_scores.append(mse)

            test_predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, test_predictions)
            mse_test_scores.append(mse)

        scores = [pd.mean(mse_train_scores), pd.mean(mse_test_scores)]
        return scores