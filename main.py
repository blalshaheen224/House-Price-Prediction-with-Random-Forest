import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

path = '/content/AmesHousing.csv'

def load_data(path):
    df = pd.read_csv(path)
    return df

def explore_data(df):
    print('HEAD:')
    print(df.head())
    print('\nINFO:')
    print(df.info())
    print('\nDESCRIBE:')
    print(df.describe())
    print('\nMISSING VALUES:')
    print(df.isnull().sum())

def clean_data(df):
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            df[col]=df[col].fillna(df[col].mean())
        elif df[col].dtype == 'object':
            if not df[col].mode().empty:
              df[col] = df[col].fillna(df[col].mode()[0])
            else:
               df[col] = df[col].fillna("Unknown")
    df = df.drop_duplicates()
    return df

def preprocessing_data(df):
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df

def get_feature_importance(df):
    X = df.drop(['SalePrice'], axis=1)
    y = df['SalePrice']
    model = RandomForestClassifier()
    model.fit(X, y)
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    pd.set_option('display.max_rows', None)         
    pd.set_option('display.max_columns', None)      
    print(feature_importance_df)

def feature_engineering(df):
 cols_to_keep = [
    'Gr Liv Area', 'Lot Area', '1st Flr SF', 'Bsmt Unf SF',
    'Garage Area', 'Total Bsmt SF', 'Lot Frontage', 'Garage Yr Blt',
    'BsmtFin SF 1', 'Year Built', 'Mo Sold', 'Year Remod/Add',
    'Open Porch SF', 'Wood Deck SF', 'Mas Vnr Area', 'Neighborhood',
    'TotRms AbvGrd', 'Yr Sold', '2nd Flr SF', 'Exterior 2nd',
    'Exterior 1st', 'Overall Qual', 'BsmtFin Type 1', 'Overall Cond',
    'Bedroom AbvGr', 'Fireplace Qu', 'Bsmt Exposure', 'MS SubClass',
    'Garage Finish', 'SalePrice'
]
 df = df[cols_to_keep]
 return df

def model_training(df):
    if 'SalePrice' not in df.columns:
        print("❌ Column 'SalePrice' not found in dataset.")
        return None, None, None
    X = df.drop(['SalePrice'], axis=1)
    y = df['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
    print("Root Mean Squared Error (RMSE):", mean_squared_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # خط المثالية
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")
    plt.grid(True)
    plt.show()



def plot_top_features():
    x = ['Gr Liv Area', 'Lot Area', '1st Flr SF', 'Bsmt Unf SF',
         'Garage Area', 'Total Bsmt SF', 'Lot Frontage', 'Garage Yr Blt']

    y = [0.036150, 0.035952, 0.035239, 0.033235,
         0.033119, 0.032716, 0.032078, 0.028795]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=y, y=x, palette="viridis", hue=y )
    plt.title("Top Features Affecting SalePrice")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.legend().remove() 
    plt.tight_layout()
    plt.show()


def main():
    df = load_data(path)
    #explore_data(df)
    df = clean_data(df)
    df = preprocessing_data(df)
    df = feature_engineering(df)
    model, X_test, y_test = model_training(df)
    if model:
        evaluate_model(model, X_test, y_test)
        plot_top_features()
main()