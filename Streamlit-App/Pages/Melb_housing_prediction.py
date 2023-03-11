import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score

def preprocessData():
    global data
    print("Shape:", data.shape)
    print("Info:", data.info())
    print("Columns:", data.columns)
    # identify object type columns
    print(data.select_dtypes(['object']).columns)
    obj = ['Suburb', 'Address', 'Type', 'Method', 'SellerG', 'Date', 'CouncilArea', 'Regionname', 'Postcode']
    for i in obj:
        data[i] = data[i].astype('category')
        data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data.describe()
    data = data.drop(columns=['Bedroom2'])
    data.info()

    data['Age'] = 2017 - data['YearBuilt']
    data['Historic'] = np.where(data['Age'] >= 50, 'Historic', 'Contemporary')
    data['Historic'] = data['Historic'].astype('category')
    data.info()
    data.isnull().sum()
    data = data.dropna()
    data.isnull().sum()
    data = data[data['BuildingArea'] != 0]
    data.describe()


def showHeatmap():
    global data
    plt.figure(figsize=(10, 5))
    sns.histplot(data['Price'], kde=False)
    plt.figure(figsize=(20, 10))
    sns.heatmap(data.corr(numeric_only=True), cmap='coolwarm', linewidth=2, annot=True, annot_kws={"size": 11})
    plt.title('Melbourne Correlation')


def train(xTrain, xTest, yTrain, yTest):
    print("--------------")
    print("YTrain SHAPE: ", yTrain.shape)
    print("YTest SHAPE: ", yTest.shape)

    _teacher = LinearRegression()
    _learner = _teacher.fit(xTrain, yTrain)
    print('Coefficients: ', _learner.__getattribute__("coef_"))
    print('Intercept: ', _learner.__getattribute__("intercept_"))
    return _teacher, _learner


def predict(xTest, teacher):
    yPred = teacher.predict(xTest)
    return yPred


def showFeatureOutputPlot(xTrain, _learner):
    sns.set_style('darkgrid')
    f, axes = plt.subplots(4, 2, figsize=(20, 20))

    # Rooms vs Price
    axes[0, 0].plot(xTrain['Rooms'], _learner.coef_[0][0] * xTrain['Rooms'] + _learner.__getattribute__("intercept_")[0])
    axes[0, 0].set_title('Rooms vs Price')
    axes[0, 0].set_xlabel('Rooms')
    axes[0, 0].set_ylabel('Price')

    # Distance vs Price
    axes[0, 1].plot(xTrain['Distance'],
                    _learner.coef_[0][1] * xTrain['Distance'] + _learner.__getattribute__("intercept_")[0])
    axes[0, 1].set_title('Distance vs Price')
    axes[0, 1].set_xlabel('Distance')
    axes[0, 1].set_ylabel('Price')

    # Bathroom vs Price
    axes[1, 0].plot(xTrain['Bathroom'],
                    _learner.coef_[0][2] * xTrain['Bathroom'] + _learner.__getattribute__("intercept_")[0])
    axes[1, 0].set_title('Bathroom vs Price')
    axes[1, 0].set_xlabel('Bathroom')
    axes[1, 0].set_ylabel('Price')

    # Car vs Price
    axes[1, 1].plot(xTrain['Car'], _learner.coef_[0][3] * xTrain['Car'] + _learner.__getattribute__("intercept_")[0])
    axes[1, 1].set_title('Car vs Price')
    axes[1, 1].set_xlabel('Car')
    axes[1, 1].set_ylabel('Price')

    # Landsize vs Price
    axes[2, 0].plot(xTrain['Landsize'],
                    _learner.coef_[0][4] * xTrain['Landsize'] + _learner.__getattribute__("intercept_")[0])
    axes[2, 0].set_title('Landsize vs Price')
    axes[2, 0].set_xlabel('Landsize')
    axes[2, 0].set_ylabel('Price')

    # BuildingArea vs Price
    # axes[2,1].scatter(XTrain['BuildingArea'],YTrain)
    axes[2, 1].plot(xTrain['BuildingArea'],
                    _learner.coef_[0][5] * xTrain['BuildingArea'] + _learner.__getattribute__("intercept_")[0])
    axes[2, 1].set_title('BuildingArea vs Price')
    axes[2, 1].set_xlabel('BuildingArea')
    axes[2, 1].set_ylabel('Price')

    # Propertycount vs Price
    axes[3, 0].plot(xTrain['Propertycount'],
                    _learner.coef_[0][6] * xTrain['Propertycount'] + _learner.__getattribute__("intercept_")[0])
    axes[3, 0].set_title('Propertycount vs Price')
    axes[3, 0].set_xlabel('Propertycount')
    axes[3, 0].set_ylabel('Price')

    # Age vs Price
    axes[3, 1].plot(xTrain['Age'], _learner.coef_[0][7] * xTrain['Age'] + _learner.__getattribute__("intercept_")[0])
    axes[3, 1].set_title('Age vs Price')
    axes[3, 1].set_xlabel('Age')
    axes[3, 1].set_ylabel('Price')

    plt.show()


def showFeaturePredictionTestPlot(xTest, yPred, yTest):
    sns.set_style('darkgrid')
    _, axes = plt.subplots(4, 2, figsize=(20, 20))

    # Rooms vs yPred vs yTest
    axes[0, 0].scatter(xTest['Rooms'], yPred, color='red')
    axes[0, 0].scatter(xTest['Rooms'], yTest, color='blue')
    axes[0, 0].set_title('Rooms vs Pred and Act')
    axes[0, 0].legend(["yPred", "yTest"])
    axes[0, 0].set_xlabel('Rooms')
    axes[0, 0].set_ylabel('Price')

    # Distance vs yPred vs yTest
    axes[0, 1].scatter(xTest['Distance'], yPred, color='red')
    axes[0, 1].scatter(xTest['Distance'], yTest, color='blue')
    axes[0, 1].set_title('Distance vs Pred and Act')
    axes[0, 1].legend(["yPred", "yTest"])
    axes[0, 1].set_xlabel('Distance')
    axes[0, 1].set_ylabel('Price')

    # Bathroom vs yPred vs yTest
    axes[1, 0].scatter(xTest['Bathroom'], yPred, color='red')
    axes[1, 0].scatter(xTest['Bathroom'], yTest, color='blue')
    axes[1, 0].set_title('Bathroom vs Pred and Act')
    axes[1, 0].legend(["yPred", "yTest"])
    axes[1, 0].set_xlabel('Bathroom')
    axes[1, 0].set_ylabel('Price')

    # Car vs yPred vs yTest
    axes[1, 1].scatter(xTest['Car'], yPred, color='red')
    axes[1, 1].scatter(xTest['Car'], yTest, color='blue')
    axes[1, 1].set_title('Car vs Pred and Act')
    axes[1, 1].legend(["yPred", "yTest"])
    axes[1, 1].set_xlabel('Car')
    axes[1, 1].set_ylabel('Price')

    # Landsize vs yPred vs yTest
    axes[2, 0].scatter(xTest['Landsize'], yPred, color='red')
    axes[2, 0].scatter(xTest['Landsize'], yTest, color='blue')
    axes[2, 0].set_title('Landsize vs Pred and Act')
    axes[2, 0].legend(["yPred", "yTest"])
    axes[2, 0].set_xlabel('Landsize')
    axes[2, 0].set_ylabel('Price')

    # BuildingArea vs yPred vs yTest
    axes[2, 1].scatter(xTest['BuildingArea'], yPred, color='red')
    axes[2, 1].scatter(xTest['BuildingArea'], yTest, color='blue')
    axes[2, 1].set_title('BuildingArea vs Pred and Act')
    axes[2, 1].legend(["yPred", "yTest"])
    axes[2, 1].set_xlabel('BuildingArea')
    axes[2, 1].set_ylabel('Price')

    # Propertycount vs yPred vs yTest
    axes[3, 0].scatter(xTest['Propertycount'], yPred, color='red')
    axes[3, 0].scatter(xTest['Propertycount'], yTest, color='blue')
    axes[3, 0].set_title('Propertycount vs Pred and Act')
    axes[3, 0].legend(["yPred", "yTest"])
    axes[3, 0].set_xlabel('Propertycount')
    axes[3, 0].set_ylabel('Price')

    # Age vs yPred vs yTest
    axes[3, 1].scatter(xTest['Age'], yPred, color='red')
    axes[3, 1].scatter(xTest['Age'], yTest, color='blue')
    axes[3, 1].set_title('Age vs Pred and Act')
    axes[3, 1].legend(["yPred", "yTest"])
    axes[3, 1].set_xlabel('Age')
    axes[3, 1].set_ylabel('Price')

    plt.show()


def showScatterplot(yTest, yPred):
    plt.figure(figsize=(10, 5))
    plt.scatter(yTest, yPred)
    plt.show()


def showHistPlot(yTest, yPred):
    plt.figure(figsize=(10, 5))
    sns.histplot((yTest - yPred))


path = '/media/anmol/Study/GITHUB/Workshop_NIT_Meghalaya/Datasets/melbourne_housing/melb_data.csv'
data = pd.read_csv(path)
preprocessData()
showHeatmap()
print(data.select_dtypes(['float64', 'int64']).columns)

X = data[['Rooms', 'Distance', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Propertycount', 'Age']]
Y = data['Price']
XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.2, random_state=0)
print("XTrain SHAPE: ", XTrain.shape)
print("XTest SHAPE: ", XTest.shape)
YTrain = YTrain.values.reshape(4956, 1)
print("YTrain SHAPE: ", YTrain.shape)
YTest = YTest.values.reshape(1239, 1)
print("YTest SHAPE: ", YTest.shape)
teacher, learner = train(XTrain, XTest, YTrain, YTest)

YPred = predict(XTest,teacher)
print('Mean Absolute Error:', mean_absolute_error(YTest, YPred))
print('Mean Squared Error:', mean_squared_error(YTest, YPred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(YTest, YPred)))
print('R-Squared ERROR:', explained_variance_score(YTest, YPred))
showFeatureOutputPlot(XTrain, learner)
showFeaturePredictionTestPlot(XTest, YPred, YTest)
showScatterplot(YTest, YPred)
showHistPlot(YTest, YPred)




'''
def house_price_prediction():
    # Load the dataset
    data = pd.DataFrame({
        'bedrooms': [3, 2, 4, 3, 3, 2, 4, 3, 2, 3],
        'bathrooms': [2, 1, 3, 2, 2, 1, 3, 2, 1, 2],
        'sqft_living': [2000, 1000, 2500, 1800, 2000, 1200, 2800, 1800, 1000, 2000],
        'sqft_lot': [10000, 5000, 12500, 9000, 10000, 6000, 14000, 9000, 5000, 10000],
        'floors': [1, 1, 2, 1, 1, 1, 2, 1, 1, 1],
        'waterfront': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        'view': [0, 0, 2, 0, 0, 0, 3, 0, 0, 0],
        'condition': [3, 3, 4, 3, 3, 3, 5, 3, 3, 3],
        'grade': [7, 6, 8, 7, 7, 6, 9, 7, 6, 7],
        'yr_built': [1990, 1980, 2000, 1995, 1990, 1985, 2005, 1995, 1980, 1990],
        'price': [300000, 150000, 400000, 280000, 320000, 180000, 450000, 290000, 160000, 300000]
    })


    # Set up the sidebar
    st.sidebar.subheader('Input Features')
    bedrooms = st.sidebar.slider('Bedrooms', 1, 10, 3)
    bathrooms = st.sidebar.slider('Bathrooms', 1, 10, 2)
    sqft_living = st.sidebar.slider('Square Footage', 500, 10000, 2000)
    sqft_lot = st.sidebar.slider('Lot Size (in square feet)', 500, 50000, 10000)
    floors = st.sidebar.slider('Floors', 1, 5, 1)
    waterfront = st.sidebar.selectbox('Waterfront', [0, 1])
    view = st.sidebar.selectbox('View', [0, 1, 2, 3, 4])
    condition = st.sidebar.selectbox('Condition', [1, 2, 3, 4, 5])
    grade = st.sidebar.selectbox('Grade', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    yr_built = st.sidebar.slider('Year Built', 1900, 2022, 2000)

    # Create a dictionary with the input values
    input_data = {
        'rooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft_living,
        'sqft_lot': sqft_lot,
        'floors': floors,
        'waterfront': waterfront,
        'view': view,
        'condition': condition,
        'grade': grade,
        'yr_built': yr_built
    }

    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    # Load the model
    model = LinearRegression()
    model.fit(data.drop('price', axis=1), data['price'])

    # Make a prediction
    prediction = model.predict(input_df)

    # Display the prediction
    st.subheader('Predicted Price')
    st.write('${:,.2f}'.format(prediction[0]))

    # Set up the title and the description
    st.title('Housing Price Prediction')
    st.write('This app predicts the price of a house based on its features.')

    # Display the dataset
    st.subheader('Dataset')
    st.write(data)
    '''
