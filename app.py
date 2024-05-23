import sklearn
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

df = pd.read_csv(r'dataset_part_2.csv')
df_1 = pd.read_csv(r'dataset_part_3.csv')
columns = ['FlightNumber', 'PayloadMass', 'Flights', 'GridFins', 'Reused', 'Legs', 'Block']
X = pd.DataFrame(df_1, columns=columns)

Y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=2, test_size=0.2)
scaler = preprocessing.StandardScaler()
scale = scaler.fit(X_train)
X_train_scaled = scale.transform(X_train)
X_test_scaled = scale.transform(X_test)

# converting the numpy array into dataframe
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

tree = DecisionTreeClassifier(criterion='gini', max_depth=18, max_features='log2',
                              min_samples_leaf=2, min_samples_split=2, splitter='random')
tree.fit(X_train_scaled, y_train)
st.title('SpaceX falcon9 landing outcome prediction using DecisionTreeClassifer')

features_df, original_df = st.tabs(['Features Dataframe', 'Original Dataframe with target'])
with features_df:
    st.subheader('Features used for prediction')
    st.write(df_1.iloc[:, 0:7])

with original_df:
    st.subheader('Dataframe with target variable class')
    st.write(df)


features_list = []


def get_features():
    flight_number = st.slider('Flight number', min_value=0, max_value=250, value=None)
    payload_mass = st.slider('Payload mass value(kg)', min_value=0, max_value=25000, value=None)
    flights = st.slider('Enter number of flights', min_value=0, max_value=50, value=None)
    grid_fins = st.selectbox('GridFins yes/no?(yes=1,no=0)', (0, 1), placeholder='Select a value')
    reused = st.selectbox('Was the first stage reused yes/no?(yes=1,no=0)', (0, 1), placeholder='Select a value')
    legs = st.selectbox('Legs yes/no?(yes=1,no=0)', (0, 1), placeholder='Select a value')
    blocks = st.slider('Enter number of blocks', min_value=0, max_value=20, value=None)
    features_list.extend((flight_number, payload_mass, flights, grid_fins, reused, legs, blocks))
    return features_list


features = get_features()

if 'clicked' not in st.session_state:
    st.session_state.clicked = False


def click_button():
    st.session_state.clicked = True
    predicted = tree.predict([features])
    return predicted

st.subheader('Click the button for predicting the landing of first stage!')
st.button('Make Prediction', on_click=click_button, type = "primary")

if st.session_state.clicked:
    # The message and nested widget will remain on the page
    prediction = click_button()
    output = ''
    if prediction[0] == 1:
        output = 'the first stage will land !'
    else:
        output = 'the first stage will not land'
    st.write(f'The predicted class value is {prediction[0]}, that is {output}')
