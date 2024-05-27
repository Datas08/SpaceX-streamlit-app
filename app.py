import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
import streamlit as st

df = pd.read_csv(r'dataset_part_2.csv')
X = pd.read_csv(r'dataset_part_3.csv')

Y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=2, test_size=0.2)

scaler = preprocessing.StandardScaler()
scale = scaler.fit(X_train)
X_train_scaled = scale.transform(X_train)
X_test_scaled = scale.transform(X_test)

# converting the numpy array into dataframe
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

svm = SVC(C=1.0, gamma=0.03162277660168379, kernel='sigmoid')
svm.fit(X_train_scaled, y_train)

st.title('SpaceX falcon9 landing outcome prediction using Support Vector Machine')

features_df, original_df = st.tabs(['Features Dataframe', 'Original Dataframe with target'])
with features_df:
    st.subheader('Features used for prediction')
    st.write(X)

with original_df:
    st.subheader('Dataframe with target variable class')
    st.write(df)

flight_no = st.slider('Flight number', min_value=0, max_value=250, value=None)
payload_mass = st.slider('Payload mass value(kg)', min_value=0, max_value=25000, value=None)
orbit_name = st.selectbox('Orbit',
                          ('LEO', 'ISS', 'PO', 'GTO', 'ES-L1', 'SSO', 'HEO', 'MEO', 'VLEO', 'SO', 'GEO', 'TLI'))
site_name = st.selectbox('Launch Site', ('CCSFS SLC 40', 'VAFB SLC 4E', 'KSC LC 39A', 'Kwajalein Atoll'))
serial = st.selectbox('Serial Number', ('B0003', 'B0005', 'B0007', 'B1003', 'B1004', 'B1005', 'B1006',
                                        'B1007', 'B1008', 'B1011', 'B1010', 'B1012', 'B1013', 'B1015',
                                        'B1016', 'B1018', 'B1019', 'B1017', 'B1020', 'B1021', 'B1022',
                                        'B1023', 'B1025', 'B1026', 'B1028', 'B1029', 'B1031', 'B1030',
                                        'B1032', 'B1034', 'B1035', 'B1036', 'B1037', 'B1039', 'B1038',
                                        'B1040', 'B1041', 'B1042', 'B1043', 'B1044', 'B1045', 'B1046',
                                        'B1047', 'B1048', 'B1049', 'B1050', 'B1054', 'B1051', 'B1056',
                                        'B1059', 'B1058', 'B1060', 'B1062', 'B1061', 'B1063', 'B1067',
                                        'B1069', 'B1052', 'B1071', 'B1073', 'B1072', 'B1077'))
pad_name = st.selectbox('Landing Pad', ('5e9e3032383ecb761634e7cb', '5e9e3032383ecb6bb234e7ca',
                                        '5e9e3032383ecb267a34e7c7', '5e9e3033383ecbb9e534e7cc',
                                        '5e9e3032383ecb554034e7c9', '5e9e3033383ecb075134e7cd'))
flights_count = st.slider('Enter number of flights', min_value=0, max_value=50, value=None)
grid_fins = st.selectbox('GridFins yes/no?(yes=1,no=0)', (0, 1), placeholder='Select a value')
reused_val = st.selectbox('Was the first stage reused yes/no?(yes=1,no=0)', (0, 1), placeholder='Select a value')
legs_no = st.selectbox('Legs yes/no?(yes=1,no=0)', (0, 1), placeholder='Select a value')
blocks = st.slider('Enter number of blocks', min_value=0, max_value=20, value=None)
reused_counts = st.slider('Reused Count', min_value=0, max_value=15, value=None)


def get_values(flight_number, payloadmass, orbitname, site, serial_no, pad, flights, gridfins, reused, legs,
                    block, reused_count):
    loc_orbit = np.where(X.columns == f'Orbit_{orbitname}')[0][0]
    loc_site = np.where(X.columns == f'LaunchSite_{site}')[0][0]
    loc_serial = np.where(X.columns == f'Serial_{serial_no}')[0][0]
    loc_pad = np.where(X.columns == f'LandingPad_{pad}')[0][0]

    x = np.zeros(len(X.columns))
    x[0] = flight_number
    x[1] = payloadmass
    x[2] = flights
    x[3] = gridfins
    x[4] = reused
    x[5] = legs
    x[6] = block
    x[7] = reused_count
    x[loc_orbit] = 1
    x[loc_site] = 1
    x[loc_serial] = 1
    x[loc_pad] = 1

    return x

values = get_values(flight_no, payload_mass, orbit_name, site_name, serial, pad_name, flights_count,
                                grid_fins, reused_val, legs_no, blocks, reused_counts)
values_scaled = scale.transform([values])
st.write(values_scaled)


if 'clicked' not in st.session_state:
    st.session_state.clicked = False


def click_button():
    st.session_state.clicked = True
    predicted = svm.predict(values_scaled)
    return predicted


st.subheader('Click the button for predicting the landing of first stage!')
st.button('Make Prediction', on_click=click_button, type="primary")

if st.session_state.clicked:
    # The message and nested widget will remain on the page
    prediction = click_button()
    output = ''
    if prediction == 1:
        output = 'the first stage will land !'
    else:
        output = 'the first stage will not land'
    st.write(f'The predicted class value is {prediction}, that is {output}')
