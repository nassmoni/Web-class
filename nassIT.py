# import pandas as pd
# #Pandas automatically decompresses and reads the CSV inside the .zip file!
# df = pd.read_csv(titanic.zip)

# #Display the first 5 rows to make sure it loaded correctly
# print(df.head())


import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title("Iris Flower Prediction")
st.write("A simple model")
st.write("Select an option")

iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state= 42, test_size= 0.2)
model = RandomForestClassifier(n_estimators= 100, random_state=42)
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
st.write(f"Model accuracy: {accuracy:.0%}")
st.header("Confirm your flower")
sepal_length = st.slider("Sepal Lenght (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.0, 3.0 )
petal_lenght = st.slider("Petal Lenght (cm)", 1.0, 7.0 , 4.0)
petal_Width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.5)

features = [[sepal_length, sepal_width, petal_lenght, petal_Width]]

if st.button("Predict"):
    predict = model.predict(features)[0]
    st.success(f"prediction: {iris.target_names[predict]}")

predict = model.predict(features)[0]
st.write(predict)