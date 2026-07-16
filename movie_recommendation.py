import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import requests
import base64


def mycleaning(doc):
    return re.sub("[^a-zA-z0-9 ]","",doc).lower()

st.set_page_config(layout="wide")

def add_banner():
    with open("banner.jpg", "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>
        .banner {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            height: 250px;
            border-radius: 15px;
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 30px;
        }}

        .banner h1 {{
            color: white;
            font-size: 56px;
            font-weight: bold;
            background: rgba(0,0,0,0.55);
            padding: 15px 30px;
            border-radius: 12px;
            text-shadow: 2px 2px 8px black;
        }}
        </style>

        <div class="banner">
            <h1> Movie Recommendation System </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

add_banner()
st.sidebar.image("banner_2.jpg")
st.sidebar.title("About Project")
st.sidebar.write("Movie Recommendation System is a Machine Learning application that recommends similar movies based on user selection. The recommendation engine uses content-based filtering with cosine similarity to find movies having similar genres, keywords, cast and overview.")

st.sidebar.title("Contact Us 📱")
st.sidebar.write("abhisheksaini80610@gmail.com")

st.sidebar.image("github.png")
st.sidebar.write("https://github.com/saini1826")

st.sidebar.title("About us 👥")
st.sidebar.write("We are passionate Data Science and Machine Learning enthusiasts dedicated to building intelligent recommendation systems. Our goal is to create user-friendly AI applications that improve user experience through personalized recommendations.")

df = joblib.load("dataset.pkl")
model = joblib.load("movie_model.pkl")
tv = joblib.load("movie_vectorizer.pkl")

movie = st.selectbox("Select a movie ",df.name)
if movie:
    index = df[df.name==movie].index[0]
    vector = tv.transform([df.loc[index].values[2]])
    distances,indexes = model.kneighbors(vector,n_neighbors=6)

    movie_id = df.loc[indexes[0][1:]].movie_id.values
    movie_name = df.loc[indexes[0][1:]].name.values

    for m,i in zip(movie_name,movie_id):
        st.write(m)
        api = f"http://www.omdbapi.com/?i={i}&apikey=9ef47dcb"
        resp = requests.get(api)
        data = resp.json()
        poster=data["Poster"]
        st.image(poster)