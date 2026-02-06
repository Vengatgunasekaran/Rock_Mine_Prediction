import streamlit as st
import numpy as np
import pickle
import pandas as pd

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

data = pd.read_csv("sonar.csv", header=None)

st.set_page_config(page_title="Rock vs Mine Predictor")

st.title("🪨 Rock vs 💣 Mine Prediction")

st.write("Enter 60 sonar signal values or try a random sample from the dataset.")

if "features" not in st.session_state:
    st.session_state.features = [0.0] * 60


if st.button("🎯 Try a random sample from dataset"):
    row = data.sample(1)

    st.session_state.features = row.iloc[0, :-1].tolist()
    st.session_state.true_label = row.iloc[0, -1]

    st.info(f"True label of this sample: {st.session_state.true_label}")


features = []

cols = st.columns(4)

for i in range(60):
    with cols[i % 4]:
        val = st.number_input(
            f"F{i+1}",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.features[i]),
            format="%.4f",
            key=f"f{i}"
        )
        features.append(val)


if st.button("Predict"):

    input_array = np.array(features).reshape(1, -1)

    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0]

    st.subheader("Result")

    if prediction == "R":
        st.success("🪨 Prediction : ROCK")
    else:
        st.error("💣 Prediction : MINE")

    st.write(f"Confidence : {np.max(probability)*100:.2f}%")

    st.write({
        "Mine (M)": float(probability[list(model.classes_).index("M")]),
        "Rock (R)": float(probability[list(model.classes_).index("R")])
    })
