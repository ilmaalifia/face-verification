import json

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
import os

st.title("Face Verification App")

name = st.text_input("Hi, what's your name?")

option = st.segmented_control(
    "Do you want to take a selfie or upload a selfie?",
    ("Take a Selfie", "Upload a Selfie"),
    default="Upload a Selfie",
)

is_duplicate = None
img_file_buffer = None
embedding = None
if option == "Take a Selfie":
    img_file_buffer = st.camera_input("")
else:
    img_file_buffer = st.file_uploader("", type=["jpg", "jpeg", "png"])
    if img_file_buffer:
        st.image(img_file_buffer)

if img_file_buffer:
    if st.button("Verify"):
        with st.spinner("Verifying ..."):
            img_bytes = img_file_buffer.getvalue()
            response = requests.post(
                f"{os.getenv('API_PATH')}/verify/",
                files={"img_file_buffer": img_bytes},
            )
            if response.status_code == 200:
                resp = response.json()
                if resp.get("error"):
                    st.error(f"Error: {resp['error']}")
                else:
                    is_duplicate = resp.get("is_duplicate", False)
                    if is_duplicate:
                        st.success("Duplicate image found in the database.")
                        for res in resp.get("results", []):
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.image(res["entity"]["file_path"], width=120)
                            with col2:
                                st.subheader(res["entity"]["name"])
                                st.caption(f"ID: {res['id']}")
                                st.write(f"Distance: {res['distance']:.4f}")
                    else:
                        st.info("No duplicate image found in the database.")
                        embedding = resp.get("embedding", [])
            else:
                st.error(
                    f"Failed to verify image. Status code: {response.status_code}."
                )
