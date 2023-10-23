import streamlit as st
import joblib
import pandas as pd
# other necessary imports ...

# Importing functions from the notebook-converted script
from irwa_fine_tech import preprocess_text, extract_product_name, generate_response

# Load the trained model and data from the respective folders
intent_classifier = joblib.load('model/intent_classifier.pkl')
product_df = pd.read_csv('dataset/realistic_product_data.csv')
conversational_df = pd.read_csv('dataset/realistic_conversational_data.csv')


# Streamlit app interface
st.set_page_config(
    page_title="FineTech Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Design header
col1, col2, col3 = st.beta_columns([1,6,1])
with col2:
    st.image("image/logo1.png", width=200)
    st.title('FineTech Assistant 🤖')
    st.write("Welcome to FineTech! Your tech shopping assistant. Ask me anything!")

# Sidebar with company info and branding
st.sidebar.title("FineTech")
st.sidebar.image("image/logo2.png", width=200)
st.sidebar.write("FineTech is your one-stop destination for all tech products. From the latest smartphones to cutting-edge laptops, we've got it all!")
st.sidebar.write("Have questions? Ask our chatbot here or visit our [website](#)!")  # Replace # with your website link

# Chatbox interaction
chat_container = st.beta_container()

user_input = st.text_input("Type your question...")

if user_input:
    response = generate_response(user_input)
    with chat_container:
        st.markdown(f"💬 **You**: {user_input}")
        st.markdown(f"🤖 **FineTech Assistant**: {response}")

# A footer or any additional branding can be added here
st.write("---")
st.write("© 2023 FineTech - Bringing technology closer to you!")
st.write("[Privacy Policy](#) | [Terms of Service](#)")  # Replace # with actual links if available
