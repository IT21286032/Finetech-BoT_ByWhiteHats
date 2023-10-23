import streamlit as st
import joblib
import pandas as pd
from irwa_fine_tech import preprocess_text, extract_product_name, generate_response

# Load trained model and data
intent_classifier = joblib.load('model/intent_classifier.pkl')
product_df = pd.read_csv('dataset/realistic_product_data.csv')
conversational_df = pd.read_csv('dataset/realistic_conversational_data.csv')

# Streamlit app interface configuration
st.set_page_config(
    page_title="FineTech Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Design header
col1, col2, col3 = st.columns([1,6,1])
with col2:
    st.image("image/logo1.png", width=200)
    st.title('FineTech Assistant 🤖')
    st.write("Welcome to FineTech! Your tech shopping assistant. Ask me anything!")

# Sidebar with company info and branding
st.sidebar.title("FineTech")
st.sidebar.image("image/logo2.png", width=200)
st.sidebar.write("FineTech is your one-stop destination for all tech products. From the latest smartphones to cutting-edge laptops, we've got it all!")
st.sidebar.write("Have questions? Ask our chatbot here or visit our [website](#)!")  # Replace # with your website link

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    role, content = message["role"], message["content"]
    with st.chat_message(role):
        st.write(content)

# Accept user input
user_input = st.chat_input("Type your question...")
if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Generate bot response
    response = generate_response(user_input)
    
    # Display bot's response
    with st.chat_message("assistant"):
        st.write(response)
    
    # Add bot response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Footer branding
st.write("---")
st.write("© 2023 FineTech - Bringing technology closer to you!")
st.write("[Privacy Policy](#) | [Terms of Service](#)")  # Replace # with actual links if available
