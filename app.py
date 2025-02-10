import streamlit as st
import nltk
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('punkt')
nltk.download('stopwords')



chatbot = pipeline("text-generation", model="distilgpt2")



def healthcare_chatbot(user_input):
    if "symptom" in user_input:
        return "Please consult Doctor for accurate advice"
    elif "appointment" in user_input:
        return "Would you like to schedule appointment with the doctor? "
    elif "medication" in user_input:
        return "It's important to take prescribed medicines regularly. If you have concerns, consult your doctor."
    else:
        response = chatbot("question-answering", model="distilbert-base-uncased-distilled-squad")
    return response[0]['generated_text']

def main():
    st.title("Healthcare Assistant Chatbot")
    user_input = st.text_input("How can I assist you today?")
    if st.button("Submit"):
        if user_input:
            st.write("User : ",user_input)
            with st.spinner("Processing your query, please wait...."):
                response=healthcare_chatbot(user_input)
            st.write("Healthacare Assistant : ",response)
            print(response)
        else:
            st.write("Please enter a message to get response.")

if __name__ == "__main__":
    main()