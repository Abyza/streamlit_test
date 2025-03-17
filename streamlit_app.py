import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Title of the app
st.title("GPT-2 Chatbot")

# Load the model and tokenizer with caching to avoid reloading
@st.cache_resource
def load_model():
    model_name = "gpt2"  # Works well with Streamlit's free tier
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Text input for user question
input_text = st.text_input("Ask something:")

# Button to generate response
if st.button("Generate"):
    if input_text.strip() != "":
        with st.spinner("Generating..."):
            try:
                # Tokenize input and generate response
                input_ids = tokenizer.encode(input_text, return_tensors="pt")
                output = model.generate(input_ids, max_new_tokens=50)
                response = tokenizer.decode(output[0], skip_special_tokens=True)
                # Display response
                st.write(response)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter some text!")

# Footer
st.markdown("---")
st.caption("Powered by GPT-2 and Streamlit")

