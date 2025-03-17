
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model_name = "mistralai/Mistral-7B-v0.1"  # Change this to your preferred model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

st.title("Hugging Face LLM in Streamlit 🚀")

# User input
input_text = st.text_area("Enter your prompt here:")

if st.button("Generate"):
    if input_text.strip():
        with st.spinner("Generating..."):
            inputs = tokenizer(input_text, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=100)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.write("### Generated Response:")
            st.write(generated_text)
    else:
        st.warning("Please enter some text to generate a response!")

# Optional: Add a clear button
if st.button("Clear"):
    st.experimental_rerun()
