import streamlit as st
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import easyocr

# Load the question-answering model and tokenizer
#model_name = "t5-base"
model_name = "google/flan-t5-base"
qa_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
qa_tokenizer = AutoTokenizer.from_pretrained(model_name)
reader = easyocr.Reader(['en'])

# Function to extract text from image using easyocr
def extract_text_from_image(image):
    # Perform OCR on the image using easyocr
    ocr_result = reader.readtext(image, detail=0)
    text = " ".join(ocr_result)
    return text

# Function to get response from the language model
def get_response_from_llm(extracted_text, question):
    # Prepare the input for the model
    input_text = f"question: {question} context: {extracted_text}"
    inputs = qa_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate the response
    with torch.no_grad():
        outputs = qa_model.generate(inputs, max_length=150, num_return_sequences=1)

    # Decode the response
    response = qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit App
st.set_page_config(page_title="Invoice Extractor")
st.header("Invoice Extractor")

# Sidebar for uploading image and entering question
st.sidebar.header("Upload Your Image")
uploaded_file = st.sidebar.file_uploader("Upload an invoice image...", type=["jpg"])
if uploaded_file is not None:
    image = None
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption="Uploaded Image.", use_column_width=True)
st.subheader("Enter your question about the invoice:")
question = st.text_input("")
submit = st.button("Generate Response")

if submit:
    if image is None:
        st.warning("Please upload an image.")
    else:
        extracted_text = extract_text_from_image(image)
        response = get_response_from_llm(extracted_text, question)
        st.subheader("Extracted Information:")
        st.write(response)
