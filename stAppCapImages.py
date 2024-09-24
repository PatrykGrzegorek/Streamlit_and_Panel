import streamlit as st
from transformers import BlipForConditionalGeneration, AutoProcessor
from PIL import Image
import torch

st.set_page_config(page_title="Image Caption Generator", layout="centered")

@st.cache_resource
def load_model():
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return model, processor

model, processor = load_model()

st.title("Image Caption Generator")
st.write("Upload an image, and the model will generate its caption.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        pass
    with col2:
        st.image(image, caption="Uploaded image", width=300)
    with col3:
        pass
    inputs = processor(images=image, return_tensors="pt").to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    with st.spinner('Generating caption...'):
        out = model.generate(**inputs)
        description = processor.decode(out[0], skip_special_tokens=True)
    st.markdown("### Image Caption:")
    st.write(description)
