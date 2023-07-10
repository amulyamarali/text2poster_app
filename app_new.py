import streamlit as st
import os
import torch
from diffusers import StableDiffusionPipeline

st.title("Text2Poster AI")
input_text = st.text_input("Enter the product name")
def gen_img_SD():
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype = torch.float16)
    pipe = pipe.to("cuda")
    # pipe = pipe(input_text).image[0]
    image = pipe(input_text).images[0]
    st.image(image)
    # return image
button = st.button("Generate",on_click=gen_img_SD)
# st.image(gen_img_SD("Sports Shoe"))