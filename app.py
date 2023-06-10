import streamlit as st
import os
import torch
from diffusers import StableDiffusionPipeline

st.title("Text2Poster AI")

# generate imae using stable difusion 
def gen_img_SD(input_text):
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype = torch.float16)
    pipe = pipe(input_text).image[0]
    image = pipe(input_text).images[0]
    return image