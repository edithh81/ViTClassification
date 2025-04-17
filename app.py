import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from model.viT import ViT
import cv2

@st.cache_resource
def load_model(model_path, num_classes=10):
    viTmodel = ViT(embed_dim=512, num_heads=8, num_layers=4, ff_dim = 1024, num_classes=num_classes, patch_size=7, drop_out=0.2, img_size=28)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    viTmodel.load_state_dict(state_dict)
    viTmodel.eval()
    return viTmodel
# Preprocess the canvas image to grayscale and threshold
def preprocess_canvas_image(img_np):
    img_gray = Image.fromarray(img_np).convert("L")  # Convert to grayscale
    img_gray = img_gray.resize((28, 28))  # Resize to 28x28
    img_np = np.array(img_gray)
    _, img_thresh = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Threshold the image
    return Image.fromarray(img_thresh)

path = 'model/model_withnoise.pt'
model = load_model(path, num_classes=10)
def inference(image, model):
    # w, h = image.size
    # if w != h:
    #     crop = transforms.CenterCrop(min(w, h))
    #     image = crop(image)
    #     wnew, hnew = image.size
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])
    
    img_tensor = img_transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        predictions = model(img_tensor)
    preds = F.softmax(predictions, dim=1)
    p_max, yhat = torch.max(preds.data, 1)
    # Convert the image tensor back to numpy array for displaying after transformation
    img_after = img_tensor.squeeze(0).numpy()  # Remove the batch dimension
    img_after = np.squeeze(img_after)  # Remove the single channel dimension
    return p_max.item()*100, yhat.item(), img_after
        
def main():
    st.title("🖌️ Digit Recognizer using ViT")
    st.subheader("Draw a digit or upload an image to predict")
    st.markdown("This app uses a Vision Transformer (ViT) model to recognize handwritten digits. You can either draw a digit using the canvas or upload an image of a digit.")
    option = st.sidebar.selectbox("Choose an option", ("Draw a digit", "Upload an image"))

    if option == "Draw a digit":
        st.markdown("Draw a digit (0-9) below:")

        canvas_result = st_canvas(
            fill_color="#000000",           # Fill (background) = white like MNIST
            stroke_width=5,                 # Thinner strokes to match MNIST line width
            stroke_color="#FFFFFF",         # Draw in black (like MNIST digits)
            background_color="#000000",     # White background
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )

        if st.button("Predict") and canvas_result.image_data is not None:
            img = canvas_result.image_data
            inp_img = preprocess_canvas_image(img)  # Preprocess the canvas image
            p, label, img_after = inference(inp_img, model)
            img_after = img_after * 255
            img_after = img_after.astype(np.uint8)
            img_after = Image.fromarray(img_after, mode='L')
            st.image(img_after, caption="Drawn Digit", use_column_width=True)
            st.success(f"Prediction: **{label}** with confidence {p:.2f}%")
    elif option == "Upload an image":
        uploaded = st.file_uploader("Upload a 28x28 digit image", type=["png", "jpg", "jpeg"])
        if uploaded:
            img = Image.open(uploaded)
            p, label  = inference(img, model)
            st.image(img, caption="Uploaded Image", use_column_width=True)
            st.success(f"Prediction: **{label}** with confidence {p:.2f}%")

if __name__ == "__main__":
    main()