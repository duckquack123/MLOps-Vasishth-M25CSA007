import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os

# =========================
# MODEL DEFINITION
# =========================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SimpleUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.down1 = DoubleConv(n_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(256, 512)

        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.pool1(x1)

        x3 = self.down2(x2)
        x4 = self.pool2(x3)

        x5 = self.down3(x4)
        x6 = self.pool3(x5)

        x7 = self.down4(x6)

        x = self.up1(x7)
        x = torch.cat([x, x5], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)

        return self.outc(x)


# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():
    model = SimpleUNet(n_channels=3, n_classes=23)
    model.load_state_dict(torch.load('Question2/unet_model.pth', map_location='cpu'))
    model.eval()
    return model


# =========================
# COLOR MAP (IMPORTANT FIX)
# =========================

def create_color_map(num_classes=23):
    np.random.seed(42)
    colors = np.random.randint(0, 255, (num_classes, 3))
    return colors

COLOR_MAP = create_color_map()


def decode_mask(mask):
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(len(COLOR_MAP)):
        colored[mask == i] = COLOR_MAP[i]

    return colored


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="UNet Segmentation App", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Training Metrics", "Inference"])


# =========================
# TRAINING PAGE
# =========================

if page == "Training Metrics":
    st.title("Training Metrics")

    try:
        col1, col2 = st.columns(2)

        with col1:
            st.image("Question2/train_loss.png", caption="Loss Curve")

        with col2:
            st.image("Question2/metrics.png", caption="mIoU / mDice")

        with open('Question2/test_metrics.txt', 'r') as f:
            miou, mdice = map(float, f.read().split(','))

        st.info(f"mIoU: {miou:.4f}\n\nmDice: {mdice:.4f}")

    except:
        st.error("Training files not found.")


# =========================
# INFERENCE PAGE
# =========================

elif page == "Inference":

    st.title("Segmentation Inference")

    uploaded_files = st.file_uploader(
        "Upload exactly 4 images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:

        if len(uploaded_files) != 4:
            st.warning("Upload exactly 4 images")

        else:
            model = load_model()
            st.success("Model loaded")

            for file in uploaded_files:

                st.subheader(file.name)

                col1, col2, col3 = st.columns(3)

                # ---- INPUT IMAGE ----
                image = Image.open(file).convert("RGB")
                image_resized = image.resize((256, 256))

                img_array = np.array(image_resized).astype(np.float32) / 255.0

                # 🔥 IMPORTANT FIX: normalize (if used during training)
                img_array = (img_array - 0.5) / 0.5

                img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0)

                # ---- MODEL PREDICTION ----
                with torch.no_grad():
                    output = model(img_tensor)
                    pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

                # ---- GT MASK ----
                mask_path = os.path.join('data/CameraMask', file.name)

                if os.path.exists(mask_path):
                    gt_mask = Image.open(mask_path).convert("L").resize((256, 256), Image.NEAREST)
                    gt_mask = np.array(gt_mask)
                else:
                    gt_mask = np.zeros((256, 256), dtype=np.uint8)

                # ---- COLORIZE ----
                pred_colored = decode_mask(pred_mask)
                gt_colored = decode_mask(gt_mask)

                # ---- DISPLAY ----
                with col1:
                    st.image(image_resized, caption="Input")

                with col2:
                    st.image(gt_colored, caption="Ground Truth")

                with col3:
                    st.image(pred_colored, caption="Prediction")

                st.divider()