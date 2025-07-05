# Setup de modelo y utilidades
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import torch
import os
import pickle
from torchvision import transforms, models
from sklearn.neighbors import KernelDensity
from scipy.stats import yeojohnson


# 1) Carga de prototipos KDE
current_dir = os.path.dirname(__file__)
pickle_path = os.path.join(current_dir, "kde_univar_protos_vit.pkl")
with open(pickle_path, "rb") as f:
    PROTO_VIT = pickle.load(f)

# 2) Preparamos ambos encoders y transforms

# -- ViT --
net = models.vit_b_16(pretrained=True)
net.heads = torch.nn.Identity()
encoder_vit = net.to(device).eval()

# Funcion de transformación (igual para ambos)
tf_resnet = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std =[0.229,0.224,0.225])
])

def get_emb_vit(crop: Image.Image) -> np.ndarray:
    """Embedding 768-d con ViT."""
    x = tf_resnet(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        f = encoder_vit(x)      # (1,768,1,1)
    return f.squeeze().cpu().numpy()  # (768,)