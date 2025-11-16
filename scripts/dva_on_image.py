import streamlit as st
from PIL import Image, ImageDraw
import torch
import timm
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import cv2
import os # <-- UNCHANGED
from dotenv import load_dotenv # <-- UNCHANGED
from groq import Groq # <-- CHANGED: Replaced google.generativeai
import base64 # <-- ADDED: For image encoding
from io import BytesIO # <-- ADDED: For image encoding

# --- LOAD .ENV FILE ---
load_dotenv() # <-- UNCHANGED

# Set Streamlit page config
st.set_page_config(layout="wide", page_title="Image Difference Finder")

# --- Model Loading (Cached) ---

@st.cache_resource
def load_dino_model(model_name="vit_small_patch16_224_dino"):
    """
    Loads the pre-trained DINO model and puts it in evaluation mode.
    Caches the model resource for fast subsequent runs.
    """
    st.info(f"Loading DINO model ({model_name})... This may take a moment on first run.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    model.to(device)
    st.success("DINO model loaded successfully.")
    return model, device

@st.cache_resource
def get_dino_transform():
    """
    Gets the standard image transformation pipeline for DINO models.
    This squashes/stretches to 224x224, NO CROPPING.
    """
    return T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC), # Force resize
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

# --- Feature Extraction ---

def get_patch_features(model, processed_image, device):
    """
    Runs the image through the DINO model and returns the patch tokens.
    """
    with torch.no_grad():
        features = model.forward_features(processed_image.to(device))
    patch_features = features[:, 1:, :].squeeze(0)
    patch_features = F.normalize(patch_features, p=2, dim=1)
    return patch_features

# --- Difference Calculation Functions ---

def get_dino_difference(image1, image2, model, transform, device, threshold=0.7):
    """
    Finds differences using DINO features on 224x224 resized images.
    Returns a 14x14 binary mask.
    """
    proc_img1 = transform(image1).unsqueeze(0)
    proc_img2 = transform(image2).unsqueeze(0)

    features1 = get_patch_features(model, proc_img1, device)
    features2 = get_patch_features(model, proc_img2, device)

    similarity_matrix = torch.matmul(features2, features1.T)
    best_sims, _ = torch.max(similarity_matrix, dim=1)
    diff_scores = 1.0 - best_sims

    patch_grid_size = int(features2.shape[0]**0.5) # 14
    diff_map_14x14 = diff_scores.reshape(patch_grid_size, patch_grid_size).cpu().numpy()

    if diff_map_14x14.max() > diff_map_14x14.min():
        norm_map_14x14 = (diff_map_14x14 - diff_map_14x14.min()) / \
                         (diff_map_14x14.max() - diff_map_14x14.min())
    else:
        norm_map_14x14 = np.zeros_like(diff_map_14x14)

    mask_14x14 = (norm_map_14x14 > threshold).astype(np.uint8)
    
    return mask_14x14

def get_pixel_difference(image1, image2, threshold=0.1):
    """
    Finds differences using pixel subtraction on 224x224 resized images.
    Returns a 14x14 patch mask.
    """
    img1_np = np.array(image1.resize((224, 224)))
    img2_np = np.array(image2.resize((224, 224)))
    
    img1_gray = cv2.cvtColor(img1_np, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2_np, cv2.COLOR_RGB2GRAY)
    
    diff_224 = cv2.absdiff(img1_gray, img2_gray)
    
    pixel_threshold = int(threshold * 255)
    _, mask_224 = cv2.threshold(diff_224, pixel_threshold, 1, cv2.THRESH_BINARY)
    
    patch_size = 16
    mask_14x14 = np.zeros((14, 14), dtype=np.uint8)

    for i in range(14):
        for j in range(14):
            y1, y2 = i * patch_size, (i + 1) * patch_size
            x1, x2 = j * patch_size, (j + 1) * patch_size
            mask_patch = mask_224[y1:y2, x1:x2]
            if np.any(mask_patch):
                mask_14x14[i, j] = 1
            
    return mask_14x14

def draw_patch_boxes(image_pil, mask_14x14, color="red", thickness=2):
    """
    Draws a 14x14 grid scaled to the original image's size.
    """
    image_with_boxes = image_pil.copy()
    draw = ImageDraw.Draw(image_with_boxes)
    orig_width, orig_height = image_pil.size
    patch_width = orig_width / 14.0
    patch_height = orig_height / 14.0

    for i in range(14):
        for j in range(14):
            if mask_14x14[i, j] == 1:
                x1 = j * patch_width
                y1 = i * patch_height
                x2 = (j + 1) * patch_width
                y2 = (i + 1) * patch_height
                draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
                
    return image_with_boxes

# --- *** NEW: Groq VLM Analysis Function (Replaces Gemini) *** ---

def _pil_to_base64(pil_image):
    """Converts a PIL Image to a Base64 encoded string."""
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG") # Use JPEG for efficiency
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

@st.cache_data
def get_groq_analysis(_image1, _result_image, api_key):
    """
    Sends the baseline image and the result image (with boxes) to Groq
    and asks it to describe the changes.
    """
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        return f"Error configuring Groq API: {e}. Is your API key correct in .env?"
        
    model_id = "meta-llama/llama-4-maverick-17b-128e-instruct"
    
    prompt = """
    You will be given two images.
    - **Image 1** is the original 'baseline' image.
    - **Image 2** is the 'result' image, which is based on a second, un-seen image. It shows the differences from Image 1 highlighted with red boxes.
    
    Your task is to analyze the areas inside the red boxes on Image 2 and describe what is different compared to Image 1.
    
    For example:
    - "A person has been added to the scene."
    - "The text on the sign has been removed."
    - "The car in the driveway has changed color."
    
    Be concise and focus only on the changes marked by the red boxes.
    """
    
    try:
        # Convert PIL images to Base64
        base64_image1 = _pil_to_base64(_image1)
        base64_image2 = _pil_to_base64(_result_image)

        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image1}"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image2}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling Groq API: {e}")
        return None

# --- Streamlit UI ---

st.title("🖼️ DINO vs. Pixel Image Difference Finder")
st.write("""
Upload two images. The app will analyze them and draw **red squares** on the
**right-hand image** to show the patches that are different from the left-hand image.
""")

# --- Sidebar Controls ---
st.sidebar.title("Controls")
diff_mode = st.sidebar.radio(
    "Difference Mode",
    ["DINO (Semantic)", "Pixel (Simple)"],
    help="""
**DINO (Semantic):** Uses an AI model (ViT) to find *semantically* different patches. Good for finding added/removed *objects*, ignoring light changes. (Slower)
**Pixel (Simple):** Simple pixel-by-pixel subtraction. Fast, but sensitive to *any* change, including lighting and alignment.
"""
)

if diff_mode == "DINO (Semantic)":
    threshold = st.sidebar.slider(
        "Difference Threshold", 0.0, 1.0, 0.7, 0.05,
        help="Controls sensitivity. A *higher* value means it will only find *very* different patches."
    )
else:
    threshold = st.sidebar.slider(
        "Difference Threshold", 0.0, 1.0, 0.1, 0.05,
        help="Controls sensitivity. A *lower* value means it will detect even tiny pixel changes."
    )

# --- Main App Body ---
col1, col2 = st.columns(2)

with col1:
    img_file_1 = st.file_uploader("Upload Image 1 (Baseline)", type=["jpg", "png", "jpeg"])

with col2:
    img_file_2 = st.file_uploader("Upload Image 2 (Compare)", type=["jpg", "png", "jpeg"])

if img_file_1 and img_file_2:
    image1 = Image.open(img_file_1).convert("RGB")
    image2 = Image.open(img_file_2).convert("RGB")
    
    st.divider()
    
    st.subheader("Difference Analysis")
    st.info("ℹ️ **Note:** To analyze, the images are **resized (squashed/stretched)** to 224x224. The 14x14 patch grid from this analysis is then scaled up and drawn on your original Image 2.")

    mask_14x14 = None
    result_image = None

    if diff_mode == "DINO (Semantic)":
        with st.spinner("Analyzing differences with DINO..."):
            model, device = load_dino_model()
            dino_transform = get_dino_transform()
            mask_14x14 = get_dino_difference(image1, image2, model, dino_transform, device, threshold)
    else: # Pixel (Simple)
        with st.spinner("Analyzing pixel differences..."):
            mask_14x14 = get_pixel_difference(image1, image2, threshold)

    if mask_14x14 is not None:
        
        result_image = draw_patch_boxes(image2, mask_14x14)

        res1, res2 = st.columns(2)
        with res1:
            st.image(image1, caption="Image 1 (Baseline)", use_column_width=True)
        with res2:
            st.image(result_image, caption=f"Result: Different Patches on Image 2 ({diff_mode})", use_column_width=True)
            
        st.success("Analysis complete.")
        
        # --- *** MODIFIED: Groq Analysis Section *** ---
        st.divider()
        st.subheader("🤖 Groq Llama-4 Vision Analysis") # <-- CHANGED
        
        # Check if the API key was loaded from .env
        api_key = os.environ.get("GROQ_API_KEY") # <-- CHANGED
        
        if api_key:
            if st.button("Ask Groq (Llama-4) to describe the changes"): # <-- CHANGED
                with st.spinner("Groq (Llama-4) is analyzing the changes..."): # <-- CHANGED
                    groq_response = get_groq_analysis(image1, result_image, api_key) # <-- CHANGED
                    if groq_response:
                        st.markdown(groq_response)
        else:
            # Updated warning message
            st.warning("""
            **Could not find Groq API Key.**
            
            To enable this feature:
            1. Create a file named `.env` in the same directory as this script.
            2. Add this line to the `.env` file: `GROQ_API_KEY="YOUR_API_KEY_HERE"`
            3. Rerun the app.
            """) # <-- CHANGED