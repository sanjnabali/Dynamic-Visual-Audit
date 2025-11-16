#
# ----------------------------------------------------------------------
# DVA (Dynamic Visual Audit) - FULL PIPELINE (STAGE 1 + 2 + 3)
#
# PROFESSIONAL UPGRADE V18: Altair Heatmap Scheme Fix
# ----------------------------------------------------------------------
#
# This script fixes a `ValueError: 'viridis' is an invalid value
# for range` in the Altair chart.
#
# 1. Altair Heatmap Fix (Section 5, Tab 3):
#    - The `range` parameter in `alt.Scale` does not accept
#      colormap names.
#    - Changed `scale=alt.Scale(range='viridis')` to
#      `scale=alt.Scale(scheme='viridis')` to correctly
#      apply the Viridis color scheme.
#
# ----------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import tempfile
import io
import json
import time
from collections import deque
import textwrap
import threading
import queue
import traceback # For detailed error logging
import base64 # <-- NEW: For Groq image encoding
from io import BytesIO # <-- NEW: For Groq image encoding
import pandas as pd # <-- NEW V16: For analytics dashboard
import altair as alt # <-- NEW V17: For heatmap

# --- VLM / AI Imports ---
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
import onnxruntime as ort
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
from enum import Enum

# --- NEW: Groq VLM Imports ---
from groq import Groq
from dotenv import load_dotenv
import warnings

# --- Load .env file (for GROQ_API_KEY) ---
load_dotenv()

# =======================================================================
# SECTION 1A: DINOv2 (STAGE 2) LOGIC
# (*** MODIFIED V17: JSON Serialization Fix ***)
# =======================================================================

# --- DINO Profiles ---
class DetectionProfile(Enum):
    F1_CRITICAL = "f1_critical"
    RETAIL = "retail"
    RETAIL_STABLE = "retail_stable"
    CUSTOM = "custom"

@dataclass
class ThresholdConfig:
    name: str
    description: str
    semantic_threshold: float
    change_ratio: float
    top_k: int
    temporal_stability_frames: int = 1

    def min_changed_patches(self, total_patches: int = 256) -> int:
        return max(1, int(total_patches * self.change_ratio))

PROFILES = {
    DetectionProfile.F1_CRITICAL: ThresholdConfig(
        name="F1/Critical Sensitivity",
        description="Catch tiny changes (cracks, defects). (VLM: High Traffic)",
        semantic_threshold=0.985,
        change_ratio=0.005, # ~2 patches
        top_k=5,
        temporal_stability_frames=1
    ),
    DetectionProfile.RETAIL: ThresholdConfig(
        name="Retail/Logistics (Noisy)",
        description="Maximum noise rejection. Ignore shadows, people. (VLM: High Traffic)",
        semantic_threshold=0.985,
        change_ratio=0.05, # ~13 patches
        top_k=10,
        temporal_stability_frames=1
    ),
    DetectionProfile.RETAIL_STABLE: ThresholdConfig(
        name="Retail/Logistics (Stable)",
        description="Noise rejection + Temporal Gate. (VLM: Low Traffic)",
        semantic_threshold=0.985,
        change_ratio=0.05, # ~13 patches
        top_k=10,
        temporal_stability_frames=3
    ),
    DetectionProfile.CUSTOM: ThresholdConfig(
        name="Custom",
        description="User-defined thresholds",
        semantic_threshold=0.92,
        change_ratio=0.02, # ~5 patches
        top_k=10,
        temporal_stability_frames=1
    )
}

# --- DINO Config ---
@dataclass
class DinoConfig:
    MODEL_ID: str = "facebook/dinov2-with-registers-small"
    ONNX_PATH: str = "dino_semantic_gate.onnx"
    PATCH_SIZE: int = 14
    INPUT_SIZE: int = 224
    PATCHES_PER_DIM: int = 16
    NUM_PATCHES: int = 256
    BATCH_SIZE: int = 2
    USE_GPU: bool = True

dino_config = DinoConfig()

# --- ONNX Export (Run once) ---
@st.cache_resource
def export_dino_to_onnx(force_export: bool = False) -> bool:
    if os.path.exists(dino_config.ONNX_PATH) and not force_export:
        return True
    st.toast(f"⚙️ Exporting {dino_config.MODEL_ID} to ONNX...")
    try:
        device = torch.device("cpu")
        model = AutoModel.from_pretrained(dino_config.MODEL_ID).to(device).eval()
        dummy_input = torch.randn(dino_config.BATCH_SIZE, 3, dino_config.INPUT_SIZE, dino_config.INPUT_SIZE, device=device)
        dynamic_axes = {'pixel_values': {0: 'batch_size'}, 'last_hidden_state': {0: 'batch_size'}}
        
        with torch.no_grad():
            torch.onnx.export(
                model, dummy_input, dino_config.ONNX_PATH,
                export_params=True, opset_version=17, do_constant_folding=True,
                input_names=['pixel_values'], output_names=['last_hidden_state'],
                dynamic_axes=dynamic_axes
            )
        st.toast(f"✓ ONNX export complete.")
        return True
    except Exception as e:
        st.error(f"✗ ONNX export failed: {e}")
        return False

# --- DINO Analyzer Class ---
class SemanticAnalyzer:
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, config: DinoConfig):
        self.config = config
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if config.USE_GPU else ['CPUExecutionProvider']
        
        try:
            self.session = ort.InferenceSession(config.ONNX_PATH, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}. Is ONNX Runtime (GPU) installed?")
        
        st.success(f"✅ DINOv2 Analyzer ready (Provider: {self.session.get_providers()[0]})")

    def _preprocess_frame_fast(self, frame_cv2: np.ndarray) -> Optional[np.ndarray]:
        if frame_cv2 is None:
            print("Warning: _preprocess_frame_fast received a None frame.")
            return None
        try:
            rgb_cv2 = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB)
            resized_rgb = cv2.resize(
                rgb_cv2, 
                (self.config.INPUT_SIZE, self.config.INPUT_SIZE), 
                interpolation=cv2.INTER_CUBIC
            )
            normalized = (resized_rgb.astype(np.float32) / 255.0 - self.MEAN) / self.STD
            transposed = normalized.transpose(2, 0, 1)
            return np.expand_dims(transposed, axis=0)
        except Exception as e:
            print(f"Error in DINO preprocessing: {e}")
            return None

    def _extract_patch_features(self, img_before_cv2: np.ndarray, img_after_cv2: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        processed_before = self._preprocess_frame_fast(img_before_cv2)
        processed_after = self._preprocess_frame_fast(img_after_cv2)
        
        if processed_before is None or processed_after is None:
            raise ValueError("DINO received a None frame, possibly from preprocessing failure.")
        
        pixel_values = np.concatenate([processed_before, processed_after], axis=0)
        
        ort_inputs = {self.input_name: pixel_values}
        ort_outputs = self.session.run([self.output_name], ort_inputs)[0]
        features = torch.from_numpy(ort_outputs)
        
        features_before = features[0, 1:self.config.NUM_PATCHES + 1, :]
        features_after = features[1, 1:self.config.NUM_PATCHES + 1, :]
        
        return features_before, features_after
    
    def compare_frames(self, img_before_cv2: np.ndarray, img_after_cv2: np.ndarray, threshold_config: ThresholdConfig) -> Dict:
        try:
            features_before, features_after = self._extract_patch_features(img_before_cv2, img_after_cv2)
            
            similarity_scores = F.cosine_similarity(features_before, features_after, dim=-1)
            similarity_map = similarity_scores.reshape(self.config.PATCHES_PER_DIM, self.config.PATCHES_PER_DIM).cpu().numpy()
            
            raw_change_mask_2d = similarity_map < threshold_config.semantic_threshold
            stable_change_mask_2d = raw_change_mask_2d
            
            num_changed_patches = int(np.sum(stable_change_mask_2d))
            min_required = threshold_config.min_changed_patches(self.config.NUM_PATCHES)
            
            is_meaningful = num_changed_patches >= min_required
            
            changed_patch_coords = []
            if is_meaningful:
                all_changed_coords = np.argwhere(stable_change_mask_2d)
                scores_of_changed_patches = similarity_map[stable_change_mask_2d]
                sorted_indices = np.argsort(scores_of_changed_patches)
                top_k = threshold_config.top_k
                top_k_indices = sorted_indices[:top_k]
                final_coords_to_draw = all_changed_coords[top_k_indices]

                for row, col in final_coords_to_draw:
                    x1 = col * self.config.PATCH_SIZE
                    y1 = row * self.config.PATCH_SIZE
                    x2 = x1 + self.config.PATCH_SIZE
                    y2 = y1 + self.config.PATCH_SIZE
                    # --- *** MODIFIED V17: Cast all numpy types to standard int *** ---
                    changed_patch_coords.append({
                        'x1': int(x1), 
                        'y1': int(y1), 
                        'x2': int(x2), 
                        'y2': int(y2)
                    })
            
            avg_similarity = float(np.mean(similarity_map))
            
            return {
                'is_meaningful_change': is_meaningful,
                'changed_patch_coords': changed_patch_coords,
                'num_changed_patches': int(num_changed_patches), # <-- MODIFIED V17: Cast to int
                'avg_similarity': avg_similarity,
                'threshold_config': {'required_patches': int(min_required)} # <-- MODIFIED V17: Cast to int
            }
            
        except Exception as e:
            print(f"✗ Error in DINO comparison: {e}")
            return {'is_meaningful_change': False, 'changed_patch_coords': [], 'num_changed_patches': 0, 'error': str(e)}

# =======================================================================
# SECTION 1B: GROQ VLM (STAGE 3) LOGIC
# (No changes in this section)
# =======================================================================

class VLMAuditor:
    # --- Prompts are unchanged ---
    DEFAULT_AUDIT_PROMPT = (
        "You are an AI assistant for F1 race control, identifying crash damage. "
        "You will be given a sequence of images showing an event unfolding. "
        "The first image is 'Before' the event, and the following images are the 'After' state. "
        "**Analyze the sequence and describe the damage or incident in 20 words or less.** "
        "Be extremely concise. If no crash/damage, say 'No significant damage detected.'\n\n"
        "EXAMPLES:\n"
        "- 'Front wing is broken and dragging.'\n"
        "- 'Left rear tire puncture, car spinning.'\n"
        "- 'Car impacted the barrier at Turn 5.'\n"
        "- 'No significant damage detected.'"
    )
    DEFAULT_SCOUT_PROMPT = (
        "You are an F1 'Scout' system. You will be given a sequence of images. "
        "The first image is the 'Baseline-Wing', and the following images are the 'Competitor-Wing'. "
        "**Describe the key design difference and its purpose in 20 words or less.** "
        "Example: 'New endplate design to improve outwash.'"
    )
    DEFAULT_AUDIT_FOCUSED_PROMPT = (
        "You are an AI assistant for F1 race control, analyzing a *zoomed-in ROI*. "
        "You will be given a *sequence* of cropped images showing an event unfolding. "
        "The first image is 'Before' the event, and the following images are the 'After' state. "
        "**Analyze the sequence and describe the damage or incident in 20 words or less.** "
        "Be extremely concise. If no crash/damage, say 'No significant damage detected.'\n\n"
        "EXAMPLES:\n"
        "- 'Front wing is broken and dragging.'\n"
        "- 'Left rear tire puncture, car spinning.'\n"
        "- 'Car impacted the barrier at Turn 5.'\n"
        "- 'No significant damage detected.'"
    )
    DEFAULT_SCOUT_FOCUSED_PROMPT = (
        "You are an F1 'Scout' system analyzing a *zoomed-in* ROI. "
        "You will be given a sequence of images. "
        "The first image is the 'Baseline-Wing', and the following images are the 'Competitor-Wing'. "
        "**Describe the key design difference and its purpose in 20 words or less.** "
        "Example: 'Competitor wing has a more aggressive gurney flap for downforce.'"
    )

    def __init__(self, model_id: str):
        print(f"VLMAuditor: Initializing... Using Groq API and model '{model_id}'.")
        self.client = None
        self.model_id = model_id
        self.api_key = os.environ.get("GROQ_API_KEY")
        
        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
                print("VLMAuditor: Groq client initialized successfully.")
            except Exception as e:
                print(f"VLMAuditor: CRITICAL ERROR initializing Groq client. {e}")
        else:
            print("VLMAuditor: Skipping init, GROQ_API_KEY not found.")

    def _pil_to_base64(self, img_pil: Image.Image) -> str:
        """Converts a PIL Image to a Base64 encoded string."""
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def describe_sequence(
        self, 
        frame_sequence: List[Image.Image], 
        prompt_template: str = None
    ):
        """
        Performs a VLM audit on a *sequence* of images using Groq.
        """
        if not self.client:
            return "Error: VLMAuditor is not initialized. Check GROQ_API_KEY."
        if not frame_sequence:
            return "Error: VLM received an empty frame sequence."

        try:
            if prompt_template is None:
                prompt_template = self.DEFAULT_AUDIT_PROMPT
            
            # Build the messages payload for Groq VLM
            messages = [
                {
                    "role": "user",
                    "content": []
                }
            ]
            
            # 1. Add the text prompt
            messages[0]["content"].append({
                "type": "text",
                "text": prompt_template
            })
            
            # 2. Add the sequence of images
            for i, img in enumerate(frame_sequence):
                base64_image = self._pil_to_base64(img)
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })

            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=100 # 20 words is ~30-40 tokens. 100 is safe.
            )
            
            description = response.choices[0].message.content.strip()
            return description

        except Exception as e:
            print(f"VLMAuditor: Error during Groq API call (sequence). {e}")
            return f"Error: Model generation failed via Groq API (sequence). {e}"

    def describe_focused_sequence(
        self, 
        frame_sequence: List[Image.Image], 
        dino_report: Dict, 
        dino_input_size: int, 
        prompt_template: str
    ):
        """
        Performs a VLM audit on a *cropped ROI* applied to *all frames in a sequence* using Groq.
        """
        if not self.client:
            return "Error: VLMAuditor is not initialized. Check GROQ_API_KEY."
        if not frame_sequence:
            return "Error: VLM received an empty focused sequence."

        try:
            coords_list = dino_report.get('changed_patch_coords')
            
            if not coords_list:
                warnings.warn("describe_focused_sequence called but no coords found. Falling back to default.")
                return self.describe_sequence(frame_sequence, prompt_template)

            # --- 2. Calculate Bounding Box from DINO patches ---
            min_x = min(p['x1'] for p in coords_list)
            min_y = min(p['y1'] for p in coords_list)
            max_x = max(p['x2'] for p in coords_list)
            max_y = max(p['y2'] for p in coords_list)

            # --- 3. Scale Bounding Box to Original Image Size ---
            orig_w, orig_h = frame_sequence[0].size
            scale_x = orig_w / dino_input_size
            scale_y = orig_h / dino_input_size
            crop_x1 = int(min_x * scale_x)
            crop_y1 = int(min_y * scale_y)
            crop_x2 = int(max_x * scale_x)
            crop_y2 = int(max_y * scale_y)
            
            # --- 4. Add Padding (Robustness) ---
            pad_w = int((crop_x2 - crop_x1) * 0.15)
            pad_h = int((crop_y2 - crop_y1) * 0.15)
            crop_x1 = max(0, crop_x1 - pad_w)
            crop_y1 = max(0, crop_y1 - pad_h)
            crop_x2 = min(orig_w, crop_x2 + pad_w)
            crop_y2 = min(orig_h, crop_y2 + pad_h)

            # --- 5. Build Groq VLM payload with Focused Prompt & Cropped Images ---
            messages = [
                {
                    "role": "user",
                    "content": []
                }
            ]
            
            # 1. Add the text prompt
            messages[0]["content"].append({
                "type": "text",
                "text": prompt_template # This should be the FOCUSED prompt
            })

            for i, img in enumerate(frame_sequence):
                # --- 6. Crop *each* image in the sequence ---
                cropped_pil = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                
                # Convert cropped PIL to base64
                base64_image = self._pil_to_base64(cropped_pil)
                
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })

            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=100
            )
            
            description = response.choices[0].message.content.strip()
            
            return f"[Focused Analysis]: {description}"

        except Exception as e:
            print(f"VLMAuditor: Error during FOCUSED Groq API call (sequence). {e}")
            return f"Error: Model generation failed via Groq API (Focused sequence). {e}"
    
    # --- DEPRECATED Functions ---
    def describe_change(self, *args, **kwargs):
        return "Error: describe_change is deprecated. Use describe_sequence."
    def describe_focused_change(self, *args, **kwargs):
        return "Error: describe_focused_change is deprecated. Use describe_focused_sequence."

# =======================================================================
# SECTION 2: FAST CV2 SIEVE (STAGE 1)
# (No changes in this section)
# =======================================================================

def trigger_dva_sieve_fast(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    pixel_threshold: int = 30,
    min_changed_pixels: int = 1000,
    blur_kernel_size: tuple = (7, 7)
) -> dict:
    report = {
        "trigger_audit": False,
        "change_score": 0,
        "reason": "No significant change detected."
    }
    try:
        if frame_a is None or frame_b is None:
            report["reason"] = "Sieve error: Received a None frame."
            return report
        if frame_a.shape != frame_b.shape:
            frame_b = cv2.resize(frame_b, (frame_a.shape[1], frame_a.shape[0]))
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
        blur_a = cv2.GaussianBlur(gray_a, blur_kernel_size, 0)
        blur_b = cv2.GaussianBlur(gray_b, blur_kernel_size, 0)
        diff = cv2.absdiff(blur_a, blur_b)
        _, thresh = cv2.threshold(diff, pixel_threshold, 255, cv2.THRESH_BINARY)
        changed_pixels = cv2.countNonZero(thresh)
        report["change_score"] = int(changed_pixels)
        if changed_pixels > min_changed_pixels:
            report["trigger_audit"] = True
            report["reason"] = f"Triggered: {changed_pixels} pixels changed."
        else:
            report["reason"] = f"Change below threshold ({changed_pixels} pixels)."
        return report
    except Exception as e:
        report["reason"] = f"Sieve error: {e}"
        return report

# =======================================================================
# SECTION 3: HELPER FUNCTIONS & VISUALIZATION
# (*** MODIFIED V15: Dynamic Multi-line Header ***)
# =======================================================================

def cv2_to_pil(image_cv2: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))

def pil_to_cv2(image_pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def create_dino_visualization(
    original_img_pil: Image.Image,
    changed_patch_coords: List[Dict],
    dino_report: Dict,
    vlm_description: str
) -> Image.Image:
    
    img_width, img_height = original_img_pil.size
    scale_x = img_width / dino_config.INPUT_SIZE
    scale_y = img_height / dino_config.INPUT_SIZE
    
    annotated = original_img_pil.copy()
    draw = ImageDraw.Draw(annotated)
    
    # --- 1. Draw DINO patches ---
    for patch in changed_patch_coords:
        x1 = int(patch['x1'] * scale_x)
        y1 = int(patch['y1'] * scale_y)
        x2 = int(patch['x2'] * scale_x)
        y2 = int(patch['y2'] * scale_y)
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
    
    # --- 2. Load Fonts ---
    try:
        font_size = 32
        top_font = ImageFont.truetype("arial.ttf", font_size)
        bottom_font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        try:
            top_font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            bottom_font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except IOError:
            top_font = ImageFont.load_default()
            bottom_font = ImageFont.load_default()

    # --- 3. Top Bar (VLM Text) ---
    
    # --- 3a. Text Wrapping Logic ---
    vlm_text_prefix = "Groq (Llama-4):"
    full_vlm_text = f"{vlm_text_prefix} {vlm_description}"

    try:
        # Get average char width for proper wrapping
        avg_char_width = top_font.getlength("abcdefghijklmnopqrstuvwxyz") / 26
    except AttributeError:
        # Fallback for older Pillow versions
        avg_char_width = top_font.getsize("a")[0] 
        
    padding_horizontal = 20 # 10px on each side
    available_width_px = img_width - padding_horizontal
    wrap_width_chars = max(10, int(available_width_px / avg_char_width)) # Ensure at least 10 chars

    wrapped_lines = textwrap.wrap(full_vlm_text, width=wrap_width_chars)
    final_text = "\n".join(wrapped_lines)
    num_lines = len(wrapped_lines)

    # --- 3b. Calculate new bar height ---
    try:
        # Get line height from font bounding box
        left, top, right, bottom = top_font.getbbox("A")
        line_height_px = bottom - top
    except AttributeError:
        # Fallback for older Pillow versions
        line_height_px = top_font.getsize("A")[1]
    
    line_spacing = 5
    padding_vertical = 10 # 10px top and bottom
    top_bar_height = (num_lines * line_height_px) + ((num_lines - 1) * line_spacing) + (padding_vertical * 2)

    # --- 3c. Draw Header ---
    vlm_color = (255, 0, 0) # Red
    draw.rectangle([0, 0, img_width, top_bar_height], fill='black')
    draw.multiline_text(
        (10, padding_vertical), # (x, y) top-left corner
        final_text,
        fill=vlm_color,
        font=top_font,
        spacing=line_spacing
    )
    
    # --- 4. Bottom Bar (DINO Stats) ---
    bottom_bar_height = 50 # Keep this one static and simple
    num_patches_drawn = len(changed_patch_coords)
    num_patches_total = dino_report.get('num_changed_patches', 0)
    dino_text = f"DINO: {num_patches_drawn} Worst Patches (of {num_patches_total} total)"
    
    draw.rectangle([0, img_height - bottom_bar_height, img_width, img_height], fill='black')
    # Center the bottom text vertically
    text_y = img_height - bottom_bar_height + (bottom_bar_height - line_height_px) / 2
    draw.text((10, text_y), dino_text, fill=(255, 255, 0), font=bottom_font) # Yellow
    
    return annotated

# =======================================================================
# SECTION 4: INTEGRATED VIDEO PROCESSOR
# (*** MODIFIED V16: Add patch coords to log ***)
# =======================================================================

def process_video(
    source_path: str,
    # --- Stage 1 (Sieve) Config ---
    sieve_mode: str,
    sieve_pixel_thresh: int,
    sieve_min_pixels: int,
    sieve_sample_rate: int,
    sieve_soft_update: float,
    sieve_dynamic_gap: int, 
    # --- Stage 2 (DINO) Config ---
    dino_analyzer: SemanticAnalyzer,
    dino_profile: ThresholdConfig,
    # --- Stage 3 (VLM) Config ---
    vlm_auditor: VLMAuditor,
    vlm_prompt: str,
    vlm_sequence_size: int,
    enable_vlm: bool,
    enable_vlm_focus: bool,
    vlm_results_queue: queue.Queue,
    # --- NEW V14: Stage 4 (Annotation) Config ---
    ui_annotation_duration_sec: int
):
    
    # --- vlm_thread_worker (Unchanged) ---
    def vlm_thread_worker(
        frame_sequence: List[Image.Image], 
        report: Dict, 
        dino_size: int, 
        prompt: str, 
        focus: bool
    ):
        try:
            if focus and report:
                description = vlm_auditor.describe_focused_sequence(
                    frame_sequence, report, dino_size, prompt
                )
            else:
                description = vlm_auditor.describe_sequence(
                    frame_sequence, prompt
                )
            vlm_results_queue.put(description)
        
        except Exception as e:
            error_msg = f"Error: VLM thread failed. {e}"
            print(error_msg)
            vlm_results_queue.put(error_msg)
    
    # 1. Setup Video Capture
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        st.error("Error opening video file.")
        return None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # --- NEW V14: Convert annotation duration from seconds to frames ---
    # Handle videos with 0 FPS (e.g., from images)
    if fps == 0:
        st.warning("Video FPS is 0. Defaulting to 30 FPS for duration calculation.")
        fps = 30
    annotation_duration_frames = int(ui_annotation_duration_sec * fps)
    
    # 2. Setup Video Writer
    out_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        st.error(f"Error: Could not open video writer for path: {out_path}")
        cap.release()
        return None, None

    # 3. Initialize State
    baseline_frame = None
    frame_history = deque(maxlen=sieve_dynamic_gap) 
    frame_count = 0
    last_sieve_report = {"trigger_audit": False, "change_score": 0}
    all_scores = [] 
    
    N_stable_frames = max(1, dino_profile.temporal_stability_frames)
    dino_confirmation_buffer = deque(maxlen=N_stable_frames)
    vlm_cooldown_active = False
    last_stable_dino_report = None
    last_stable_vlm_description = ""
    
    # --- NEW V14: Replaced `vlm_response_is_damage` with timer ---
    annotation_display_frames_remaining = 0
    
    vlm_frame_buffer = deque(maxlen=vlm_sequence_size)
    last_dino_input_after_pil = None
    
    progress_bar = st.progress(0, text="Processing video... Please wait.")
    st_status_text = st.empty()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame_to_write = frame.copy() 
            
            run_dino = False
            sieve_ran_this_loop = False
            current_frame_dino_meaningful = False
            dino_input_before = None 
            dino_input_after = None 
            
            # --- 4. Handle Baseline (First Frame) ---
            if baseline_frame is None:
                baseline_frame = frame.copy()
                dino_confirmation_buffer.append(False)
                out.write(frame_to_write)
                all_scores.append({"frame": frame_count, "type": "baseline", "score": 0})
                frame_history.append(frame.copy()) 
                continue

            # --- 5. Core Sieve Logic (Stage 1) ---
            if sieve_mode == "Static Scene (Adaptive Baseline)":
                if frame_count % sieve_sample_rate == 0:
                    sieve_ran_this_loop = True
                    sieve_report = trigger_dva_sieve_fast(
                        baseline_frame, frame, sieve_pixel_thresh, sieve_min_pixels
                    )
                    last_sieve_report = sieve_report
                    
                    if sieve_report["trigger_audit"]:
                        run_dino = True
                        dino_input_before = baseline_frame.copy() 
                        dino_input_after = frame.copy() 
                        baseline_frame = frame.copy()
                    else:
                        baseline_frame = cv2.addWeighted( 
                            baseline_frame, 1 - sieve_soft_update, 
                            frame, sieve_soft_update, 0
                        )
                else:
                    sieve_ran_this_loop = False

            elif sieve_mode == "Dynamic Scene (Consecutive Frame)":
                frame_n = frame.copy() 
                frame_history.append(frame_n) 
                
                if len(frame_history) < sieve_dynamic_gap:
                    sieve_ran_this_loop = False
                    last_sieve_report["reason"] = f"Buffering (need {sieve_dynamic_gap} frames)"
                    last_sieve_report["change_score"] = 0
                else:
                    sieve_ran_this_loop = True
                    frame_n_minus_x = frame_history[0] 
                    sieve_report = trigger_dva_sieve_fast(
                        frame_n_minus_x, frame_n, sieve_pixel_thresh, sieve_min_pixels
                    )
                    last_sieve_report = sieve_report
                    if sieve_report["trigger_audit"]:
                        run_dino = True
                        dino_input_before = frame_n_minus_x.copy() 
                        dino_input_after = frame_n.copy()
            
            # --- 6. DINO Audit (Stage 2) ---
            if run_dino:
                st_status_text.info(f"Frame {frame_count}: Sieve triggered! Running DINO...")
                dino_report = dino_analyzer.compare_frames(
                    dino_input_before, dino_input_after, dino_profile
                )
                current_frame_dino_meaningful = dino_report['is_meaningful_change']
                
                if current_frame_dino_meaningful:
                    last_dino_input_after_pil = cv2_to_pil(dino_input_after) 
                    last_stable_dino_report = dino_report
                    if not vlm_cooldown_active:
                        is_new_event = not any(dino_confirmation_buffer)
                        if is_new_event:
                            vlm_frame_buffer.clear()
                            vlm_frame_buffer.append(cv2_to_pil(dino_input_before))
                        vlm_frame_buffer.append(last_dino_input_after_pil.copy())
                
                all_scores.append({
                    "frame": frame_count, 
                    "type": "dino_semantic", 
                    "score": dino_report.get("avg_similarity", 0),
                    "meaningful": dino_report['is_meaningful_change'],
                    # --- NEW V16: Add coords for heatmap ---
                    "changed_patch_coords": dino_report.get('changed_patch_coords', []),
                    "num_changed_patches": dino_report.get('num_changed_patches', 0)
                })
            
            elif sieve_ran_this_loop:
                all_scores.append({
                    "frame": frame_count,
                    "type": "sieve_pixels",
                    "score": last_sieve_report["change_score"],
                    "meaningful": False
                })

            # --- Check for VLM results (MODIFIED V14) ---
            try:
                new_vlm_description = vlm_results_queue.get_nowait()
                last_stable_vlm_description = new_vlm_description
                
                if "no significant damage" in new_vlm_description.lower():
                    # VLM said no damage, do nothing
                    st_status_text.success(f"Frame {frame_count}: Groq (Llama-4): No damage found.")
                else:
                    # VLM *confirmed* damage, start the annotation timer
                    st_status_text.success(f"Frame {frame_count}: Groq (Llama-4) analysis *received*! (Damage Confirmed)")
                    annotation_display_frames_remaining = annotation_duration_frames
                
                # Reset cooldown regardless of damage
                st_status_text.info(f"Frame {frame_count}: VLM response received. Resetting cooldown.")
                vlm_cooldown_active = False
                vlm_frame_buffer.clear()
                dino_confirmation_buffer.clear() 

                all_scores.append({
                    "frame": frame_count,
                    "type": "VLM_EVENT_RECEIVED",
                    "score": 0,
                    "meaningful": True,
                    "vlm_description": new_vlm_description
                })
            except queue.Empty:
                pass


            # --- 7. Temporal Gate & VLM (Stage 2.5 / 3) (MODIFIED V14) ---
            dino_confirmation_buffer.append(current_frame_dino_meaningful)
            
            is_buffer_full = (len(dino_confirmation_buffer) == N_stable_frames)
            is_stably_confirmed = all(dino_confirmation_buffer)
            
            should_trigger_vlm = is_buffer_full and is_stably_confirmed and not vlm_cooldown_active
            is_streak_broken = not is_stably_confirmed
            
            if is_streak_broken and not vlm_cooldown_active:
                # DINO stopped seeing the event, and we're not waiting for a VLM response
                # Clear the buffer for the *next* event.
                vlm_frame_buffer.clear()
            
            if should_trigger_vlm:
                vlm_cooldown_active = True
                
                if enable_vlm:
                    st_status_text.error(f"Frame {frame_count}: STABLE CHANGE. Sending {len(vlm_frame_buffer)}-frame sequence to Groq (Llama-4)...")
                    
                    frames_to_send = list(vlm_frame_buffer)
                    
                    threading.Thread(
                        target=vlm_thread_worker,
                        args=(
                            frames_to_send,
                            last_stable_dino_report.copy(),
                            dino_config.INPUT_SIZE,
                            vlm_prompt,
                            enable_vlm_focus
                        ),
                        daemon=True
                    ).start()
                    
                    last_stable_vlm_description = "[VLM Analysis in Progress...]"
                    
                    all_scores.append({
                        "frame": frame_count,
                        "type": "VLM_EVENT_SENT",
                        "score": 0,
                        "meaningful": True,
                        "vlm_description": f"VLM call initiated with {len(frames_to_send)} frames."
                    })
                else:
                    st_status_text.error(f"Frame {frame_count}: STABLE CHANGE DETECTED. VLM disabled.")
                    last_stable_vlm_description = "VLM analysis disabled."

            # --- 8. Unified Frame Visualization & Writing (MODIFIED V14) ---
            if annotation_display_frames_remaining > 0:
                # Timer is active, draw annotations
                
                # Use the *current* frame to draw on, so the boxes "stick"
                # even if the underlying `last_dino_input_after_pil` is old
                current_frame_pil = cv2_to_pil(frame)
                
                if last_stable_dino_report:
                    annotated_pil = create_dino_visualization(
                        current_frame_pil, # Draw on the *current* frame
                        last_stable_dino_report['changed_patch_coords'], 
                        last_stable_dino_report,
                        last_stable_vlm_description
                    )
                    frame_to_write = pil_to_cv2(annotated_pil)
                
                # Decrement timer
                annotation_display_frames_remaining -= 1
                
                if annotation_display_frames_remaining == 0:
                    # Timer just expired
                    st_status_text.info(f"Frame {frame_count}: Annotation display timer expired.")
            
            # --- 9. Write Frame & Update Progress ---
            out.write(frame_to_write)
            progress_bar.progress(frame_count / total_frames, text=f"Processing frame {frame_count}/{total_frames}")

    except Exception as e:
        st.error(f"FATAL ERROR during video processing at frame {frame_count}: {e}")
        print(f"--- FATAL ERROR (Frame {frame_count}) ---")
        traceback.print_exc()
        st.error("Processing stopped. Check console for details.")
        
    finally:
        # 10. Cleanup & Finalize
        cap.release()
        out.release()
        progress_bar.empty()
        st_status_text.empty()
        print("Video capture and writer released.")
    
    scores_path = os.path.join(tempfile.gettempdir(), "scores.json")
    
    try:
        with open(scores_path, 'w') as f:
            json.dump(all_scores, f, indent=2)
    except Exception as e:
        st.error(f"Failed to write scores.json file: {e}")
        traceback.print_exc() # Print the full error
        return out_path, None
    
    st.success("Video processing complete!")
    return out_path, scores_path

# =======================================================================
# SECTION 5: STREAMLIT APP UI
# (*** MODIFIED V18: Altair Heatmap Fix ***)
# =======================================================================

st.set_page_config(layout="wide", page_title="DVA Full Pipeline")
st.title("🏎️ DVA Full Pipeline: Sieve + DINO + VLM")
st.markdown("This app runs the full three-stage pipeline. The **Sieve** finds changes, **DINO** confirms them, and **Groq (Llama-4)** describes them.")

# --- Session State Initialization ---
if 'output_video_path' not in st.session_state:
    st.session_state.output_video_path = None
if 'output_scores_path' not in st.session_state:
    st.session_state.output_scores_path = None
if 'dino_analyzer' not in st.session_state:
    st.session_state.dino_analyzer = None
if 'vlm_auditor' not in st.session_state:
    st.session_state.vlm_auditor = None
if 'vlm_results_queue' not in st.session_state:
    st.session_state.vlm_results_queue = queue.Queue()

# Init DINO
if st.session_state.dino_analyzer is None:
    with st.spinner("Initializing DINOv2 Analyzer..."):
        export_dino_to_onnx()
        try:
            st.session_state.dino_analyzer = SemanticAnalyzer(dino_config)
        except Exception as e:
            st.error(f"Failed to initialize DINO Analyzer: {e}")
            st.stop()
else:
    st.success("✅ DINOv2 Analyzer is loaded and ready.")

# Init VLM (Groq)
if st.session_state.vlm_auditor is None:
    with st.spinner("Initializing Groq VLM Auditor..."):
        try:
            st.session_state.vlm_auditor = VLMAuditor(
                model_id="meta-llama/llama-4-maverick-17b-128e-instruct"
            )
            if not st.session_state.vlm_auditor.client:
                st.warning("VLM Auditor failed to init. Check .env file for GROQ_API_KEY.")
            else:
                st.success("✅ Groq Auditor (Llama-4) is loaded and ready.")
        except Exception as e:
            st.error(f"Failed to initialize Groq: {e}")
else:
    st.success("✅ Groq Auditor (Llama-4) is loaded and ready.")


# --- Sidebar for ALL Controls ---
st.sidebar.title("Audit Pipeline Controls")
uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

st.sidebar.divider()

# --- Stage 1 (Sieve) Controls ---
st.sidebar.subheader("Stage 1: Sieve (Fast Diff)")
sieve_mode = st.sidebar.selectbox(
    "Sieve Analysis Mode",
    ["Static Scene (Adaptive Baseline)", "Dynamic Scene (Consecutive Frame)"],
    help="**Static:** For security cams. **Dynamic:** Compares frame N vs N-X for F1/fast-moving."
)
sieve_pixel_thresh = st.sidebar.slider(
    "Sieve Pixel Threshold", 5, 100, 30, 1,
    help="A pixel is 'changed' if its 0-255 value differs by more than this. Lower = more sensitive."
)
sieve_min_pixels = st.sidebar.slider(
    "Sieve Min. Changed Pixels", 100, 50000, 1000, 100,
    help="The *number* of pixels that must change to trigger DINO. Higher = less sensitive."
)
is_dynamic = (sieve_mode == "Dynamic Scene (Consecutive Frame)")
sieve_sample_rate = st.sidebar.slider(
    "Sieve Sample Rate (1 Frame Every...)", 1, 30, 10, 1,
    disabled=is_dynamic, help="Only for Static Scene mode."
)
sieve_soft_update = st.sidebar.slider(
    "Sieve Baseline 'Blend' Rate", 0.01, 0.5, 0.02, 0.01,
    disabled=is_dynamic, help="Only for Static Scene mode."
)
sieve_dynamic_gap = st.sidebar.slider(
    "Dynamic Frame Gap (N vs N-X)",
    1, 30, 1, 1, 
    disabled=not is_dynamic,
    help="Only for Dynamic Scene mode. '1' (N vs N-1) is best for crashes."
)
if is_dynamic:
    st.sidebar.info(f"Dynamic Mode will compare Frame N vs Frame N-{sieve_dynamic_gap}.")

st.sidebar.divider()

# --- Stage 2 (DINO) Controls ---
st.sidebar.subheader("Stage 2: Gate (DINOv2)")
dino_profile_choice = st.sidebar.selectbox(
    "Load Base DINO Profile",
    options=[p.value for p in DetectionProfile],
    format_func=lambda x: PROFILES[DetectionProfile(x)].name,
    help="Select a profile to load its settings below. 'Stable' profiles use the Temporal Gate."
)
base_profile = PROFILES[DetectionProfile(dino_profile_choice)]
st.sidebar.markdown(f"**Editing Settings** (based on `{base_profile.name}`)")
ui_semantic_threshold = st.sidebar.slider(
    "DINO Semantic Threshold", 
    min_value=0.80, max_value=1.00, 
    value=base_profile.semantic_threshold, 
    step=0.001, format="%.3f",
    help="DINO's patch similarity sensitivity. Lower = less sensitive. Higher = more sensitive."
)
default_min_patches = base_profile.min_changed_patches(dino_config.NUM_PATCHES)
ui_min_patches = st.sidebar.slider(
    "DINO Min. Patches to Trigger",
    min_value=1, max_value=50, 
    value=default_min_patches, 
    step=1,
    help="The number of 16x16 patches that must change to trigger."
)
ui_top_k = st.sidebar.slider(
    "DINO Focus (Top-K Patches)",
    min_value=1, max_value=50,
    value=base_profile.top_k, 
    step=1,
    help="Instead of all changes, only mask the 'Top-K' *worst* patches."
)
ui_temporal_frames = st.sidebar.slider(
    "DINO Temporal Stability (Frames)",
    min_value=1, max_value=10,
    value=base_profile.temporal_stability_frames, 
    step=1,
    help=(
        "**This is the VLM fix.** How many consecutive frames DINO must trigger "
        "before calling the VLM.\n\n"
        "- **1**: Calls VLM on *every* DINO trigger (High traffic, noisy).\n"
        "- **3-5**: Calls VLM *once* per stable event (Low traffic, robust)."
    )
)
ui_change_ratio = ui_min_patches / dino_config.NUM_PATCHES
dino_profile_config = ThresholdConfig(
    name=f"Custom (Base: {base_profile.name})",
    description=base_profile.description,
    semantic_threshold=ui_semantic_threshold,
    change_ratio=ui_change_ratio,
    top_k=ui_top_k,
    temporal_stability_frames=ui_temporal_frames
)
st.sidebar.info(f"""
**Final DINO Settings:**
- Min Patches: **{ui_min_patches}**
- Semantic Threshold: {dino_profile_config.semantic_threshold:.3f}
- Temporal Gate: **{dino_profile_config.temporal_stability_frames} Frames**
""")

st.sidebar.divider()

# --- Stage 3 (VLM) Controls ---
st.sidebar.subheader("Stage 3: Auditor (Groq VLM)")
enable_vlm = st.sidebar.toggle(
    "Enable Groq Llama-4 Analysis", 
    value=True,
    help="If enabled, sends DINO-confirmed changes to Groq API for description. Requires a valid GROQ_API_KEY."
)
enable_vlm_focus = st.sidebar.toggle(
    "Enable VLM Focus (DINO-Guided ROI)",
    value=True,
    disabled=(not enable_vlm),
    help=(
        "**HIGHLY RECOMMENDED.** Uses DINO's detected patches to send a *cropped ROI sequence* to "
        "Groq. This gives maximum accuracy for defect analysis."
    )
)

vlm_sequence_size = st.sidebar.slider(
    "VLM Max Sequence Size (Frames)",
    min_value=2, max_value=16, # Llama-4 supports many, but 16 is a reasonable cap
    value=10, 
    step=1,
    disabled=(not enable_vlm),
    help=(
        "The *maximum* number of frames (e.g., a 'movie') to send to Groq. "
        "This includes 1 'Before' frame + (N-1) 'After' frames."
    )
)

vlm_prompt_choice = st.sidebar.selectbox(
    "VLM Prompt",
    options=["Default Audit", "F1 Scout"],
    help="**Default Audit:** 'Describe the change.' **F1 Scout:** 'Analyze the design difference.'"
)

if vlm_prompt_choice == "Default Audit":
    vlm_prompt = VLMAuditor.DEFAULT_AUDIT_FOCUSED_PROMPT if enable_vlm_focus else VLMAuditor.DEFAULT_AUDIT_PROMPT
else: # F1 Scout
    vlm_prompt = VLMAuditor.DEFAULT_SCOUT_FOCUSED_PROMPT if enable_vlm_focus else VLMAuditor.DEFAULT_SCOUT_PROMPT
st.sidebar.caption(f"**Using VLM Prompt:** *{vlm_prompt.splitlines()[0]}...*")

st.sidebar.divider()

# --- NEW V14: Stage 4 (Annotation) Controls ---
st.sidebar.subheader("Stage 4: Annotation")
ui_annotation_duration_sec = st.sidebar.slider(
    "Annotation Display Duration (Seconds)",
    min_value=1, max_value=30, value=5, step=1,
    help="How long the VLM analysis should stay on-screen *after* it is received."
)


# --- Main Page Logic ---
if st.sidebar.button("🚀 Process Video (Full Pipeline)", type="primary"):
    
    st.session_state.output_video_path = None
    st.session_state.output_scores_path = None

    if (uploaded_file and 
        st.session_state.dino_analyzer and 
        st.session_state.vlm_auditor):
        
        if enable_vlm and not st.session_state.vlm_auditor.client:
            st.error("VLM is enabled, but the Auditor is not initialized. Please check your GROQ_API_KEY in the .env file.")
            st.stop()
            
        temp_video_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_video_path = tmp.name
            
            st.info(f"Processing '{uploaded_file.name}' with Sieve, DINO (Stable={ui_temporal_frames}f), VLM (Enabled: {enable_vlm}), VLM Focus (Enabled: {enable_vlm_focus})...")
            
            while not st.session_state.vlm_results_queue.empty():
                try:
                    st.session_state.vlm_results_queue.get_nowait()
                except queue.Empty:
                    break
            
            output_video_path, output_scores_path = process_video(
                temp_video_path,
                sieve_mode,
                sieve_pixel_thresh,
                sieve_min_pixels,
                sieve_sample_rate,
                sieve_soft_update,
                sieve_dynamic_gap, 
                st.session_state.dino_analyzer,
                dino_profile_config,
                st.session_state.vlm_auditor,
                vlm_prompt,
                vlm_sequence_size,
                enable_vlm,
                enable_vlm_focus,
                st.session_state.vlm_results_queue,
                ui_annotation_duration_sec # <-- NEW V14 ARG
            )
            
            if output_video_path:
                st.session_state.output_video_path = output_video_path
            if output_scores_path:
                st.session_state.output_scores_path = output_scores_path

        except Exception as e:
            st.error(f"An unexpected error occurred in the main process: {e}")
            traceback.print_exc()
        
        finally:
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                    print(f"Successfully cleaned up temp file: {temp_video_path}")
                except Exception as e:
                    print(f"Warning: Could not delete temp file {temp_video_path}. Error: {e}")
                    st.warning(f"Could not delete temp file: {temp_video_path}. You may need to clear your temp directory manually.")

    elif not uploaded_file:
        st.sidebar.error("Please upload a video file first.")
    else:
        st.sidebar.error("One of the AI modules is not ready. Please reload.")

# --- Display results from session state ---
if st.session_state.output_video_path and os.path.exists(st.session_state.output_video_path):
    st.subheader("Processed Video Output")
    st.markdown("This video shows the final output. The header will display the Groq (Llama-4) analysis for confirmed damage.")
    
    try:
        with open(st.session_state.output_video_path, "rb") as file:
            st.download_button(
                label="Download Processed Video",
                data=file,
                file_name="pipeline_output.mp4",
                mime="video/mp4"
            )
        st.video(st.session_state.output_video_path)
    except FileNotFoundError:
        st.error("Processed video file not found. It may have been cleared from temp. Please re-run processing.")
        st.session_state.output_video_path = None # Clear bad path
    
if st.session_state.output_scores_path and os.path.exists(st.session_state.output_scores_path):
    
    # --- NEW V16: Analytics Dashboard (MODIFIED V18) ---
    try:
        st.subheader("📊 Pipeline Analytics Dashboard")
        data = pd.read_json(st.session_state.output_scores_path)
        data = data.dropna(subset=['type']) # Clean any potential bad rows

        tab1, tab2, tab3 = st.tabs(["Event Funnel & Summary", "Activity Timelines", "Change Heatmap"])

        with tab1:
            st.markdown("#### Event Filtering Funnel")
            st.markdown("This shows how the pipeline filters noisy events down to meaningful VLM calls.")
            
            # --- Vis 1: Event Filtering Funnel ---
            sieve_triggers = len(data[data['type'] == 'sieve_pixels'])
            dino_triggers = len(data[data['type'] == 'dino_semantic'])
            vlm_events = len(data[data['type'] == 'VLM_EVENT_SENT'])

            col1, col2, col3 = st.columns(3)
            col1.metric(label="Sieve Triggers (Pixel Noise)", value=sieve_triggers)
            col2.metric(label="DINO Triggers (Semantic Change)", value=dino_triggers)
            col3.metric(label="VLM Calls (Confirmed Events)", value=vlm_events)

            st.divider()

            # --- Vis 2: Alert Category Breakdown ---
            st.markdown("#### VLM Alert Categories")
            vlm_reports = data[data['type'] == 'VLM_EVENT_RECEIVED']
            
            if vlm_reports.empty:
                st.info("No VLM reports were received in this run.")
            else:
                st.markdown("This chart counts the most frequent descriptions from the VLM.")
                report_counts = vlm_reports['vlm_description'].value_counts()
                st.bar_chart(report_counts, use_container_width=True)

        with tab2:
            st.markdown("#### Sieve Activity Timeline (Pixel Noise)")
            st.markdown("This chart shows the raw pixel difference score over time. High spikes represent *any* change, including noise like shadows or lighting.")
            sieve_data = data[data['type'] == 'sieve_pixels'][['frame', 'score']].set_index('frame')
            if not sieve_data.empty:
                st.line_chart(sieve_data, use_container_width=True)
            else:
                st.info("No Sieve data was logged (this might happen if Sieve was disabled or the sample rate was too high).")
            
            st.divider()

            st.markdown("#### DINO Activity Timeline (Semantic Similarity)")
            st.markdown("This chart shows the semantic similarity score (1.0 = identical, < 1.0 = different). Notice how it only dips when a *meaningful* change occurs, ignoring the Sieve noise.")
            dino_data = data[data['type'] == 'dino_semantic'][['frame', 'score']].set_index('frame')
            if not dino_data.empty:
                st.line_chart(dino_data, use_container_width=True)
            else:
                st.info("No DINO events were triggered by the Sieve.")

        with tab3:
            st.markdown("#### Semantic Change Heatmap (DINO)")
            st.markdown(f"This 16x16 grid shows *where* in the frame DINO detected the most meaningful changes. This helps identify recurring problem areas (e.g., a specific part on a machine, a specific corner of a road).")
            
            heatmap_grid = np.zeros((dino_config.PATCHES_PER_DIM, dino_config.PATCHES_PER_DIM))
            dino_events = data[data['type'] == 'dino_semantic']
            
            patch_count = 0
            for coord_list in dino_events['changed_patch_coords']:
                if isinstance(coord_list, list):
                    for patch in coord_list:
                        row = int(patch['y1'] // dino_config.PATCH_SIZE)
                        col = int(patch['x1'] // dino_config.PATCH_SIZE)
                        if 0 <= row < 16 and 0 <= col < 16:
                            heatmap_grid[row, col] += 1
                            patch_count += 1
            
            if patch_count > 0:
                # --- *** MODIFIED V18: Heatmap Fix *** ---
                heatmap_df = pd.DataFrame(heatmap_grid)
                # Prepare data for Altair (long format)
                heatmap_df = heatmap_df.reset_index().rename(columns={'index': 'row'})
                heatmap_df_long = heatmap_df.melt('row', var_name='col', value_name='count')
                
                # Create Altair chart
                chart = alt.Chart(heatmap_df_long).mark_rect().encode(
                    x=alt.X('col:O', title='Patch Column', sort=None), # :O = Ordinal
                    y=alt.Y('row:O', title='Patch Row', sort=None),
                    color=alt.Color('count:Q', title='Detections', scale=alt.Scale(scheme='viridis')), # :Q = Quantitative
                    tooltip=['row', 'col', 'count']
                ).properties(
                    title="Heatmap of Detected Change Hotspots"
                ).interactive()
                
                st.altair_chart(chart, use_container_width=True)
                # --- *** END V18 MODIFICATION *** ---
            else:
                st.info("No DINO patch data was recorded to build a heatmap.")

    except Exception as e:
        st.error(f"Failed to generate analytics dashboard: {e}")
        traceback.print_exc()

    # --- Existing VLM Event Log Table ---
    st.subheader("VLM Event Log (Raw)")
    st.markdown("This JSON file contains the audit trail, including `VLM_EVENT_SENT` and `VLM_EVENT_RECEIVED` logs.")
    
    try:
        with open(st.session_state.output_scores_path, "rb") as file_data:
            st.download_button(
                label="Download Scores & VLM JSON",
                data=file_data,
                file_name="scores_and_vlm.json",
                mime="application/json"
            )
        
        # This dataframe is still useful for a raw log
        with open(st.session_state.output_scores_path, 'r') as f:
            scores_data = json.load(f)
            st.dataframe(scores_data, use_container_width=True)
            
    except FileNotFoundError:
        st.error("Scores JSON file not found. It may have been cleared from temp. Please re-run processing.")
        st.session_state.output_scores_path = None # Clear bad path
    except Exception as e:
        st.error(f"Error displaying raw JSON log: {e}")