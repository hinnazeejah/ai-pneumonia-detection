
import time
from pathlib import Path
from PIL import Image, ImageOps

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

# import your inference class
from app.inference import InferenceModel

# ----------------------------
# CONFIG
# ----------------------------

# Pink theme palette
PINK_BG      = "#ffe4f0"  # app background
PINK_DARK    = "#ff7ab6"  # accents
PINK_MED     = "#ff9ac8"  # secondary
PINK_LIGHT   = "#ffd2e6"  # light panels
TEXT_DARK    = "#2a2a2a"

# ----------------------------
# Helper functions
# ----------------------------
def format_percent(x: float) -> str:
    return f"{x*100:.1f}%"

def resize_preview(pil_img: Image.Image, max_side=400) -> Image.Image:
    # simple square-pad & resize for preview
    img = pil_img.convert("RGB")
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    scale = min(max_side / w, max_side / h)
    new_w, new_h = int(w*scale), int(h*scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


# ----------------------------
# App
# ----------------------------
class CuteMedApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window
        self.title("AI-MedVision: Advanced Pneumonia Detection System 🏥")
        self.geometry("950x640")
        self.minsize(900, 600)

        # customtkinter appearance
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")  # built-in theme; we'll also set custom colors

        self.configure(fg_color=PINK_BG)

        # State
        self.model: InferenceModel | None = None
        self.current_img: Image.Image | None = None

        # Layout: left (controls) / right (preview)
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.build_left_panel()
        self.build_right_panel()

    # ---------- UI ----------
    def build_left_panel(self):
        panel = ctk.CTkFrame(self, fg_color=PINK_LIGHT, corner_radius=16)
        panel.grid(row=0, column=0, sticky="nsw", padx=14, pady=14)
        panel.grid_propagate(False)
        panel.configure(width=300)

        title = ctk.CTkLabel(panel, text="AI-MedVision", font=("SF Pro Text", 24, "bold"), text_color=TEXT_DARK)
        title.pack(pady=(30, 20))

        # Load Model Button
        self.load_btn = ctk.CTkButton(panel, text="Load Model", command=self.load_model, 
                                     fg_color=PINK_DARK, hover_color=PINK_MED, height=50,
                                     font=("SF Pro Text", 16, "bold"))
        self.load_btn.pack(pady=(0, 10), padx=20, fill="x")

        self.model_status = ctk.CTkLabel(panel, text="Model not loaded", text_color="#666", font=("SF Pro Text", 12))
        self.model_status.pack(pady=(0, 30))

        # Upload Image Button
        self.upload_btn = ctk.CTkButton(panel, text="Upload Image", command=self.open_image, 
                                       fg_color=PINK_DARK, hover_color=PINK_MED, height=50,
                                       font=("SF Pro Text", 16, "bold"))
        self.upload_btn.pack(pady=(0, 10), padx=20, fill="x")

        # Predict Button
        self.pred_btn = ctk.CTkButton(panel, text="Predict", command=self.run_predict, 
                                     fg_color=PINK_DARK, hover_color=PINK_MED, height=50,
                                     font=("SF Pro Text", 16, "bold"))
        self.pred_btn.pack(pady=(0, 30), padx=20, fill="x")

        # Results
        res_box = ctk.CTkFrame(panel, fg_color="white", corner_radius=12)
        res_box.pack(fill="x", padx=20, pady=(0, 20))

        ctk.CTkLabel(res_box, text="Result", font=("SF Pro Text", 18, "bold"), text_color=TEXT_DARK)\
            .pack(pady=(15, 10))
        
        self.label_out = ctk.CTkLabel(res_box, text="—", font=("SF Pro Text", 36, "bold"))
        self.label_out.pack(pady=(0, 15))

    def build_right_panel(self):
        right = ctk.CTkFrame(self, fg_color="white", corner_radius=16)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 14), pady=14)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.preview_label = ctk.CTkLabel(right, text="Open an image to preview", text_color="#777")
        self.preview_label.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)

    # ---------- Actions ----------

    def load_model(self):
        try:
            # Auto-load the default model
            mp = "models/best.pt"
            t0 = time.time()
            self.model = InferenceModel(mp)
            dt = time.time() - t0
            self.model_status.configure(text=f"✅ Model loaded ({dt:.2f}s)", text_color="#12a454")
            self.load_btn.configure(text="Model Loaded ✓", state="disabled")
        except Exception as e:
            self.model_status.configure(text=f"❌ Error: {str(e)}", text_color="#ff4444")
            messagebox.showerror("Error loading model", str(e))

    def open_image(self):
        path = filedialog.askopenfilename(title="Select chest X-ray",
                                          filetypes=[("Images", "*.jpg *.jpeg *.png"), ("All files", "*.*")])
        if not path:
            return
        try:
            pil = Image.open(path).convert("RGB")
            self.current_img = pil
            self.show_preview(pil)
            self.clear_outputs()
        except Exception as e:
            messagebox.showerror("Image error", f"Could not open image:\n{e}")

    def show_preview(self, pil_img: Image.Image):
        img = resize_preview(pil_img, max_side=520)
        self._tkimg = ctk.CTkImage(light_image=img, size=img.size)  # keep ref to avoid GC
        self.preview_label.configure(image=self._tkimg, text="")

    def clear_outputs(self):
        self.label_out.configure(text="—", text_color=TEXT_DARK)

    def run_predict(self):
        if self.model is None:
            messagebox.showwarning("Model not loaded", "Load a model first.")
            return
        if self.current_img is None:
            messagebox.showwarning("No image", "Upload an image first.")
            return
        
        # Show loading state
        self.pred_btn.configure(text="Predicting...", state="disabled")
        self.label_out.configure(text="⏳ Analyzing...", text_color="#666")
        self.update()  # Force UI update
        
        try:
            # Use default threshold of 0.85
            out = self.model.predict(self.current_img, threshold=0.85)
            label = out["label"]
            conf  = out["confidence"]

            # Show result with color coding
            color = PINK_DARK if label.upper() == "PNEUMONIA" else "#12a454"  # green for NORMAL
            self.label_out.configure(text=f"{label}\n{format_percent(conf)}", text_color=color)

        except Exception as e:
            messagebox.showerror("Prediction error", str(e))
        finally:
            # Restore button state
            self.pred_btn.configure(text="Predict", state="normal")


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    app = CuteMedApp()
    app.mainloop()
