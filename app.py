import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import joblib
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array

IMG_SIZE = (224, 224)

st.set_page_config(page_title="The Leaf Clinic", page_icon="🌿", layout="centered")

# ---- Har bemari ke liye field notes (mashwara) ----
ADVICE = {
    "Wheat___Brown_Rust": "Brown rust (bhoori kungi) — a fungal infection. Remove affected leaves, "
                          "plant resistant varieties, and apply a recommended fungicide if it spreads. "
                          "For dosage and timing, consult your local agriculture officer.",
    "Wheat___Yellow_Rust": "Yellow rust (peeli kungi) — spreads fast in cool, moist weather. Scout the "
                           "field regularly, use resistant varieties, and consider a timely fungicide spray.",
    "Wheat___Healthy": "This leaf looks healthy. No disease detected — keep up the regular care and monitoring.",
}


@st.cache_resource
def load_feature_extractor():
    return MobileNetV2(weights="imagenet", include_top=False, pooling="avg",
                       input_shape=(224, 224, 3))


@st.cache_resource
def load_classifier():
    return joblib.load("plant_model.pkl"), joblib.load("class_names.pkl")


def nice_name(label):
    return label.replace("___", " · ").replace("__", " · ").replace("_", " ")


# =====================  DESIGN  =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600&family=Space+Mono:wght@400;700&display=swap');

:root{
  --canvas:#E9EFE3; --surface:#FCFDFA; --ink:#16321F; --muted:#5E6E60;
  --green:#2F6B43; --leaf:#5BA86F; --ochre:#B0731A; --line:#D5DECC;
}
.stApp{ background:
   radial-gradient(120% 80% at 50% -10%, #F2F6EC 0%, var(--canvas) 60%) fixed; }
#MainMenu, footer, header{visibility:hidden;}
.block-container{ max-width:720px; padding-top:2.2rem; padding-bottom:3rem; }
.stApp, .stApp p, .stApp div, .stApp span{ font-family:'Inter',sans-serif; color:var(--ink); }

.eyebrow{ font-family:'Space Mono',monospace; font-size:0.72rem; letter-spacing:0.18em;
   text-transform:uppercase; color:var(--muted); margin:0; }

/* ---- Hero ---- */
.hero{ border-bottom:1px solid var(--line); padding-bottom:20px; margin-bottom:26px; }
.hero h1{ font-family:'Fraunces',serif; font-weight:500; font-size:2.7rem; line-height:1.05;
   color:var(--ink); margin:10px 0 8px; letter-spacing:-0.01em; }
.hero .lede{ font-size:1rem; color:var(--muted); max-width:46ch; margin:0; line-height:1.5; }

/* ---- Empty hint ---- */
.hint{ font-family:'Space Mono',monospace; font-size:0.8rem; color:var(--muted);
   text-align:center; margin:8px 0 2px; letter-spacing:0.04em; }

/* ---- File uploader ---- */
[data-testid="stFileUploaderDropzone"]{
   background:var(--surface); border:1.5px dashed var(--leaf); border-radius:16px;
   padding:26px; transition:all .2s ease; }
[data-testid="stFileUploaderDropzone"]:hover{ border-color:var(--green); background:#F4F8EF; }
[data-testid="stFileUploaderDropzone"] *{ color:var(--muted) !important; }

/* ---- Uploaded image ---- */
[data-testid="stImage"] img{ border-radius:14px; box-shadow:0 8px 24px rgba(22,50,31,0.12); }

/* ---- Button ---- */
.stButton>button{
   width:100%; background:var(--green); color:#fff; border:none; border-radius:12px;
   padding:0.75rem 1rem; font-family:'Inter',sans-serif; font-weight:600;
   letter-spacing:0.01em; font-size:0.98rem; transition:all .18s ease; }
.stButton>button:hover{ background:#275A38; transform:translateY(-2px);
   box-shadow:0 8px 20px rgba(47,107,67,0.30); color:#fff; }

/* ---- Specimen card ---- */
.card{ background:var(--surface); border:1px solid var(--line); border-radius:18px;
   padding:26px 28px; margin-top:18px; position:relative; overflow:hidden;
   box-shadow:0 14px 40px rgba(22,50,31,0.10); animation:rise .45s ease both; }
.card:before{ content:""; position:absolute; left:0; top:0; bottom:0; width:6px; }
.card.healthy:before{ background:var(--leaf); }
.card.attention:before{ background:var(--ochre); }
@keyframes rise{ from{opacity:0; transform:translateY(14px);} to{opacity:1; transform:translateY(0);} }

.card-head{ display:flex; justify-content:space-between; align-items:center; }
.chip{ font-family:'Space Mono',monospace; font-size:0.68rem; letter-spacing:0.12em;
   text-transform:uppercase; padding:5px 11px; border-radius:99px; font-weight:700; }
.chip.healthy{ background:#E3F1E6; color:var(--green); }
.chip.attention{ background:#F6EAD4; color:var(--ochre); }

.dx{ font-family:'Fraunces',serif; font-weight:600; font-size:1.9rem; line-height:1.1;
   color:var(--ink); margin:8px 0 18px; }

.meter-label{ display:flex; justify-content:space-between; align-items:baseline;
   font-family:'Space Mono',monospace; font-size:0.72rem; letter-spacing:0.1em;
   text-transform:uppercase; color:var(--muted); margin-bottom:6px; }
.meter-label .val{ font-size:1.05rem; color:var(--ink); font-weight:700; letter-spacing:0; }
.track{ height:8px; background:#E7ECE0; border-radius:99px; overflow:hidden; }
.fill{ height:8px; border-radius:99px; animation:grow .6s ease both; }
.fill.healthy{ background:var(--leaf); }
.fill.attention{ background:var(--ochre); }
@keyframes grow{ from{width:0;} }

.notes-label{ margin:20px 0 6px; }
.notes{ font-size:0.95rem; line-height:1.6; color:#33402F; margin:0; }
</style>
""", unsafe_allow_html=True)

# ---- Hero ----
st.markdown("""
<div class="hero">
  <p class="eyebrow">Crop disease diagnostics</p>
  <h1>The Leaf Clinic</h1>
  <p class="lede">Photograph a wheat leaf and the model identifies the disease — and tells you what to do next.</p>
</div>
""", unsafe_allow_html=True)

feature_extractor = load_feature_extractor()
clf, class_names = load_classifier()

uploaded = st.file_uploader("Upload a leaf image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.markdown('<p class="hint">↑ Upload a leaf photo to begin</p>', unsafe_allow_html=True)
else:
    st.image(uploaded, use_container_width=True)

    if st.button("Diagnose leaf"):
        with st.spinner("Examining the leaf…"):
            img = load_img(uploaded, target_size=IMG_SIZE)
            arr = preprocess_input(np.expand_dims(img_to_array(img), axis=0))
            feat = feature_extractor.predict(arr, verbose=0)
            probs = clf.predict_proba(feat)[0]
            idx = int(np.argmax(probs))
            label = class_names[idx]
            conf = round(float(probs[idx]) * 100, 1)

        healthy = "healthy" in label.lower()
        status = "healthy" if healthy else "attention"
        chip_text = "Healthy" if healthy else "Needs attention"
        advice = ADVICE.get(label, "Consult your local agriculture expert about this condition.")

        st.markdown(f"""
        <div class="card {status}">
          <div class="card-head">
            <span class="eyebrow">Diagnosis</span>
            <span class="chip {status}">{chip_text}</span>
          </div>
          <h2 class="dx">{nice_name(label)}</h2>
          <div class="meter-label"><span>Confidence</span><span class="val">{conf}%</span></div>
          <div class="track"><div class="fill {status}" style="width:{conf}%"></div></div>
          <p class="eyebrow notes-label">Field notes</p>
          <p class="notes">{advice}</p>
        </div>
        """, unsafe_allow_html=True)