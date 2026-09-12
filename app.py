import streamlit as st
from transformers import pipeline
from deepface import DeepFace
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Tone / Brand-Fit Checker", layout="centered")


@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification",
                     model="SamLowe/roberta-base-go_emotions",
                     top_k=None)


emotion_text = load_emotion_model()

GO_EMOTIONS_TO_DEEPFACE = {
    'admiration': 'happy', 'amusement': 'happy', 'approval': 'happy',
    'desire': 'happy', 'excitement': 'happy', 'gratitude': 'happy',
    'joy': 'happy', 'love': 'happy', 'optimism': 'happy',
    'pride': 'happy', 'relief': 'happy', 'caring': 'happy',
    'anger': 'angry', 'annoyance': 'angry', 'disapproval': 'angry',
    'disgust': 'disgust',
    'fear': 'fear', 'nervousness': 'fear', 'embarrassment': 'fear',
    'sadness': 'sad', 'disappointment': 'sad', 'grief': 'sad', 'remorse': 'sad',
    'surprise': 'surprise', 'curiosity': 'surprise', 'realization': 'surprise',
    'confusion': 'neutral', 'neutral': 'neutral',
}


def bucket_scores(go_emotions_output):
    buckets = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0,
               'sad': 0, 'surprise': 0, 'neutral': 0}
    for item in go_emotions_output[0]:
        buckets[GO_EMOTIONS_TO_DEEPFACE[item['label']]] += item['score']
    return buckets


def tone_match_score(text_buckets, deepface_emotion_dict):
    df = {k: v / 100 for k, v in deepface_emotion_dict.items()}
    order = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    v1 = np.array([text_buckets[k] for k in order])
    v2 = np.array([df[k] for k in order])
    cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    top_text = max(text_buckets, key=text_buckets.get)
    top_face = max(df, key=df.get)
    return {
        'match_score': round(float(cosine_sim), 3),
        'intended_top': top_text,
        'detected_top': top_face,
        'flag': 'MISMATCH' if top_text != top_face else 'MATCH',
        'text_buckets': text_buckets,
        'face_buckets': df,
    }


def check_tone_fit(intended_tone_text, image_path):
    go_output = emotion_text(intended_tone_text)
    text_buckets = bucket_scores(go_output)
    try:
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'],
                                   detector_backend='mtcnn')
        deepface_emotions = result[0]['emotion']
    except Exception as e:
        return {'error': f"Could not analyze face: {e}"}
    return tone_match_score(text_buckets, deepface_emotions)


st.title("Tone / Brand-Fit Checker")
st.caption(
    "Checks whether a generated actor's delivered expression matches the "
    "intended script tone — a QA layer for producing creative content at scale."
)

DEMO_EXAMPLES = {
    "Happy — matches": ("examples/happy.jpg", "This is amazing, I'm so excited!"),
    "Surprised — matches": ("examples/surprised.jpg", "Whoa, I really didn't see that coming!"),
    "Sad — edge case": ("examples/sad.jpg", "I'm honestly really disappointed by this."),
    "Calm — edge case": ("examples/calm.jpg", "Everything's calm, we've got this handled."),
    "Flat delivery — mismatch": ("examples/flat1.jpg", "This is the best news ever, I'm so thrilled!"),
    "Flat delivery 2 — mismatch": ("examples/flat2.jpg", "I can't stop laughing, this is hilarious!"),
}

st.subheader("Try a demo example")
choice = st.selectbox("Pick one:", list(DEMO_EXAMPLES.keys()))
demo_img, demo_tone = DEMO_EXAMPLES[choice]

st.subheader("Or use your own")
uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
custom_tone = st.text_input("Intended tone / script line", "")

if st.button("Check tone fit", type="primary"):
    if uploaded is not None and custom_tone:
        image = Image.open(uploaded)
        image.save("temp_upload.jpg")
        img_path, tone = "temp_upload.jpg", custom_tone
    else:
        img_path, tone = demo_img, demo_tone

    if not os.path.isfile(img_path):
        st.error(f"Couldn't find {img_path} — check the examples/ folder is present.")
    else:
        st.image(img_path, width=250)
        with st.spinner("Analyzing..."):
            result = check_tone_fit(tone, img_path)

        if 'error' in result:
            st.error(result['error'])
        else:
            col1, col2 = st.columns(2)
            col1.metric("Match score", result['match_score'])
            col2.metric("Result", result['flag'])
            st.write(f"**Intended:** {result['intended_top']}  |  "
                     f"**Detected:** {result['detected_top']}")
            st.bar_chart({
                "Intended (text)": result['text_buckets'],
                "Detected (face)": result['face_buckets'],
            })
