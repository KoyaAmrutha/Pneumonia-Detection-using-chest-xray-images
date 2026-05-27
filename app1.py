import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError

# Title
st.title("🫁 Pneumonia Detection using Chest X-ray")
st.write("Upload a chest X-ray image to detect pneumonia with probability.")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pneumonia_model.h5", compile=False)

model = load_model()

# Upload
uploaded_file = st.file_uploader(
    "Choose an X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    if uploaded_file.name.startswith("._"):
        st.error("Invalid hidden file.")
        st.stop()

    try:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", width=300)

        # Preprocess
        image = image.convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict (faster)
        prediction = model(img_array, training=False).numpy()

        # REMOVE DEBUG AFTER TESTING
        st.write("🔍 Raw Output:", prediction)

        # ✅ HANDLE OUTPUT
        if prediction.shape[1] == 1:
            prob = float(prediction[0][0])

            # 🔥 TRY BOTH LABEL LOGICS (IMPORTANT FIX)
            pneumonia_prob = prob
            normal_prob = 1 - prob

            # 👉 Uncomment this IF results are reversed
            # pneumonia_prob = 1 - prob
            # normal_prob = prob

        else:
            normal_prob = float(prediction[0][0])
            pneumonia_prob = float(prediction[0][1])

        # Percent
        pneumonia_percent = pneumonia_prob * 100
        normal_percent = normal_prob * 100

        # Results
        st.subheader("📊 Prediction Results")
        st.write(f"🫁 Pneumonia: **{pneumonia_percent:.2f}%**")
        st.write(f"🫁 Normal: **{normal_percent:.2f}%**")

        st.progress(min(int(pneumonia_percent), 100))

        # Final prediction
        if pneumonia_prob >= 0.5:
            st.error("⚠️ Pneumonia Detected")
        else:
            st.success("✅ Normal Lungs")

        # Severity
        st.subheader("🩺 Severity Level")

        if pneumonia_prob > 0.8:
            st.error("High Severity 🚨")
        elif 0.5 <= pneumonia_prob < 0.8:
            st.warning("Moderate Severity ⚠️")
        else:
            st.info("Low / No Pneumonia ✅")

    except UnidentifiedImageError:
        st.error("Invalid image file.")

    except Exception as e:
        st.error("Error processing image.")
        st.write(e)