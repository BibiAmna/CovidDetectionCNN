# Importing Libraries
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import base64

# Load the pre-trained model
model = tf.keras.models.load_model("model1.h5")

# Constants for image preprocessing
IMG_HEIGHT, IMG_WIDTH = 200, 200

# Function to preprocess the image
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    image = np.array(image)
    image = 255 - image
    image = image / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI setup
st.set_page_config(page_title="COVID Detection from Chest X-Ray", layout="wide")

# Load and encode the background image
background_image_path = r"covid.jpeg"  # Change to your image path
def get_base64_image(image_file):
    with open(image_file, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

background_image = get_base64_image(background_image_path)

# Add custom CSS for styling
st.markdown(f"""
    <style>
        .header {{
            background-image: url("data:image/jpeg;base64,{background_image}");
            background-size: 100% 100%;
            # color: white;
            background-color: #fac0c0;
            display: flex;
            align-items: centre;
            padding: 30px;
            margin-bottom: 10px;
        }}
        .header h1 {{
            font-size: 2.5em;
            color: white;
        }}
        .content-container {{
            background-color: #ffffff;
            padding: 30px;
            border-radius: 8px;
        }}
    </style>
""", unsafe_allow_html=True)

# Header with image and title
st.markdown(
    f"""
    <div class="header">
        <h1>COVID Detection from Chest X-Ray</h1>
    </div>
    """, unsafe_allow_html=True
)

# Tabs for navigation
tabs = st.tabs(["Instructions", "Symptoms", "Diagnosis Methods", "Prevention", "Acknowledgment"])

# Main content for each tab
with tabs[0]:
    st.subheader("Instructions")
    st.write("1. Upload a chest X-ray image.")
    st.write("2. The model will predict if the X-ray is COVID-affected or normal.")
    st.write("3. For best results, ensure the image is clear and well-lit.")
    st.write("4. Consult a medical professional for further analysis.")

    uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file)

        # Set the desired width for display
        display_width = 250  # Adjusted width for the image

        # Create columns for displaying the image and predictions
        col1, col2 = st.columns(2)

        with col1:
            # Display the image with the specified width
            st.image(image, caption="Uploaded X-Ray Image", width=display_width)

        with col2:
            # Preprocess the uploaded image
            processed_image = preprocess_image(image)

            # Loading spinner while making predictions
            with st.spinner("Processing..."):
                # Make a prediction
                prediction = model.predict(processed_image)
                predicted_class = "COVID-Positive" if prediction > 0.7 else "Normal"
                confidence = prediction[0][0] if prediction > 0.5 else 1 - prediction[0][0]

            # Display the prediction result
            st.markdown(f"### Prediction: {predicted_class}")
            st.markdown(f"### Confidence: {confidence:.2%}")

            # Optional: display a message
            if predicted_class == "COVID-Positive":
                st.warning("This X-ray shows signs of COVID-19. Consult a medical professional for further analysis.")
            else:
                st.success(
                    "This X-ray does not show signs of COVID-19. For more assurance, consult a medical professional.")

with tabs[1]:
    st.subheader("Symptoms of COVID-19")
    st.write(""" 
    - Fever or chills
    - Cough
    - Shortness of breath or difficulty breathing
    - Fatigue
    - Muscle or body aches
    - Headache
    - New loss of taste or smell
    - Sore throat
    - Congestion or runny nose
    - Nausea or vomiting
    - Diarrhea
    """)

with tabs[2]:
    st.subheader("Methods for Diagnosis")
    st.write("""
    - **RT-PCR Test**: A lab test that detects the virus's genetic material.
    - **Rapid Antigen Test**: A quick test for viral proteins.
    - **Chest X-Ray/CT Scan**: Imaging tests for complications caused by the virus.
    """)

with tabs[3]:
    st.subheader("Prevention Strategies")
    st.write("""
    - Wear masks in crowded spaces.
    - Maintain physical distance.
    - Wash hands with soap and water.
    - Avoid large gatherings.
    - Get vaccinated.
    """)
with tab[4]:
    st.subheader("Acknowledgment")
    st.write("This project was completed by Bibi Amna and Samavia Hussain Raja as part of their academic research.")

st.markdown('</div>', unsafe_allow_html=True)  # End content container
