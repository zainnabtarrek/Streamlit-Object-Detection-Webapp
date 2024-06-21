from pathlib import Path
import PIL
import numpy as np
import streamlit as st
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Object Detection And Tracking using Streamlit and YOLOv8")
st.sidebar.header("Model")

model_type = st.sidebar.radio("Select Task", ['Detection'])
confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100

if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)

try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image Config")
source_radio = st.sidebar.radio("Select Source", settings.SOURCES_LIST)

source_img = None
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png"))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image", use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)

                st.image(source_img, caption="Uploaded Image", use_column_width=True)
                st.write(f"Uploaded image shape: {np.array(uploaded_image).shape}")
                st.write(f"Uploaded image dtype: {np.array(uploaded_image).dtype}")
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image', use_column_width=True)
        else:
            if st.sidebar.button('Analyse Image'):
                try:
                    res = model.predict(np.array(uploaded_image), conf=confidence)
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]
                    st.image(res_plotted, caption='Detected Image', use_column_width=True)

                    with st.expander("Detection Results"):
                        detected_classes = []
                        for box in boxes:
                            class_name = box.class_name
                            detected_classes.append(class_name)
                            st.write(box.data)

                        st.write("Detected Classes:")
                        for class_name in detected_classes:
                            st.write(class_name)

                except Exception as ex:
                    st.error("Error occurred during image analysis.")
                    st.error(ex)
else:
    st.error("Please select a valid source type!")
