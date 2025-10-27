import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess_input
from PIL import Image


RESNET_MODEL_PATH = "https://huggingface.co/min190503/RN5/resolve/main/best_model_finetuned_RN.h5"
CUSTOM_CNN_MODEL_PATH = "https://huggingface.co/min190503/RN5/resolve/main/best_model_CNN.h5"


RESNET_IMG_SIZE = (256, 256) 
CUSTOM_CNN_IMG_SIZE = (224, 224) 

CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
               'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
               'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
               'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 
               'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
               'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
               'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
               'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
               'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']



@st.cache_resource
def load_app_model(model_path):
    try:
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình tại '{model_path}': {e}")
        return None

resnet_model = load_app_model(RESNET_MODEL_PATH)
custom_cnn_model = load_app_model(CUSTOM_CNN_MODEL_PATH)

# Hàm Tiền xử lý

def preprocess_for_resnet(image_pil):
    """Tiền xử lý cho ResNetV2 (chuẩn hóa [-1, 1])"""
    img = image_pil.resize(RESNET_IMG_SIZE) 
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array.astype(np.float32)
    img_array_preprocessed = resnet_preprocess_input(img_array)
    return np.expand_dims(img_array_preprocessed, axis=0)

def preprocess_for_custom_cnn(image_pil):
    """Tiền xử lý cho CNN cơ bản (chuẩn hóa [0, 1])"""
    img = image_pil.resize(CUSTOM_CNN_IMG_SIZE) 
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array_rescaled = img_array.astype(np.float32) / 255.0
    return np.expand_dims(img_array_rescaled, axis=0)

# Giao diện Web

st.set_page_config(layout="wide") 
st.title("🌿 So sánh Mô hình Nhận diện Bệnh lá cây")

col1, col2 = st.columns([1, 1])

# Cột 1
with col1:
    st.header("Ảnh Đầu vào")
    uploaded_file = st.file_uploader("Tải lên một ảnh lá cây", type=["jpg","jpeg","png"])
    
    image_pil = None
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert('RGB')
        st.image(image_pil, caption="Ảnh đã tải lên", use_container_width=True)

#Cột 2: Hiển thị kết quả
with col2:
    st.header("Kết quả Chẩn đoán")
    
    if image_pil is not None and st.button('Bắt đầu Chẩn đoán'):
        
        if resnet_model is None or custom_cnn_model is None:
            st.error("Một trong hai mô hình chưa được tải thành công. Vui lòng kiểm tra lại đường dẫn.")
        else:
            with st.spinner('Đang phân tích... Xin vui lòng chờ.'):
                
                #  ResNet50V2
                st.subheader("Mô hình ResNet50V2 (Học Chuyển giao)")
                img_input_resnet = preprocess_for_resnet(image_pil)
                pred_resnet = resnet_model.predict(img_input_resnet)[0]
                
                conf_resnet = np.max(pred_resnet)
                class_resnet = CLASS_NAMES[np.argmax(pred_resnet)]
                
                st.success(f"Kết quả: {class_resnet.replace('___', ' - ')}")
                
                st.divider() 

                #CNN
                st.subheader("Mô hình CNN Cơ bản (Tự xây dựng)")
                img_input_custom = preprocess_for_custom_cnn(image_pil)
                pred_custom = custom_cnn_model.predict(img_input_custom)[0]
                
                conf_custom = np.max(pred_custom)
                class_custom = CLASS_NAMES[np.argmax(pred_custom)]
                
                st.warning(f"Kết quả: {class_custom.replace('___', ' - ')}")

    elif image_pil is None:
        st.info("Vui lòng tải ảnh lên ở cột bên trái và nhấn nút 'Bắt đầu Chẩn đoán'.")

