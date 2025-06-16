# CIFAR-10 Image Classifier - Complete Streamlit App with Trained Model

import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Configure page
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="wide"
)

# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

@st.cache_resource
def load_trained_model():
    """Load the trained CNN model with weights"""
    try:
        # Loading the saved model
        if os.path.exists('cifar10_cnn_model.h5'):
            st.info("Loading the trained model (cifar10_cnn_model.h5)...")
            model = tf.keras.models.load_model('cifar10_cnn_model.h5')
            return model, True
        
        elif os.path.exists('cifar10_cnn_model.keras'):
            st.info("Loading the trained model (cifar10_cnn_model.keras)...")
            model = tf.keras.models.load_model('cifar10_cnn_model.keras')
            return model, True
            
        # If model doesn't exist, create architecture and load weights
        elif os.path.exists('cifar10_cnn.weights.h5'):
            st.info("Loading model architecture and trained weights...")
            model = create_model_architecture()
            model.load_weights('cifar10_cnn.weights.h5')
            return model, True
            
        else:
            st.warning("No trained model found. Using untrained model for demo.")
            st.write("Files checked: cifar10_cnn_model.h5, cifar10_cnn_model.keras, cifar10_cnn.weights.h5")
            model = create_model_architecture()
            return model, False
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.write("Using untrained model for demo...")
        model = create_model_architecture()
        return model, False

def create_model_architecture():
    """Create the CNN model architecture"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to 32x32 (CIFAR-10 size)
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def predict_image(model, image):
    """Make prediction on preprocessed image"""
    predictions = model.predict(image, verbose=0)
    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))
    
    return predicted_class, confidence, predictions[0]

def main():
    # Header
    st.title("CIFAR-10 Image Classifier")
    st.markdown("CNN for Image Classification")
    st.markdown("---")
    
    # Load model
    model, is_trained = load_trained_model()
    
    # Model status indicator
    if is_trained:
        st.success("Trained Model Loaded: - 77.06% Accuracy Expected!")
    else:
        st.warning("Demo Mode: - Using untrained model (low accuracy)")
    
    # Sidebar with model information
    with st.sidebar:
        st.header("Model Information")
        
        if is_trained:
            st.metric("Test Accuracy", "77.06%")
            st.metric("Parameters", "737,834")
            st.metric("Performance", "Excellent")
        else:
            st.metric("Demo Accuracy", "~10%")
            st.metric("Parameters", "737,834")
            st.metric("Status", "Untrained")
        
        st.write("Architecture: CNN")
        st.write("Dataset: CIFAR-10")
        st.write("Input Size: 32×32×3")
        
        st.header("CIFAR-10 Classes")
        class_emojis = ["✈️", "🚗", "🐦", "🐱", "🦌", "🐕", "🐸", "🐎", "🚢", "🚛"]
        for i, (emoji, class_name) in enumerate(zip(class_emojis, CLASS_NAMES)):
            st.write(f"{i}: {emoji} {class_name}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload & Classify")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Please upload images containing one of the following: airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, or trucks"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption=f"{uploaded_file.name}", use_container_width=True)
            
            # Image info
            file_size = len(uploaded_file.getvalue()) / 1024  # KB
            st.write(f"Size: {image.size[0]}×{image.size[1]} pixels | File: {file_size:.1f} KB")
            
            # Classify button
            if st.button("Proceed", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    try:
                        # Preprocess image
                        processed_image = preprocess_image(image)
                        
                        # Make prediction
                        predicted_class, confidence, all_predictions = predict_image(model, processed_image)
                        
                        # Store results for display
                        st.session_state['prediction_results'] = {
                            'predicted_class': predicted_class,
                            'confidence': confidence,
                            'all_predictions': all_predictions,
                            'is_trained': is_trained
                        }
                        
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
    
    with col2:
        if 'prediction_results' in st.session_state:
            results = st.session_state['prediction_results']
            
            st.header("Classification Results")
            
            predicted_class = results['predicted_class']
            confidence = results['confidence']
            all_predictions = results['all_predictions']
            is_trained_model = results['is_trained']
            
            # Main prediction
            st.success(f"Predicted Class: {CLASS_NAMES[predicted_class]}")
            st.info(f"Confidence: {confidence:.1%}")
            
            # Confidence interpretation
            if is_trained_model:
                if confidence > 0.8:
                    st.write("Very High Confidencee")
                elif confidence > 0.6:
                    st.write("High Confidence")
                elif confidence > 0.4:
                    st.write("Good Confidencee")
                else:
                    st.write("Low Confidence")
            else:
                st.write("Demo Mode (Untrained Model)")
            
            # Progress bar
            st.progress(confidence)
            
            # Top 3 predictions
            st.subheader("Top 3 Predictions")
            top_3_indices = np.argsort(all_predictions)[-3:][::-1]
            
            for i, idx in enumerate(top_3_indices):
                prob = all_predictions[idx]
                emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                class_emoji = ["✈️", "🚗", "🐦", "🐱", "🦌", "🐕", "🐸", "🐎", "🚢", "🚛"][idx]
                st.write(f"{emoji} **{class_emoji} {CLASS_NAMES[idx]}**: {prob:.1%}")
            
            # Probability chart
            st.subheader("All Probabilities")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['lightgreen' if i == predicted_class else 'lightblue' 
                     for i in range(len(CLASS_NAMES))]
            
            bars = ax.bar(CLASS_NAMES, all_predictions * 100, color=colors, 
                         edgecolor='darkblue', linewidth=1, alpha=0.8)
            
            ax.set_ylabel('Probability (%)')
            ax.set_title('Class Probabilities')
            plt.xticks(rotation=45)
            
            # Add labels for significant probabilities
            for bar, prob in zip(bars, all_predictions):
                height = bar.get_height()
                if height > 1:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Performance note
            if is_trained_model:
                st.success("Using trained model with 77.06% accuracy!")
            else:
                st.warning("Note: Load the trained model for accurate predictions!")
        
        else:
            st.header("How to Use")
            st.write("""
            Quick Start:
            1. Upload an image (PNG, JPG, JPEG)
            2. Click "Proceed"
            3. View detailed results
            
            Supported Classes:
            ✈️ airplane • 🚗 automobile • 🐦 bird • 🐱 cat • 🦌 deer  
            🐕 dog • 🐸 frog • 🐎 horse • 🚢 ship • 🚛 truck
            """)
    
    # Footer
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Target Accuracy", "77.06%")
    with col2:
        st.metric("Model Size", "8.53 MB")
    with col3:
        st.metric("Parameters", "737K")
    with col4:
        st.metric("Dataset", "CIFAR-10")

if __name__ == "__main__":
    main()