import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import io
from model import create_resnet_model
from utils.data_loader import get_single_image_transform

# Page config
st.set_page_config(
    page_title="Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    device = torch.device("cpu")
    
    try:
        # Load checkpoint
        checkpoint = torch.load('models/best_model.pth', map_location=device)
        class_names = checkpoint['class_names']
        
        # Create and load model
        model = create_resnet_model(num_classes=len(class_names))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, class_names, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def predict_image(image, model, class_names, device):
    """Make prediction on uploaded image"""
    # Transform image
    transform = get_single_image_transform()
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item(), probabilities[0]

# Recycling information
RECYCLING_INFO = {
    'paper': {
        'emoji': '📄',
        'instructions': 'Place in paper recycling bin. Remove any plastic windows or staples.',
        'examples': 'Newspapers, magazines, office paper, cardboard boxes'
    },
    'cardboard': {
        'emoji': '📦',
        'instructions': 'Flatten boxes and place in cardboard recycling. Keep dry.',
        'examples': 'Shipping boxes, cereal boxes, paper tubes'
    },
    'plastic': {
        'emoji': '🥤',
        'instructions': 'Rinse clean and check recycling number. Most #1-7 accepted.',
        'examples': 'Bottles, containers, bags, packaging'
    },
    'metal': {
        'emoji': '🥫',
        'instructions': 'Rinse cans and remove labels. Aluminum and steel accepted.',
        'examples': 'Soda cans, tin cans, aluminum foil'
    },
    'glass': {
        'emoji': '🍾',
        'instructions': 'Rinse and remove lids. Separate by color if required.',
        'examples': 'Bottles, jars, containers'
    },
    'trash': {
        'emoji': '🗑️',
        'instructions': 'Non-recyclable waste. Dispose in general waste bin.',
        'examples': 'Contaminated items, mixed materials, broken items'
    }
}

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">♻️ Waste Classification System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image to classify waste and get recycling instructions</p>', unsafe_allow_html=True)
    
    # Load model
    model, class_names, device = load_model()
    
    if model is None:
        st.error("Failed to load model. Please ensure 'models/best_model.pth' exists.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write("""
        This AI-powered app classifies waste into different categories 
        and provides recycling instructions.
        """)
        st.write(f"**Categories:** {', '.join(class_names)}")
        st.write("**Model:** ResNet50 with Transfer Learning")
        
        st.header("Tips for Best Results")
        st.write("""
        - Use clear, well-lit photos
        - Center the waste item in frame
        - Avoid cluttered backgrounds
        - One item at a time works best
        """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of waste item"
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption='Uploaded Image', use_container_width=True)
        
        with col2:
            with st.spinner('Analyzing...'):
                # Make prediction
                predicted_idx, confidence, probabilities = predict_image(
                    image, model, class_names, device
                )
                predicted_class = class_names[predicted_idx]
                
                # Display result
                st.success("Classification Complete!")
                
                info = RECYCLING_INFO.get(predicted_class, RECYCLING_INFO['trash'])
                
                st.markdown(f"### {info['emoji']} {predicted_class.upper()}")
                st.metric("Confidence", f"{confidence*100:.1f}%")
                
                # Progress bar for confidence
                st.progress(confidence)
        
        # Recycling instructions
        st.markdown("---")
        st.subheader("♻️ Recycling Instructions")
        
        info = RECYCLING_INFO.get(predicted_class, RECYCLING_INFO['trash'])
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write("**How to dispose:**")
            st.info(info['instructions'])
        
        with col2:
            st.write("**Examples:**")
            st.info(info['examples'])
        
        # All predictions
        with st.expander("📊 View All Class Probabilities"):
            for i, class_name in enumerate(class_names):
                prob = probabilities[i].item() * 100
                st.write(f"**{class_name.capitalize()}:** {prob:.2f}%")
                st.progress(prob/100)
    
    else:
        # Instructions when no image uploaded
        st.info("👆 Upload an image to get started!")
        
        # Example categories
        st.subheader("Waste Categories")
        cols = st.columns(3)
        for idx, (category, info) in enumerate(RECYCLING_INFO.items()):
            with cols[idx % 3]:
                st.markdown(f"### {info['emoji']} {category.capitalize()}")
                st.caption(info['examples'])

if __name__ == '__main__':
    main()