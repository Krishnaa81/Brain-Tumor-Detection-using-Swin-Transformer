import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import timm

st.title("🧠 Brain Tumor Detection")

class SwinTransformerBrainTumor(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('swin_brain_tumor_complete.pth', map_location=device, weights_only=False)
    
    # CRITICAL: Use exact class names from checkpoint to match training order
    class_names = checkpoint['class_names']
    num_classes = checkpoint['num_classes']
    
    model = SwinTransformerBrainTumor(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Display model info
    st.sidebar.write(f"✅ Model loaded successfully")
    st.sidebar.write(f"📊 Training Accuracy: {checkpoint.get('accuracy', 0)*100:.2f}%")
    st.sidebar.write(f"🎯 Classes: {', '.join(class_names)}")
    
    return model, device, class_names

model, device, class_names = load_model()

uploaded_file = st.file_uploader("Upload MRI Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, width=300)
    
    if st.button("Predict"):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        result = class_names[predicted.item()]
        conf = confidence.item() * 100
        
        st.success(f"**Prediction:** {result.upper()}")
        st.info(f"**Confidence:** {conf:.2f}%")
        
        st.write("**All Probabilities:**")
        for i, cls in enumerate(class_names):
            st.write(f"{cls}: {probs[0][i].item()*100:.2f}%")
