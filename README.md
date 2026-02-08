# 🧠 Brain Tumor Detection System using Swin Transformer

A professional AI-powered medical imaging application for automated brain tumor classification from MRI scans using state-of-the-art Swin Transformer architecture. This comprehensive system achieves **99.08% accuracy** and includes tumor size estimation, medication recommendations, and professional PDF report generation.

## ✨ Key Features

### 🎯 AI-Powered Diagnosis
- **99.08% Accuracy** using Swin Transformer deep learning model
- Real-time MRI scan analysis
- Confidence score for each prediction
- Probability distribution across all tumor types

### 📏 Tumor Size Estimation
- Automated tumor region detection using advanced image processing
- Size categorization (Small, Medium, Large, Very Large)
- Percentage of affected brain area calculation
- Visual tumor region highlighting with bounding boxes

### 💊 Medication Recommendations
- Comprehensive medication database for each tumor type
- **13 different medications** with detailed information:
  - Generic and brand names
  - Purpose and mechanism of action
  - Typical dosage guidelines
  - Common side effects
  - Important precautions
- Specialized drugs for Glioma, Meningioma, and Pituitary tumors

### 📄 Professional PDF Reports
- Complete medical report generation
- Patient information section
- MRI scan image inclusion
- Diagnosis results with confidence scores
- Tumor size analysis
- Probability distribution tables
- Disease information and symptoms
- Medical recommendations
- Detailed medication information
- Professional formatting with color-coded sections

### 🎨 Modern User Interface
- Clean, professional medical-grade design
- Gradient color scheme (purple/blue theme)
- Responsive two-column layout
- Interactive expandable sections
- Progress bars for probability visualization
- Card-based information display
- Mobile-friendly design

## 📋 Tumor Classification

This application classifies brain MRI scans into four categories:
- **Glioma** - Tumors originating from glial cells (High severity)
- **Meningioma** - Tumors arising from the meninges (Moderate severity)
- **Pituitary** - Tumors in the pituitary gland (Moderate severity)
- **No Tumor** - Healthy brain tissue

## 🎯 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 99.08% |
| Precision | 99.11% |
| Recall | 99.08% |
| F1-Score | 99.09% |

## 🏗️ Architecture

- **Model:** Swin Transformer Base (swin_base_patch4_window7_224)
- **Input Size:** 224×224×3
- **Framework:** PyTorch
- **Pre-training:** ImageNet weights
- **Total Parameters:** ~88M
- **Classes:** 4 (Glioma, Meningioma, No Tumor, Pituitary)

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Git (for cloning)

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/Krishnaa81/Brain-Tumor-Detection-using-Swin-Transformer.git
cd Brain-Tumor-Detection-using-Swin-Transformer
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download the trained model:**
   - The model file `swin_brain_tumor_complete.pth` (331 MB) should be in the project root
   - If missing, download from your model storage location
   - Place the downloaded file in the project root directory

4. **Run the application:**
```bash
streamlit run app.py
```

5. **Access the application:**
   - Open your browser and navigate to `http://localhost:8501`
   - The app will automatically open in your default browser

## 🚀 Usage Guide

### Step 1: Upload MRI Scan
- Click "Browse files" or drag and drop an MRI image
- Supported formats: PNG, JPG, JPEG
- Ensure the image is clear and properly oriented

### Step 2: Enter Patient Information
- **Patient Name**: Full name of the patient
- **Patient ID**: Unique identifier (e.g., P12345)
- **Age**: Patient's age
- **Gender**: Select from dropdown (Male/Female/Other)

### Step 3: Analyze
- Click the "🚀 Analyze MRI Scan" button
- Wait for the AI to process the image (usually 2-5 seconds)

### Step 4: Review Results
- **Diagnosis**: Primary prediction with confidence score
- **Tumor Size**: Size category and affected area percentage
- **Probability Distribution**: Confidence across all tumor types
- **Disease Information**: Detailed description and symptoms
- **Medications**: Recommended drugs with complete details

### Step 5: Download Report
- Fill in all patient information fields
- Click "📥 Download PDF Report"
- Save the professional medical report for records

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 99.08% |
| **Precision** | 99.11% |
| **Recall** | 99.08% |
| **F1-Score** | 99.09% |

## 🏗️ Technical Architecture

### Model Specifications
- **Architecture:** Swin Transformer Base (swin_base_patch4_window7_224)
- **Input Size:** 224×224×3 RGB images
- **Framework:** PyTorch with timm library
- **Pre-training:** ImageNet weights
- **Total Parameters:** ~88 million
- **Classes:** 4 (Glioma, Meningioma, No Tumor, Pituitary)

### Tumor Size Detection
- **Method:** OpenCV-based image processing
- **Techniques:** 
  - Adaptive thresholding
  - Morphological operations
  - Contour detection and analysis
- **Output:** Size category and percentage of affected area

### PDF Generation
- **Library:** ReportLab
- **Features:** Professional medical report formatting
- **Includes:** Patient info, MRI image, diagnosis, medications

## 📁 Project Structure

```
Brain-Tumor-Detection-using-Swin-Transformer/
├── app.py                          # Main Streamlit application with UI
├── swin_brain_tumor_complete.pth   # Trained model weights (331 MB)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation (this file)
├── SETUP_GUIDE.md                 # Detailed setup and feature guide
├── .gitignore                      # Git ignore rules
└── .git/                          # Git repository data
```

## 🔧 Dependencies

```txt
streamlit>=1.28.0          # Web application framework
torch>=2.0.0               # Deep learning framework
torchvision>=0.15.0        # Computer vision library
timm>=0.9.0                # PyTorch Image Models
Pillow>=10.0.0             # Image processing
numpy>=1.24.0              # Numerical computing
reportlab>=4.0.0           # PDF generation
opencv-python>=4.8.0       # Image processing for tumor detection
```

## 💊 Medication Database

### Glioma Medications (4 drugs)
1. **Temozolomide (Temodar)** - Chemotherapy agent
2. **Bevacizumab (Avastin)** - Targeted therapy
3. **Levetiracetam (Keppra)** - Anti-seizure medication
4. **Dexamethasone** - Reduces brain swelling

### Meningioma Medications (4 drugs)
1. **Hydroxyurea** - Slows tumor growth
2. **Phenytoin (Dilantin)** - Anti-seizure medication
3. **Dexamethasone** - Reduces swelling
4. **Mifepristone (Korlym)** - Progesterone receptor blocker

### Pituitary Tumor Medications (5 drugs)
1. **Cabergoline (Dostinex)** - Dopamine agonist
2. **Bromocriptine (Parlodel)** - Reduces prolactin
3. **Octreotide (Sandostatin)** - Controls growth hormone
4. **Pasireotide (Signifor)** - For Cushing's disease
5. **Levothyroxine (Synthroid)** - Thyroid hormone replacement

## 🎓 Training Details

### Dataset
- **Source:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle
- **Training Set:** ~5,700 images
- **Testing Set:** ~1,300 images
- **Classes:** 4 (balanced distribution)

### Data Augmentation
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness & contrast ±20%)
- Random affine transformation

### Training Configuration
- **Optimizer:** AdamW (lr=1e-4, weight_decay=0.01)
- **Loss Function:** Cross-Entropy Loss
- **Batch Size:** 16
- **Epochs:** 25 (with early stopping)
- **Scheduler:** ReduceLROnPlateau
- **Train/Val Split:** 85/15

### Preprocessing
- Resize to 224×224
- ImageNet normalization:
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]

## 🖼️ Screenshots

### Main Interface
- Modern gradient design with purple/blue theme
- Two-column responsive layout
- Professional medical-grade appearance

### Analysis Results
- Large diagnosis display with confidence score
- Tumor size card with visual indicators
- Interactive probability distribution bars
- Expandable disease information sections

### PDF Report
- Professional medical report format
- Includes MRI scan image
- Complete patient and diagnosis information
- Detailed medication recommendations
- Color-coded sections for easy reading

## 🔬 Research & Citation

This implementation is based on the Swin Transformer architecture:

```bibtex
@inproceedings{liu2021swin,
  title={Swin transformer: Hierarchical vision transformer using shifted windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10012--10022},
  year={2021}
}
```

## ⚠️ Important Disclaimers

### Medical Use
- This application is for **educational and research purposes only**
- **NOT** a substitute for professional medical diagnosis
- Always consult qualified healthcare professionals for medical advice
- Clinical decisions should never be based solely on this tool

### Medication Information
- Medication recommendations are for **reference only**
- Must be prescribed by a licensed physician
- Dosages should be individualized based on patient factors
- Never self-medicate based on this information

### Tumor Size Estimation
- Size estimation is approximate and based on image processing
- Clinical imaging tools provide more accurate measurements
- Should be verified by medical imaging specialists

## 🔬 Research & Citation

This implementation is based on the Swin Transformer architecture:

```bibtex
@inproceedings{liu2021swin,
  title={Swin transformer: Hierarchical vision transformer using shifted windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10012--10022},
  year={2021}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🐛 Troubleshooting

### Common Issues

**1. Model file not found:**
- Ensure `swin_brain_tumor_complete.pth` is in the project root directory
- Check file permissions

**2. PDF download not working:**
- Verify patient name and ID are filled in
- Ensure reportlab is installed: `pip install reportlab`

**3. Styling issues:**
- Clear browser cache and refresh
- Try a different browser (Chrome/Firefox recommended)

**4. Import errors:**
- Reinstall dependencies: `pip install -r requirements.txt --upgrade`
- Check Python version (3.8+ required)

**5. Tumor detection not showing:**
- Ensure opencv-python is installed: `pip install opencv-python`
- Check that the uploaded image is a valid MRI scan

## 📧 Contact & Support

- **GitHub Issues:** [Report bugs or request features](https://github.com/Krishnaa81/Brain-Tumor-Detection-using-Swin-Transformer/issues)
- **Repository:** [View source code](https://github.com/Krishnaa81/Brain-Tumor-Detection-using-Swin-Transformer)

## 🙏 Acknowledgments

- **Dataset:** [Masoud Nickparvar - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Model Architecture:** [Swin Transformer by Microsoft Research](https://github.com/microsoft/Swin-Transformer)
- **Framework:** [PyTorch](https://pytorch.org/) & [timm](https://github.com/huggingface/pytorch-image-models)
- **Web Framework:** [Streamlit](https://streamlit.io/)
- **PDF Generation:** [ReportLab](https://www.reportlab.com/)
- **Image Processing:** [OpenCV](https://opencv.org/)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🌟 Features Roadmap

- [ ] Multi-language support
- [ ] 3D MRI scan visualization
- [ ] Integration with DICOM format
- [ ] Historical patient data tracking
- [ ] Comparison with previous scans
- [ ] Email report delivery
- [ ] Mobile application version
- [ ] API for integration with hospital systems

## 📈 Version History

### Version 2.0 (Current)
- ✅ Professional UI with gradient design
- ✅ Patient information form
- ✅ Tumor size estimation
- ✅ Medication recommendations (13 drugs)
- ✅ PDF report generation with MRI image
- ✅ Disease information database
- ✅ Visual tumor region detection

### Version 1.0
- ✅ Basic tumor classification
- ✅ Simple Streamlit interface
- ✅ Confidence scores
- ✅ Probability distribution

---

**Made with ❤️ for Final Year Project | Advancing Medical AI Research**

**⭐ If you find this project helpful, please consider giving it a star on GitHub!**
