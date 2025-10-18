# 🧠 Brain Tumor Detection using Swin Transformer

A deep learning application for automated brain tumor classification from MRI images using Swin Transformer architecture. This project achieves **99.08% accuracy** on the test dataset.

## 📋 Overview

This application uses a state-of-the-art Swin Transformer model to classify brain MRI scans into four categories:
- **Glioma** - A type of tumor that occurs in the brain and spinal cord
- **Meningioma** - A tumor that arises from the meninges
- **Pituitary** - A tumor in the pituitary gland
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
- pip

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd brain-tumor-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the trained model:
   - The model file is too large for GitHub (331 MB)
   - Download `swin_brain_tumor_complete.pth` from [Google Drive/OneDrive/Dropbox - Add your link]
   - Place the downloaded file in the project root directory
   - Or train your own model using the training code

## 🚀 Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Application

1. Upload an MRI brain scan image (PNG, JPG, or JPEG format)
2. Click the "Predict" button
3. View the prediction results:
   - Primary prediction with confidence score
   - Probability distribution across all classes

## 📁 Project Structure

```
brain-tumor-detection/
├── app.py                          # Streamlit web application
├── swin_brain_tumor_complete.pth   # Trained model weights (download separately)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore rules
├── verify_model.py                 # Model verification script
├── fix_checkpoint.py               # Checkpoint repair utility
├── TRAINING_CODE_FIX.md           # Training code bug documentation
└── SOLUTION_SUMMARY.md            # Issue resolution summary
```

## 📥 Model Download

The trained model file (`swin_brain_tumor_complete.pth`) is **331 MB** and cannot be hosted on GitHub.

**Options to get the model:**
1. **Download pre-trained model:** [Add your Google Drive/OneDrive link here]
2. **Train your own:** Use the training code from the research paper implementation
3. **Use Git LFS:** If you have Git LFS installed, you can track large files

After downloading, place the `.pth` file in the project root directory.

## 🔧 Requirements

```txt
streamlit>=1.28.0
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
Pillow>=10.0.0
numpy>=1.24.0
```

## 🧪 Model Verification

To verify the model is working correctly:

```bash
python verify_model.py
```

This script will:
- Load the checkpoint
- Display model information
- Run a test prediction
- Verify output format and probabilities

## 📊 Dataset

The model was trained on the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle, which contains:
- **Training Set:** ~5,700 images
- **Testing Set:** ~1,300 images
- **Classes:** 4 (balanced distribution)

## 🎓 Training Details

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

## 🐛 Known Issues & Fixes

### Class Name Ordering Bug (FIXED)
The original training code had a mismatch between sorted and unsorted class names. This has been corrected in the checkpoint file. See `TRAINING_CODE_FIX.md` for details.

If you encounter prediction issues:
```bash
python fix_checkpoint.py
```

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

## ⚠️ Disclaimer

This application is for **educational and research purposes only**. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical advice.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🙏 Acknowledgments

- Dataset: [Masoud Nickparvar - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- Model Architecture: [Swin Transformer by Microsoft Research](https://github.com/microsoft/Swin-Transformer)
- Framework: [PyTorch](https://pytorch.org/) & [timm](https://github.com/huggingface/pytorch-image-models)
- Web Framework: [Streamlit](https://streamlit.io/)

---

**Made with ❤️ for advancing medical AI research**
