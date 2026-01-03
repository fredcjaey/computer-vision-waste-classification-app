# ♻️ Waste Classification System

Deep learning model that classifies waste into 6 categories: paper, cardboard, plastic, metal, glass, and trash.

## 🚀 Features
- ResNet50 transfer learning model
- Real-time image classification
- Recycling instructions
- Interactive Streamlit web app
- 90%+ accuracy

## 📦 Installation
```bash
# Clone repository
git clone <your-repo-url>
cd waste_classification_project

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Usage

### Training
```bash
python train.py
```

### Evaluation
```bash
python evaluate.py
```

### Run Web App
```bash
streamlit run app.py
```

## 📊 Dataset
Uses Garbage Classification dataset with 6 categories.

## 🏗️ Model Architecture
- Base: ResNet50 (pretrained on ImageNet)
- Custom classification head
- Input size: 224x224x3

## 📈 Results
- Training Accuracy: ~95%
- Validation Accuracy: ~90%

## 🤝 Contributing
Pull requests welcome!

## 📝 License
MIT License"# computer-vision-waste-classification-app" 
