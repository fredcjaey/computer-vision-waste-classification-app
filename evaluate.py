import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from model import create_resnet_model
from utils.data_loader import get_data_loaders

def evaluate_model():
    # Load data
    _, test_loader, class_names = get_data_loaders()
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_resnet_model(num_classes=len(class_names))
    
    checkpoint = torch.load('models/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Get predictions
    all_preds = []
    all_labels = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Classification report
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
    plt.show()
    
    # Per-class accuracy
    class_correct = cm.diagonal()
    class_total = cm.sum(axis=1)
    class_acc = 100 * class_correct / class_total
    
    print("\n" + "="*50)
    print("PER-CLASS ACCURACY")
    print("="*50)
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {class_acc[i]:.2f}%")

if __name__ == '__main__':
    evaluate_model()