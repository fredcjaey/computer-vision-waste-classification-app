import os
import shutil
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Define paths
project_root = Path("c:\\Users\\CJAEY\\waste_classification_project")
classes = ["cardboard", "glass", "metal", "paper", "plastic"]
train_dir = project_root / "train"
test_dir = project_root / "test"
split_ratio = 0.8  # 80% train, 20% test

# Create subdirectories in train and test folders
for class_name in classes:
    (train_dir / class_name).mkdir(parents=True, exist_ok=True)
    (test_dir / class_name).mkdir(parents=True, exist_ok=True)

# Split and move files
for class_name in classes:
    class_dir = project_root / class_name
    files = list(class_dir.glob("*.jpg"))
    
    # Shuffle and split
    random.shuffle(files)
    split_idx = int(len(files) * split_ratio)
    train_files = files[:split_idx]
    test_files = files[split_idx:]
    
    # Move files to train folder
    for file in train_files:
        shutil.move(str(file), str(train_dir / class_name / file.name))
    
    # Move files to test folder
    for file in test_files:
        shutil.move(str(file), str(test_dir / class_name / file.name))
    
    print(f"{class_name}: {len(train_files)} train, {len(test_files)} test")

print("\nDataset split complete!")
