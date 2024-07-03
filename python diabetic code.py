mkdir -p ~/.kaggle
mv /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle competitions download -c diabetic-retinopathy-detection
unzip diabetic-retinopathy-detection.zip -d ./diabetic-retinopathy-detection
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Path to the dataset
data_dir = './diabetic-retinopathy-detection'

# Load labels
labels = pd.read_csv(os.path.join(data_dir, 'trainLabels.csv'))

# Display the first few rows of the labels
print(labels.head())

# Load and display a sample image
sample_image_path = os.path.join(data_dir, 'train', labels['image'][0] + '.jpeg')
sample_image = cv2.imread(sample_image_path)
sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

plt.imshow(sample_image)
plt.title('Sample Image')
plt.show()
