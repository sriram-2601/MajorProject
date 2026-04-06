import os
import numpy as np
from PIL import Image

def create_dummy_data(base_dir='data', num_classes=2, images_per_class=10):
    for split in ['train', 'val']:
        for i in range(num_classes):
            class_dir = os.path.join(base_dir, split, f'class_{i}')
            os.makedirs(class_dir, exist_ok=True)
            
            for j in range(images_per_class):
                # Create a random RGB image
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(os.path.join(class_dir, f'img_{j}.jpg'))
                
    print(f"Created dummy dataset in {base_dir} with {num_classes} classes.")

if __name__ == '__main__':
    create_dummy_data()
