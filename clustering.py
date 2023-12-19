import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

def load_images(folder_path):
    images = []
    valid_extensions = ['.png', '.jpg', '.jpeg']  # Geçerli dosya uzantıları

    for filename in os.listdir(folder_path):
        _, extension = os.path.splitext(filename)
        if extension.lower() in valid_extensions:
            img_path = os.path.join(folder_path, filename)
            if os.path.isfile(img_path):
                images.append(cv2.imread(img_path))
    
    return images

def extract_color_features(image):
    # Resmin renk histogramını çıkar
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = hist.flatten()

    return hist

def cluster_images(images, num_clusters):
    data = np.array([extract_color_features(image) for image in images])

    # KMeans kümeleme
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data)

    # Küme etiketlerini al
    labels = kmeans.labels_

    clustered_images = {}
    for i, label in enumerate(labels):
        if label not in clustered_images:
            clustered_images[label] = []
        clustered_images[label].append(images[i])

    return clustered_images

def save_clustered_images(clustered_images, output_folder):
    for label, images in clustered_images.items():
        cluster_folder = os.path.join(output_folder, str(label))
        os.makedirs(cluster_folder, exist_ok=True)
        for i, image in enumerate(images):
            cv2.imwrite(os.path.join(cluster_folder, f'image_{i}.png'), image)

if __name__ == "__main__":
    input_folder = 'imagesoutput'
    output_folder = 'clustered_images'
    num_clusters = 5 # Dilerseniz bu sayıyı değiştirebilirsiniz

    # Resimleri yükle
    images = load_images(input_folder)

    # Resimleri kümele
    clustered_images = cluster_images(images, num_clusters)

    # Kümelere ayrılan resimleri kaydet
    save_clustered_images(clustered_images, output_folder)
