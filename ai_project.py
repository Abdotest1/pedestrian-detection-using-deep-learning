import os
import shutil
import cv2
import xml.etree.ElementTree as ET
import random
import matplotlib.pyplot as plt
from ultralytics import YOLO


IMG_DIR = r"C:/Users/abdoo/OneDrive/Desktop/test/LLVIP/visible/train"
ANNOT_DIR = r"C:/Users/abdoo/OneDrive/Desktop/test/LLVIP/Annotations"
OUTPUT_DIR = r"C:/Users/abdoo/OneDrive/Desktop/test/LLVIP_YOLO"
train_ratio = 0.8
num_images_to_use = 12000

# Convert bounding box to YOLO format 
def convert_bbox(size, box):    
    dw = 1. / size[0]
    dh = 1. / size[1]
    xmin, ymin, xmax, ymax = box
    x_center = (xmin + xmax) / 2.0 * dw
    y_center = (ymin + ymax) / 2.0 * dh
    width = (xmax - xmin) * dw
    height = (ymax - ymin) * dh
    return (x_center, y_center, width, height)


def show_image_with_boxes(image_path, yolo_labels):
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    for line in yolo_labels:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, x_center, y_center, width, height = map(float, parts)
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()


def draw_boxes_and_save(image_path, results, output_path, model):
    image = cv2.imread(image_path)
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box.astype(int)
            class_names = model.names  
            label = f"{class_names[int(cls)]} {score:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)


def run_yolo_inference(folder_path):
    model = YOLO('yolov8l.pt')
    output_dir = os.path.join("output")
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]

    for img_name in image_files:
        img_path = os.path.join(folder_path, img_name)
        results = model(img_path)
        output_path = os.path.join(output_dir, f"det_{img_name}")
        draw_boxes_and_save(img_path, results, output_path, model)  
        print(f"Saved result: {output_path}")


def convert_and_split():
    for subdir in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)

    image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith('.jpg')]
    print(f"Total images found in source folder: {len(image_files)}")
    random.shuffle(image_files)
    image_files = image_files[:num_images_to_use]
    print(f"Images to process (limited to {num_images_to_use}): {len(image_files)}")

    split_idx = int(train_ratio * len(image_files))
    showed_sample = False
    copied_count, skipped_no_xml, skipped_no_pedestrian = 0, 0, 0

    for idx, img_name in enumerate(image_files):
        base = os.path.splitext(img_name)[0]
        xml_path = os.path.join(ANNOT_DIR, base + '.xml')
        img_path = os.path.join(IMG_DIR, img_name)

        if not os.path.exists(xml_path):
            skipped_no_xml += 1
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        w, h = int(size.find('width').text), int(size.find('height').text)

        yolo_lines = []
        for obj in root.findall('object'):
            label = obj.find('name').text
            if label.lower() != 'person':
                continue
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            bb = convert_bbox((w, h), (xmin, ymin, xmax, ymax))
            yolo_lines.append(f"0 {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")

        if not yolo_lines:
            skipped_no_pedestrian += 1
            continue

        split = 'train' if idx < split_idx else 'val'
        output_img_path = os.path.join(OUTPUT_DIR, 'images', split, img_name)
        output_label_path = os.path.join(OUTPUT_DIR, 'labels', split, base + '.txt')

        shutil.copy(img_path, output_img_path)
        with open(output_label_path, 'w') as f:
            f.writelines(yolo_lines)
        copied_count += 1

        print(f"Copied {img_name} with {len(yolo_lines)} pedestrian annotations to {output_img_path}")

        if not showed_sample:
            print(f"Showing sample: {img_name}")
            original = cv2.imread(img_path)
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            plt.title("Original Image (Before)")
            plt.imshow(original_rgb)
            plt.axis('off')
            plt.show()

            print("After YOLO Conversion:")
            show_image_with_boxes(img_path, yolo_lines)
            showed_sample = True

    print(f"Total images copied: {copied_count}")
    print(f"Images skipped due to missing XML: {skipped_no_xml}")
    print(f"Images skipped due to no pedestrian annotations: {skipped_no_pedestrian}")


# === Train and evaluate YOLOv8 ===
"""def train_and_evaluate():
    print("\n=== Starting YOLOv8 Training ===")
    model = YOLO("yolov8l.pt")  # You can change to a smaller version like 'yolov8n.pt' for speed
    model.model.to('cuda')  # Explicitly move to GPU

    # Train the model
    model.train(
    data='C:/Users/abdoo/OneDrive/Desktop/test/LLVIP_YOLO/../llvip.yaml',
    epochs=1,
    imgsz=640,
    batch=8,
    device='cuda'
    )

    print("\n=== Evaluating Model on Validation Set ===")
    # Evaluate after training
    metrics = model.val()

    print("\n=== Evaluation Results ===")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.precision:.4f}")
    print(f"Recall: {metrics.box.recall:.4f}")
 """
# === MAIN ===
if __name__ == "__main__":
    #train_and_evaluate()
   
    convert_and_split()

    sample_img_folder = os.path.join(OUTPUT_DIR, 'images', 'train')
    sample_images = [f for f in os.listdir(sample_img_folder) if f.lower().endswith('.jpg')]
    if sample_images:
        print(f"Running YOLO inference on {len(sample_images)} training images...")
        run_yolo_inference(sample_img_folder)
    else:
        print("No images found in training folder for inference.")
    
    






