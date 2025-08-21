import cv2
import cvlib as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import logging

# Configure logging for transparency and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_image(image_path):
    """
    Load an image from the specified path.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        numpy.ndarray: Loaded image.
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        logging.info(f"Loaded image: {image_path}")
        return img
    except Exception as e:
        logging.error(f"Error loading image: {str(e)}")
        raise

def detect_objects(img, object_type='car'):
    """
    Detect objects in the image using cvlib's YOLO model.
    
    Args:
        img (numpy.ndarray): Input image.
        object_type (str): Type of object to count (e.g., 'car').
    
    Returns:
        tuple: Bounding boxes, labels, confidences, and count of specified object.
    """
    try:
        # Detect objects using cvlib
        bbox, labels, conf = cv.detect_common_objects(img)
        object_count = sum(1 for label in labels if label.lower() == object_type.lower())
        logging.info(f"Detected {len(labels)} objects ({object_count} {object_type}s)")
        return bbox, labels, conf, object_count
    except Exception as e:
        logging.error(f"Error detecting objects: {str(e)}")
        raise

def visualize_objects(img, bbox, labels, conf, output_path):
    """
    Draw bounding boxes around detected objects and save the image.
    
    Args:
        img (numpy.ndarray): Input image.
        bbox (list): Bounding box coordinates.
        labels (list): Object labels.
        conf (list): Confidence scores.
        output_path (str): Path to save the output image.
    """
    try:
        output_img = img.copy()
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = box
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{labels[i]}: {conf[i]:.2f}"
            cv2.putText(output_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save and display the image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, output_img)
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Detected Objects')
        plt.show()
        logging.info(f"Saved visualized image: {output_path}")
    except Exception as e:
        logging.error(f"Error visualizing objects: {str(e)}")
        raise

def main(args):
    """
    Main function to handle CLI operations for object counting.
    
    Args:
        args: Parsed command-line arguments.
    """
    try:
        # Load image
        img = load_image(args.image_path)

        # Detect objects
        bbox, labels, conf, object_count = detect_objects(img, args.object)

        if args.mode == 'count':
            # Output count
            print(f"🌟 Processing image: {args.image_path}")
            print(f"🔍 Detected {len(labels)} objects ({object_count} {args.object}s)")
            print(f"✅ Number of {args.object}s: {object_count}")

        elif args.mode == 'visualize':
            # Visualize and save
            output_path = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.image_path))[0]}_detected.jpg")
            visualize_objects(img, bbox, labels, conf, output_path)
            print(f"🌟 Visualized image saved at: {output_path}")
            print(f"✅ Number of {args.object}s: {object_count}")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Count Objects in Image: Computer vision with Python")
    parser.add_argument('--mode', choices=['count', 'visualize'], default='count',
                        help="Mode: count or visualize")
    parser.add_argument('--image_path', default='traffic.jpg', help="Path to the input image")
    parser.add_argument('--object', default='car', help="Object type to count (e.g., car, person)")
    parser.add_argument('--output_dir', default='./outputs', help="Directory to save visualizations")
    args = parser.parse_args()

    main(args)