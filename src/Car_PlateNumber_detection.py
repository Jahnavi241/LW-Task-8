import cv2
import imutils
import numpy as np
import pytesseract
import argparse
import os
import sys

# Update this to your local tesseract path if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def load_image(image_path):
    """Load an image from the given path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return cv2.resize(image, (600, 400))


def preprocess_image(image):
    """Convert to grayscale and apply bilateral filter and edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)
    edged = cv2.Canny(gray, 30, 200)
    return gray, edged


def find_plate_contour(edged):
    """Find the rectangular contour that likely corresponds to the license plate."""
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
        if len(approx) == 4:
            return approx
    return None


def extract_plate_region(image, gray, plate_contour):
    """Extract the license plate region using the contour mask."""
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [plate_contour], 0, 255, -1)
    masked = cv2.bitwise_and(image, image, mask=mask)

    (x, y) = np.where(mask == 255)
    topx, topy = np.min(x), np.min(y)
    bottomx, bottomy = np.max(x), np.max(y)
    cropped = gray[topx:bottomx + 1, topy:bottomy + 1]
    return cropped


def recognize_text(image):
    """Run OCR on the cropped license plate image."""
    config = "--psm 11"
    text = pytesseract.image_to_string(image, config=config)
    return text.strip()


def display_images(original, cropped):
    """Display the original and cropped images."""
    original = cv2.resize(original, (500, 300))
    cropped = cv2.resize(cropped, (400, 200))
    cv2.imshow("Original Image", original)
    cv2.imshow("License Plate", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(image_path, show_images=True):
    """Main function to execute the plate recognition pipeline."""
    try:
        image = load_image(image_path)
        gray, edged = preprocess_image(image)
        plate_contour = find_plate_contour(edged)

        if plate_contour is None:
            print("License plate contour not detected.")
            sys.exit(1)

        cv2.drawContours(image, [plate_contour], -1, (0, 0, 255), 3)
        cropped_plate = extract_plate_region(image, gray, plate_contour)
        plate_text = recognize_text(cropped_plate)

        print("\nLicense Plate Recognition")
        print(f"Detected license plate number: {plate_text or 'Not readable'}")

        if show_images:
            display_images(image, cropped_plate)

    except Exception as e:
        print(f" Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="License Plate Recognition using OpenCV and Tesseract")
    parser.add_argument("--image", "-i", required=True, help="Path to the car image (e.g., Carplate.JPG)")
    parser.add_argument("--no-gui", action="store_true", help="Disable image display")

    args = parser.parse_args()
    main(args.image, show_images=not args.no_gui)