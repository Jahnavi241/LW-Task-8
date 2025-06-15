# Recognizing Characters on Number Plate of a Car

## License Plate Recognition using OpenCV and Tesseract

This project detects and recognizes license plates from car images using classical computer vision techniques (OpenCV) and Optical Character Recognition (Tesseract OCR).

---

## 📌 Features

- Detects the license plate region from an image using edge detection and contours.
- Crops the detected plate area.
- Extracts text from the license plate using Tesseract OCR.
- Displays the original image and cropped plate image.
- Prints the detected license plate number to the console.

---

## 🧰 Requirements

- Python 3.x
- OpenCV
- imutils
- NumPy
- pytesseract
- Tesseract OCR (installed locally)


## Install Dependencies

bash
pip install opencv-python imutils numpy pytesseract

---

## 🚀 How to Run
1.	Make sure Carplate.JPG is in the project directory.
2. Run the script

`python license_plate_detection.py`
   
---

## ⚠️ Limitations
- Works best with clear, high-resolution images taken from the front or rear.
- May not detect plates that are blurry, tilted, poorly lit, or obstructed.
- Limited to static images (no support for real-time detection or videos).
- Accuracy of OCR may vary depending on plate font, quality, and region.
