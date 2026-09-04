import argparse
import cv2
import numpy as np
import ezdxf
import logging
from pathlib import Path

# Optional imports for OCR and advanced models
try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    import easyocr
except ImportError:
    easyocr = None

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class ElectricalCADAgent:
    def __init__(self, use_easyocr=False, yolo_model_path=None):
        """
        Initializes the ML Agent for converting electrical drawings to CAD.
        """
        self.use_easyocr = use_easyocr
        self.yolo_model_path = yolo_model_path

        # Initialize OCR
        self.reader = None
        if self.use_easyocr and easyocr is not None:
            logging.info("Initializing EasyOCR...")
            self.reader = easyocr.Reader(['en'])
        elif not self.use_easyocr and pytesseract is not None:
            logging.info("Using PyTesseract for OCR.")
        else:
            logging.warning("No OCR library available. Text extraction will be skipped.")

        # Initialize YOLO model (Placeholder)
        if self.yolo_model_path:
            logging.info(f"Loading YOLO model from {self.yolo_model_path}...")
            # self.symbol_model = YOLO(self.yolo_model_path)
            self.symbol_model = None
        else:
            self.symbol_model = None
            logging.info("No symbol detection model provided. Skipping symbol recognition.")

    def preprocess_image(self, image_path):
        """
        Loads and preprocesses the scanned image (grayscale, binarization, noise removal).
        """
        logging.info(f"Preprocessing image: {image_path}")
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot open or find image: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to get a binary image (black background, white lines)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Optional: morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        clean_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        return img, gray, clean_binary

    def extract_text(self, image):
        """
        Extracts text from the image using OCR.
        Returns a list of dicts with 'text', 'bbox' (x, y, w, h).
        """
        texts = []
        if self.use_easyocr and self.reader:
            logging.info("Extracting text with EasyOCR...")
            results = self.reader.readtext(image)
            for (bbox, text, prob) in results:
                if prob > 0.5:
                    # bbox is a list of 4 points: [top-left, top-right, bottom-right, bottom-left]
                    x = int(min([pt[0] for pt in bbox]))
                    y = int(min([pt[1] for pt in bbox]))
                    w = int(max([pt[0] for pt in bbox])) - x
                    h = int(max([pt[1] for pt in bbox])) - y
                    texts.append({'text': text, 'bbox': (x, y, w, h)})
        elif not self.use_easyocr and pytesseract:
            logging.info("Extracting text with PyTesseract...")
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            for i in range(len(data['text'])):
                if float(data['conf'][i]) > 50 and data['text'][i].strip() != '':
                    texts.append({
                        'text': data['text'][i],
                        'bbox': (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                    })
        return texts

    def detect_lines(self, binary_image):
        """
        Detects straight lines using Probabilistic Hough Transform.
        Returns a list of lines: [(x1, y1, x2, y2), ...].
        """
        logging.info("Detecting lines...")
        lines_list = []
        lines = cv2.HoughLinesP(binary_image, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                lines_list.append((x1, y1, x2, y2))
        return lines_list

    def detect_symbols(self, image):
        """
        Detects electrical symbols using a trained object detection model (e.g., YOLO).
        Returns a list of dicts: {'label': 'resistor', 'bbox': (x, y, w, h)}.
        """
        symbols = []
        if self.symbol_model:
            logging.info("Detecting electrical symbols...")
            # Placeholder for YOLO inference
            # results = self.symbol_model(image)
            # parse results into symbols list
            pass
        return symbols

    def export_to_dxf(self, lines, texts, symbols, output_path, img_height):
        """
        Exports the extracted features to a DXF file.
        In DXF, the origin (0,0) is typically bottom-left, while images are top-left.
        We flip the Y coordinate.
        """
        logging.info(f"Exporting to DXF: {output_path}")
        doc = ezdxf.new(dxfversion='R2010')
        msp = doc.modelspace()

        # Add lines
        for (x1, y1, x2, y2) in lines:
            # Flip Y
            msp.add_line((x1, img_height - y1), (x2, img_height - y2))

        # Add texts
        for item in texts:
            text = item['text']
            x, y, w, h = item['bbox']
            # Flip Y. Approximate bottom-left of text
            msp.add_text(text, dxfattribs={'height': 12}).set_placement((x, img_height - y - h))

        # Add symbols (as bounding box rectangles and a label for now)
        for sym in symbols:
            label = sym['label']
            x, y, w, h = sym['bbox']
            # Draw rectangle
            y_flipped = img_height - y
            msp.add_lwpolyline([
                (x, y_flipped),
                (x + w, y_flipped),
                (x + w, y_flipped - h),
                (x, y_flipped - h),
                (x, y_flipped)
            ])
            msp.add_text(label, dxfattribs={'height': 10}).set_placement((x, y_flipped - h))

        doc.saveas(str(output_path))
        logging.info("Export complete.")

    def process_drawing(self, input_image_path, output_dxf_path):
        """
        Main pipeline to convert an image to DXF.
        """
        img, gray, binary = self.preprocess_image(input_image_path)
        img_height, img_width = img.shape[:2]

        texts = self.extract_text(gray)
        lines = self.detect_lines(binary)
        symbols = self.detect_symbols(img)

        # Mask out text and symbol areas before line detection in a full implementation
        # to avoid detecting text/symbols as lines.

        self.export_to_dxf(lines, texts, symbols, output_dxf_path, img_height)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert scanned electrical drawing to DXF CAD format.")
    parser.add_argument("input_image", type=str, help="Path to the input scanned drawing image.")
    parser.add_argument("output_dxf", type=str, help="Path to save the output DXF file.")
    parser.add_argument("--use_easyocr", action="store_true", help="Use EasyOCR instead of PyTesseract.")
    parser.add_argument("--yolo_model", type=str, default=None, help="Path to YOLO model for symbol detection.")

    # parse_known_args used to prevent crashes from injected kernel arguments in Jupyter/Kaggle environments
    args, _ = parser.parse_known_args()

    agent = ElectricalCADAgent(use_easyocr=args.use_easyocr, yolo_model_path=args.yolo_model)

    try:
        agent.process_drawing(args.input_image, args.output_dxf)
        print(f"Successfully converted {args.input_image} to {args.output_dxf}")
    except Exception as e:
        logging.error(f"Error processing drawing: {e}")
