import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import pytesseract
from typing import Tuple, List, Dict, Any
import os
import pickle
import json


class CNNChartClassifier:
    """Enhanced CNN-based chart type classifier"""

    def __init__(self):
        self.model = None
        self.classes = ['bar', 'line', 'pie', 'scatter', 'histogram', 'box', 'area']
        self.confidence_threshold = 0.7
        self._build_model()

    def _build_model(self):
        """Build CNN model with transfer learning"""
        try:
            # Try to load pre-trained model
            if os.path.exists('models/chart_classifier.h5'):
                self.model = keras.models.load_model('models/chart_classifier.h5')
            else:
                # Build new model with ResNet50 backbone
                base_model = ResNet50(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(224, 224, 3)
                )

                # Freeze base model layers
                base_model.trainable = False

                # Add custom classification head
                x = base_model.output
                x = GlobalAveragePooling2D()(x)
                x = Dense(512, activation='relu')(x)
                x = Dropout(0.3)(x)
                x = Dense(256, activation='relu')(x)
                x = Dropout(0.3)(x)
                predictions = Dense(len(self.classes), activation='softmax')(x)

                self.model = Model(inputs=base_model.input, outputs=predictions)
                self.model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )

                # For demo purposes, we'll use rule-based classification
                # In production, you would train this model on chart datasets

        except Exception as e:
            print(f"Model building failed: {e}, using rule-based classification")
            self.model = None

    def classify_chart(self, image: np.ndarray) -> Tuple[str, float]:
        """Classify chart type from image"""
        try:
            if self.model is not None:
                # Preprocess image
                processed_img = self._preprocess_image(image)

                # Get predictions
                predictions = self.model.predict(processed_img)
                class_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][class_idx])

                if confidence >= self.confidence_threshold:
                    return self.classes[class_idx], confidence

            # Fallback to rule-based classification
            return self._rule_based_classification(image)

        except Exception as e:
            print(f"Chart classification failed: {e}")
            return self._rule_based_classification(image)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for CNN model"""
        # Resize to model input size
        resized = cv2.resize(image, (224, 224))

        # Normalize
        normalized = resized.astype('float32') / 255.0

        # Add batch dimension
        return np.expand_dims(normalized, axis=0)

    def _rule_based_classification(self, image: np.ndarray) -> Tuple[str, float]:
        """Rule-based chart classification using computer vision"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Analyze contours for chart type detection
        if len(contours) == 0:
            return "unknown", 0.3

        # Check for circular shapes (pie charts)
        circular_contours = 0
        rectangular_contours = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Filter small contours
                continue

            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.7:
                    circular_contours += 1

            # Check rectangularity
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4:
                rectangular_contours += 1

        # Classify based on detected shapes
        if circular_contours > 0:
            return "pie", 0.8
        elif rectangular_contours >= 3:
            return "bar", 0.75
        else:
            # Check for lines (line charts)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

            if lines is not None and len(lines) > 5:
                return "line", 0.7
            else:
                return "scatter", 0.6


class EnhancedImageProcessor:
    """Enhanced image processing for data extraction"""

    def __init__(self):
        self.ocr_config = '--oem 3 --psm 6'

    def extract_chart_regions(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract individual chart regions from image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply morphological operations to detect chart regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        chart_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10000:  # Filter small regions
                x, y, w, h = cv2.boundingRect(contour)
                chart_region = image[y:y + h, x:x + w]
                chart_regions.append(chart_region)

        return chart_regions if chart_regions else [image]

    def extract_chart_data(self, image: np.ndarray, chart_type: str) -> Dict[str, Any]:
        """Extract data from chart based on type"""
        try:
            # Extract text using OCR
            text_data = self._extract_text_ocr(image)

            # Extract visual data based on chart type
            if chart_type == "bar":
                visual_data = self._extract_bar_data(image)
            elif chart_type == "line":
                visual_data = self._extract_line_data(image)
            elif chart_type == "pie":
                visual_data = self._extract_pie_data(image)
            elif chart_type == "scatter":
                visual_data = self._extract_scatter_data(image)
            else:
                visual_data = self._extract_generic_data(image)

            return {
                "chart_type": chart_type,
                "text_data": text_data,
                "visual_data": visual_data,
                "extracted_values": self._parse_numeric_values(text_data)
            }

        except Exception as e:
            print(f"Data extraction failed: {e}")
            return {
                "chart_type": chart_type,
                "text_data": [],
                "visual_data": {},
                "extracted_values": [],
                "error": str(e)
            }

    def _extract_text_ocr(self, image: np.ndarray) -> List[str]:
        """Extract text using OCR"""
        # Preprocess image for better OCR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Extract text
        text = pytesseract.image_to_string(enhanced, config=self.ocr_config)

        # Clean and split text
        lines = [line.strip() for line in text.split('\
') if line.strip()]
        return lines

    def _extract_bar_data(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract data from bar charts"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find rectangular contours (bars)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bars = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small areas
                x, y, w, h = cv2.boundingRect(contour)
                bars.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "area": int(area)
                })

        return {
            "bars": bars,
            "estimated_values": [bar["height"] for bar in bars]
        }

    def _extract_line_data(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract data from line charts"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

        line_data = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                line_data.append({
                    "start": (int(x1), int(y1)),
                    "end": (int(x2), int(y2)),
                    "length": int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
                })

        return {
            "lines": line_data,
            "total_lines": len(line_data)
        }

    def _extract_pie_data(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract data from pie charts"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=0, maxRadius=0
        )

        pie_data = {}
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                pie_data = {
                    "center": (int(x), int(y)),
                    "radius": int(r),
                    "estimated_segments": self._estimate_pie_segments(image, x, y, r)
                }
                break  # Use first detected circle

        return pie_data

    def _extract_scatter_data(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract data from scatter plots"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use HoughCircles to detect points
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 10,
            param1=50, param2=15, minRadius=2, maxRadius=10
        )

        points = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                points.append({"x": int(x), "y": int(y), "radius": int(r)})

        return {
            "points": points,
            "total_points": len(points)
        }

    def _extract_generic_data(self, image: np.ndarray) -> Dict[str, Any]:
        """Generic data extraction for unknown chart types"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Basic feature extraction
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return {
            "contours_count": len(contours),
            "image_dimensions": image.shape[:2],
            "average_intensity": float(np.mean(gray))
        }

    def _estimate_pie_segments(self, image: np.ndarray, cx: int, cy: int, radius: int) -> int:
        """Estimate number of pie segments"""
        # Simple approach: count color changes around the circle
        angles = np.linspace(0, 2 * np.pi, 360)
        colors = []

        for angle in angles:
            x = int(cx + radius * 0.7 * np.cos(angle))
            y = int(cy + radius * 0.7 * np.sin(angle))

            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                colors.append(tuple(image[y, x]))

        # Count color changes (simplified)
        changes = 0
        for i in range(1, len(colors)):
            if colors[i] != colors[i - 1]:
                changes += 1

        return max(2, changes // 2)  # Estimate segments from color changes

    def _parse_numeric_values(self, text_lines: List[str]) -> List[float]:
        """Parse numeric values from OCR text"""
        import re

        numeric_values = []
        for line in text_lines:
            # Find numbers in the text
            numbers = re.findall(r'-?\d+\.?\d*', line)
            for num in numbers:
                try:
                    value = float(num)
                    numeric_values.append(value)
                except ValueError:
                    continue

        return numeric_values

def analyze_pdf_charts(image: np.ndarray) -> Dict[str, Any]:
    classifier = CNNChartClassifier()
    processor = EnhancedImageProcessor()
    chart_type, confidence = classifier.classify_chart(image)
    chart_data = processor.extract_chart_data(image, chart_type)
    chart_data["confidence"] = confidence
    return chart_data





