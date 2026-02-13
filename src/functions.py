# Nick Markantonis

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from multiprocessing import get_context, cpu_count
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import cv2

logging.basicConfig(level=logging.INFO, format="[INFO] %(message)s")
LOGGER = logging.getLogger(__name__)
i: int = 0

# This file contains functions for feature extraction and saving features to CSV.
FEATURE_COLUMNS = [
  "mean_intensity",
  "std_intensity",
  "edge_mean",
  "no_edge",                     # no orientation
  "vertical_edge_strength",      # 0°–45°
  "diag45_edge_strength",        # 45°–90°
  "horizontal_edge_strength",    # 90°–135°
  "diag135_edge_strength",       # 135°–180°
  "num_contours",                # total contours found
  "max_circularity",             # max circularity of all contours
  "mean_circularity",            # average circularity
  "frac_circular_contours",       # fraction of contours with circularity > 0.7
  "num_lines",                   # total lines detected
  "avg_line_length",             # average length of detected lines
  "vertical_line_count",         # number of vertical lines
  "horizontal_line_count",       # number of horizontal lines
  "diagonal_line_count",          # number of diagonal lines
  "num_blobs"                    # number of detected blobs (circular keypoints)
]

# Saves extracted features and labels to a CSV file
def save_features_to_csv(
  features: np.ndarray,
  labels: np.ndarray,
  output_path: str,
  image_paths: list[str] | None = None,
) -> pd.DataFrame:
  LOGGER.info("Saving extracted features to %s...", output_path)
  df = pd.DataFrame(features, columns=FEATURE_COLUMNS)
  df.insert(0, "label", labels)

  if image_paths is not None:
    df.insert(1, "image_path", image_paths)

  df.to_csv(output_path, index=False)
  LOGGER.info("Feature export complete. Rows: %s", len(df))
  return df

# Compute orientation clusters using Sobel gradients and simple binning
def orientation_clustering(image: np.ndarray, threshold_ratio: float = 0.2) -> np.ndarray:
  # Compute gradients
  sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
  sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

  # Magnitude and angle
  A = np.sqrt(sobelx**2 + sobely**2)
  theta = np.arctan2(sobely, sobelx)
  theta = np.degrees(theta)
  theta[theta < 0] += 180  # Convert to [0, 180)

  # Threshold for "real edges"
  threshold = threshold_ratio * np.max(A)

  # Initialize bins with 0 (no edge)
  orientation_bins = np.zeros_like(theta, dtype=int)

  # Define bins only for strong edges
  strong_edges = A > threshold
  num_bins = 4
  bin_size = 180 / num_bins

  # Compute orientation bin (1–4)
  bins = (theta // bin_size).astype(int)
  bins[bins == num_bins] = num_bins - 1

  # Shift bins by +1 so:
  # 0 = no edge
  # 1–4 = orientations
  orientation_bins[strong_edges] = bins[strong_edges] + 1
  return orientation_bins

# Visualize orientation clusters with distinct colors
# this is just for testing to see how the orientation clustering looks on an image. It is not used in feature extraction.
def visualize_orientation_clusters(
  image: np.ndarray,
  output_path: str | None = None,
  colors: np.ndarray | None = None,
) -> np.ndarray:
  """
  Visualize orientation clusters with distinct colors.
  Returns a BGR image suitable for saving with OpenCV.
  """
  if colors is None:
    colors = np.array(
      [
        [0, 0, 0],      # black (no edge)
        [0, 0, 255],    # red
        [0, 255, 0],    # green
        [255, 0, 0],    # blue
        [0, 255, 255],  # yellow
      ],
      dtype=np.uint8,
    )

  orientation_bins = orientation_clustering(image)
  color_map = colors[orientation_bins]
  color_map = color_map.astype(np.uint8)

  if output_path is not None:
    cv2.imwrite(output_path, color_map)

  return color_map

# Extract circularity features using contours and Hough Transform
def circularity_features(image):
  img_uint8 = (image * 255).astype(np.uint8)
  
  # Edge detection
  edges = cv2.Canny(img_uint8, 50, 150)
  # Find contours
  contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
  circularities = []
  for cnt in contours:
      area = cv2.contourArea(cnt)
      perimeter = cv2.arcLength(cnt, True)
      if perimeter > 0:
          circularity = 4 * np.pi * area / (perimeter ** 2)
          circularities.append(circularity)
  # If no contours found
  if len(circularities) == 0:
      return np.array([0.0, 0.0, 0.0])
  
  # Features:
  # 1. max circularity
  # 2. mean circularity
  # 3. fraction of contours that are "mostly circular" (>0.7)
  max_circ = np.max(circularities)
  mean_circ = np.mean(circularities)
  frac_circular = np.sum(np.array(circularities) > 0.7) / len(circularities)
  
  return np.array([len(circularities), max_circ, mean_circ, frac_circular])

# detect lines using Hough Transform and compute features like number of lines, average length, and orientation counts
def detect_lines(image, min_length=5, max_gap=2):
  img_uint8 = (image * 255).astype(np.uint8)
  edges = cv2.Canny(img_uint8, 50, 150)
  
  lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi/180,
    threshold=20,
    minLineLength=min_length,
    maxLineGap=max_gap
  )

  line_features = {
    "num_lines": 0,
    "avg_length": 0,
    "vertical_count": 0,
    "horizontal_count": 0,
    "diag_count": 0
  }

  if lines is not None:
    line_features["num_lines"] = len(lines)
    lengths = []
    for line in lines:
      x1, y1, x2, y2 = line[0]
      dx = x2 - x1
      dy = y2 - y1
      length = np.sqrt(dx**2 + dy**2)
      lengths.append(length)

      angle = np.arctan2(dy, dx)
      angle = np.degrees(angle)
      if angle < 0:
        angle += 180

      # Simple orientation binning
      if 0 <= angle < 15 or 165 <= angle <= 180:
        line_features["horizontal_count"] += 1
      elif 75 <= angle <= 105:
        line_features["vertical_count"] += 1
      else:
        line_features["diag_count"] += 1

    line_features["avg_length"] = np.mean(lengths)

  return line_features

# Detect blobs using SimpleBlobDetector and return the count of detected blobs
def detect_blobs(image):
  img_uint8 = (image * 255).astype(np.uint8)
  params = cv2.SimpleBlobDetector_Params()
  params.filterByCircularity = True
  params.minCircularity = 0.7
  detector = cv2.SimpleBlobDetector_create(params)
  keypoints = detector.detect(img_uint8)
  return len(keypoints)

# Extract features from a single image and return as a feature vector
def extract_features(image: np.ndarray) -> np.ndarray:
  features = []
  image = image.astype(np.float32, copy=False)
  global i
  
  # ===============================================================================================
  # Basic intensity features
  features.append(np.mean(image))  # mean intensity
  features.append(np.std(image))   # std intensity

  # ===============================================================================================
  # Edge features
  edges = cv2.Canny((image * 255).astype(np.uint8), 100, 200)
  features.append(np.mean(edges))  # mean edge strength

  # ===============================================================================================
  # Οrientation clustering features
  orientation_bins = orientation_clustering(image)
  features.extend(np.bincount(orientation_bins.flatten(), minlength=5))

  # ===============================================================================================
  # Circle features (Hough Transform)
  features.extend(circularity_features(image))

  # ===============================================================================================
  # Line features (Hough Transform)
  features.extend(detect_lines(image).values())

  # ===============================================================================================
  # Blob features (SimpleBlobDetector)
  features.append(detect_blobs(image))

  return np.array(features)

# Extract features from a batch of images in parallel using multiprocessing
def parallel_extract_features( images: np.ndarray, n_jobs: int | None = None, chunk_size: int = 64, report_every: int = 5,) -> np.ndarray:
  total = len(images)
  if total == 0:
    return np.empty((0, len(FEATURE_COLUMNS)), dtype=np.float32)

  if n_jobs is None:
    n_jobs = max(cpu_count() - 1, 1)

  LOGGER.info("Extracting features in parallel with %s workers...", n_jobs)
  ctx = get_context("spawn")
  features = []
  next_report = 0

  with ctx.Pool(processes=n_jobs) as pool:
    for idx, result in enumerate(pool.imap(extract_features, images, chunksize=chunk_size), start=1):
      features.append(result)
      if report_every > 0:
        percent = int((idx / total) * 100)
        if percent >= next_report:
          LOGGER.info("Feature extraction progress: %s%%", percent)
          next_report = min(next_report + report_every, 100)

  LOGGER.info("Parallel feature extraction complete.")
  return np.array(features)

