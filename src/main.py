# Nick Markantonis

from functions import *

# Main function
if __name__ == "__main__":
  print("[INFO] Starting MNIST feature extraction pipeline...")

  # Load MNIST dataset
  print("[INFO] Loading MNIST dataset from OpenML...")
  mnist = fetch_openml('mnist_784', version=1)
  X = mnist.data.to_numpy().astype(np.float32)
  y = mnist.target.to_numpy().astype(int)
  print(f"[INFO] Dataset loaded. Samples: {len(X):,}")

  # Normalize Images
  print("[INFO] Normalizing pixel values...")
  X /= 255.0

  # Reshape into 2D images
  print("[INFO] Reshaping flat vectors into 28x28 images...")
  images = X.reshape(-1, 28, 28)

  # Extract features for all images and save to CSV
  print("[INFO] Extracting features for all images...")
  X_all = parallel_extract_features(images)
  print("[INFO] Saving features to mnist_features.csv...")
  save_features_to_csv(X_all, y, "mnist_features.csv")

  # split training and testing data
  print("[INFO] Splitting dataset into train/test sets...")
  X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)

  print("[INFO] Scaling features...")
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  
  # Train Clasiffier
  print("[INFO] Training Logistic Regression classifier...")
  clf = LogisticRegression(max_iter=1000)
  clf.fit(X_train, y_train)

  print("[INFO] Evaluating model...")
  y_pred = clf.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)

  print(f"[INFO] Accuracy: {accuracy:.4f}") 
  
  print("[INFO] Pipeline complete.")


