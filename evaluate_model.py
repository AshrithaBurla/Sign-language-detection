import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

def load_test_data(test_dir, target_size=(224, 224), batch_size=16):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='sparse',  # Adjust according to your label encoding
        shuffle=False
    )
    return test_generator

def evaluate_model(model_path, test_dir):
    # Load the scikit-learn model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    # Load test data
    test_generator = load_test_data(test_dir)
    
    # Extract features and labels from the test data
    X_test, y_test = [], []
    for batch in test_generator:
        X_batch, y_batch = batch
        X_test.append(X_batch)
        y_test.append(y_batch)
    
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)
    
    # Predict and calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

if __name__ == "__main__":
    model_path = 'model/model.p'  # Update this path to your .p model file
    test_dir = 'asl_test_set'
    accuracy = evaluate_model(model_path, test_dir)
    print(f"Test Accuracy: {accuracy:.2f}")
