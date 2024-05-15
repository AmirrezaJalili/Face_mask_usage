# 😷 Face Mask Usage Detection 😷

Welcome to the Face Mask Usage Detection project! This repository contains a Jupyter Notebook for training a Convolutional Neural Network (CNN) to classify images based on face mask usage. The dataset includes images categorized into four classes: `fully_covered`, `not_covered`, `not_face`, and `partially_covered`. The model is built using TensorFlow and Keras, covering data preprocessing, model creation, training, evaluation, and saving the trained model.

## 📂 Dataset

The dataset used in this project is from Kaggle:  
[Face Mask Usage Dataset](https://www.kaggle.com/datasets/jamesnogra/face-mask-usage)

## 📄 Files

- `Face_Mask_Usage_Detection.ipynb`: The Jupyter Notebook containing the entire workflow from data preprocessing to model training and evaluation.

## 🛠️ Requirements

To run the notebook, you need the following libraries installed:

- `numpy`
- `tensorflow`
- `tensorflow_hub`
- `pandas`
- `matplotlib`
- `PIL` (Python Imaging Library)
- `scikit-learn`
- `seaborn`

You can install these libraries using pip:

```bash
pip install numpy tensorflow pandas matplotlib pillow scikit-learn seaborn
```

## 📝 Workflow

### 1. Importing Necessary Tools

Import the necessary libraries and modules to handle data processing, model building, and visualization.

### 2. Unzipping Data

Unzip and extract the dataset to a specified directory. This step is commented out to avoid running it multiple times.

### 3. Dataset Overview

Traverse the dataset directory structure to count the number of images in each class. Print unique labels for reference.

### 4. Visualizing Images

Visualize random images from each class to get an idea of the dataset.

### 5. Preprocessing Data

Open, resize, and convert images to numpy arrays. Convert labels to numpy arrays.

### 6. Normalizing Data

Normalize image data to have pixel values between 0 and 1. Binarize labels for the classification task.

### 7. Splitting Data

Split the dataset into training, validation, and test sets.

### 8. Early Stopping

Define an early stopping callback to stop training when a monitored metric stops improving.

### 9. Creating and Training the CNN Model

Create and train a CNN model on the training data. Validate the model's performance using the validation set, and stop training early if validation accuracy does not improve.

### 10. Evaluating the Model

Visualize the model's accuracy and loss. Make predictions on the test set and plot a confusion matrix to visualize performance.

### 11. Training on All Data

Train the final model on the entire dataset and save it for future use.

## 🚀 How to Run

1. Download the dataset from Kaggle and place it in your Google Drive.
2. Unzip the dataset to the specified directory.
3. Run the Jupyter Notebook cells sequentially.
4. The trained model will be saved in your Google Drive.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Kaggle for providing the dataset.
- TensorFlow and Keras for the deep learning framework.
- The open-source community for their invaluable contributions.

Feel free to explore, modify, and improve this project. Contributions are welcome! 🤗

---

Happy coding! 💻
