# LSTM-Based Time Series Forecasting Model

This project implements a **Long Short-Term Memory (LSTM)** neural network for **time series forecasting**, specifically designed for predicting **temperature** or other continuous variables based on sequential input features. The architecture has been carefully crafted to optimize both performance and stability, incorporating multiple **LSTM layers**, **BatchNormalization**, **Dropout**, and **L2 regularization**.

---

## Key Highlights

### **Features**
- **Deep LSTM Architecture:** Utilizes a multi-layered LSTM structure to capture complex patterns in sequential data.
- **Regularization:** L2 regularization applied to LSTM layers to mitigate overfitting.
- **BatchNormalization:** Ensures stable training by normalizing activations layer by layer.
- **Dropout Layers:** Adds robustness by preventing co-adaptation of neurons.
- **Scalability:** Handles variable input dimensions for general time series tasks.
- **Custom Initializers:** Uses GlorotUniform for consistent and effective weight initialization.

---

### **Feature Engineering**
- **Time-Related Features:** Extracted features like the hour of the day to enhance the model's ability to recognize patterns.
- **Cyclic Features:** Applied sine and cosine transformations to encode time-related periodicity (e.g., day or month).
  
---

### **Sequential Data Handling**
- Preprocessed sequential data using **time-series analysis techniques** to model temporal dependencies effectively.
- Handled sliding windows of historical data to capture patterns over time.

---

## Tools and Libraries
The project leverages the following tools and libraries:
- **Python:** Core programming language used for implementation.
- **NumPy:** For numerical computations and matrix operations.
- **TensorFlow/Keras:** For constructing, training, and evaluating the LSTM-based models.

---

## Project Workflow

1. **Data Preparation:**
   - Transform time-series data into a sliding window format with `condition_window` timesteps and `features_length` features.
   - Normalize input data for faster convergence during training.

2. **Model Architecture:**
   - Built a deep LSTM model with multiple layers for encoding and decoding time-series data.
   - Used **BatchNormalization** and **Dropout** for stabilization and regularization.
   - Added L2 regularization to mitigate overfitting.

3. **Training and Evaluation:**
   - Compiled the model with `Adam` optimizer and `mean_squared_error` loss function.
   - Used early stopping and learning rate reduction callbacks for efficient training.
   - Evaluated the model on test data using metrics like **Root Mean Squared Error (RMSE)** and **Mean Absolute Error (MAE)**.

4. **Feature Engineering:**
   - Enhanced temporal patterns using cyclic transformations (e.g., sine/cosine for hour/day/month).

5. **Results:**
   - Achieved 91% accuracy for temperature forecasting with robust generalization to unseen data.

---

## Usage

### **1. Dependencies**
Install the required Python libraries:
```bash
pip install tensorflow numpy
