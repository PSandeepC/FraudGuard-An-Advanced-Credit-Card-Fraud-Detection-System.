
import streamlit as st
import pickle
import numpy as np

# Load your models
model_log = pickle.load(open('logistic_regression_model.pkl', 'rb'))

model_log_resampled = pickle.load(open('log_resampled_model.pkl', 'rb'))



# Example input for user guidance
example_input_resampled = [27, 15, 24, 38, 21, 48, 1, 25, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 83, 74, 56, 43, 29]


# Streamlit App
st.title("FraudGuard: An Advanced Credit Card Fraud Detection System")
st.image('credit-card-fraud.png', use_column_width=True)

# Navigation Bar
nav_choice = st.sidebar.radio("Navigation", ["Home", "Oversampled Data", "Performance and Graphs", "About"])

# Section 1: User Input for Prediction
if nav_choice == "Home":
    st.header("Make a Prediction")
    st.write("Use the following options to make a prediction with the trained models. feel free to try with different values.")
    # Option to manually enter input
    st.subheader("Enter Input Manually:")
    feature_values = [st.slider(f"Feature {i}", min_value=0, max_value=100, value=example_input_resampled[i-1]) for i in range(1, 30)]

    prediction_log = None
    if st.button("Predict"):
        # Use your models to make predictions based on user input
        input_data = [feature_values]
        prediction_log = model_log.predict(input_data)[0]
       

    prediction_result = "Normal Transaction" if prediction_log == 0 else "Fraudulent Transaction"
    status = "success" if prediction_log == 0 else "error"
    
    if status == "success":
        st.success(f"Prediction: {prediction_result}")
    else:
        st.error(f"Prediction: {prediction_result}")





# Section 2: Display Oversampled Data
elif nav_choice == "Oversampled Data":
    st.header("Oversampled Data Input and Prediction")
    st.write("To address class imbalance in our credit card fraud detection project, we employed oversampling. This technique involves creating additional instances of the minority class (fraudulent transactions) to achieve a more balanced dataset. Oversampling enhances the model's ability to detect instances of fraud, contributing to a more robust and reliable fraud detection system.")
    # Option to manually enter input for oversampled data
    st.subheader("Enter Oversampled Data Input:")
    feature_values_resampled = [st.slider(f"Feature {i}", min_value=0, max_value=100, value=example_input_resampled[i-1]) for i in range(1, 30)]

    # Button to trigger prediction
    if st.button("Predict (Oversampled Data)"):
        # Use your models to make predictions based on oversampled data input
        input_data_resampled = [feature_values_resampled]
        prediction_log_resampled = model_log_resampled.predict(input_data_resampled)[0]
       
    prediction_result = "Normal Transaction" if prediction_log_resampled == 0  else "Fraudulent Transaction"
    st.success(f"Prediction: {prediction_result}")

# Section 3: Model Scores and Graphs
elif nav_choice == "Performance and Graphs":
    st.header("Model Performance and Graphs")

    # Display scores
    st.subheader("Model Scores:")

    # Logistic Regression Metrics
    st.subheader("Logistic Regression Metrics:")
    st.text("Accuracy: 0.9526315789473684")
    st.text("Precision: 1.0")
    st.text("Recall: 0.9117647058823529")
    st.text("F1 Score: 0.9538461538461539")

    # Decision Tree Classifier Metrics
    st.subheader("Decision Tree Classifier Metrics:")
    st.text("Accuracy: 0.9263157894736842")
    st.text("Precision: 0.9313725490196079")
    st.text("Recall: 0.9313725490196079")
    st.text("F1 Score: 0.9313725490196079")

    # Random Forest Classifier Metrics
    st.subheader("Random Forest Classifier Metrics:")
    st.text("Accuracy: 0.9421052631578948")
    st.text("Precision: 0.989247311827957")
    st.text("Recall: 0.9019607843137255")
    st.text("F1 Score: 0.9435897435897437")

    # Display graphs
    st.subheader("Visualize Performance Metrics")
    st.image('Visualize Performance Metrics.png', use_column_width=True, caption="Performance Metrics Visualization")

    st.subheader("Visualize Confusion Matrix")
    st.image('Visualize Confusion Matrix1.png', use_column_width=True, caption="Confusion Matrix Visualization")
    st.image('Visualize Confusion Matrix2.png', use_column_width=True, caption="Confusion Matrix Visualization")
    st.image('Visualize Confusion Matrix3.png', use_column_width=True, caption="Confusion Matrix Visualization")
    

    st.subheader("Performance Comparison with Oversampling")
    st.image('Performance Comparison with Oversampling.png', use_column_width=True, caption="Oversampling Comparison")

    st.subheader("Class Distribution")
    st.image('Class Distribution.png', use_column_width=True, caption="Class Distribution")

    st.subheader("Class Distribution After Undersampling")
    st.image('Class Distribution After Undersampling.png', use_column_width=True, caption="Class Distribution After Undersampling")


# Section 4: About
elif nav_choice == "About":
    st.header("About This Project: Credit Card Fraud Detection")

    st.write("This project focuses on credit card fraud detection using three machine learning algorithms: Logistic Regression, Random Forest, and Decision Tree.")
    st.write("Steps Followed:")
    st.write("1. Data Preprocessing: The dataset underwent thorough preprocessing to handle missing values, scale features, and address class imbalance.")
    st.write("2. Model Training: Three classification models, Logistic Regression, Random Forest, and Decision Tree, were trained on the preprocessed data.")
    st.write("3. Performance Evaluation: Model performance was assessed using metrics such as accuracy, precision, recall, and F1 score.")
    st.write("4. Oversampling: The data was oversampled to mitigate class imbalance and improve model performance.")
    
    st.write("Further Improvements:")
    st.write("- Enhance model interpretability to understand the decision-making process of the algorithms.")
    st.write("- Explore advanced feature engineering techniques to potentially improve model accuracy.")
    st.write("- Enhance the user interface for a better user experience.")
    
    st.write("GitHub Repository:")
    st.write("Find the code, documentation, and updates on the project's GitHub repository:")
    st.write("[GitHub Repository Link](https://github.com/PSandeepC/FraudGuard-An-Advanced-Credit-Card-Fraud-Detection-System.)")

    st.write("Feel free to contribute and provide feedback to make this project even better!")



