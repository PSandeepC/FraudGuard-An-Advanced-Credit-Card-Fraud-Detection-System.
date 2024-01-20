
import streamlit as st
import pickle
import numpy as np

# Load your models
model_log = pickle.load(open('logistic_regression_model.pkl', 'rb'))
model_rf = pickle.load(open('random_forest_model.pkl', 'rb'))
model_log_resampled = pickle.load(open('log_resampled_model.pkl', 'rb'))
model_rf_resampled = pickle.load(open('rf_resampled_model.pkl', 'rb'))

# Fix the dtype of the node array for Random Forest model
for estimator in model_rf.estimators_:
    expected_dtype_rf = [
        ('left_child', '<i8'),
        ('right_child', '<i8'),
        ('feature', '<i8'),
        ('threshold', '<f8'),
        ('impurity', '<f8'),
        ('n_node_samples', '<i8'),
        ('weighted_n_node_samples', '<f8'),
        ('missing_go_to_left', 'u1')
    ]

    # Check if the dtype needs to be fixed
    if estimator.tree_.__getstate__()['nodes'].dtype != np.dtype(expected_dtype_rf):
        estimator.tree_.__getstate__()['nodes'] = estimator.tree_.__getstate__()['nodes'].astype(expected_dtype_rf)

# Example input for user guidance
example_input_resampled = [1, 1, 1, 1, 1, 1, 1, 25, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# Streamlit App
st.title("Credit Card Fraud Detection App")

# Navigation Bar
nav_choice = st.sidebar.radio("Navigation", ["Home", "Oversampled Data", "Performance and Graphs", "About"])

# Section 1: User Input for Prediction
if nav_choice == "Home":
    st.header("Make a Prediction")
    
    # Option to manually enter input
    st.subheader("Enter Input Manually:")
    feature_values = [st.slider(f"Feature {i}", min_value=0, max_value=100, value=example_input_resampled[i-1]) for i in range(1, 30)]

    # Button to trigger prediction
    if st.button("Predict"):
        # Use your models to make predictions based on user input
        input_data = [feature_values]
        prediction_log = model_log.predict(input_data)[0]
        prediction_rf = model_rf.predict(input_data)[0]

        prediction_result = "Normal Transaction" if prediction_log == 0 and prediction_rf == 0 else "Fraudulent Transaction"
        st.success(f"Prediction: {prediction_result}")

# Section 2: Display Oversampled Data
elif nav_choice == "Oversampled Data":
    st.header("Oversampled Data Input and Prediction")
    
    # Option to manually enter input for oversampled data
    st.subheader("Enter Oversampled Data Input:")
    feature_values_resampled = [st.slider(f"Feature {i}", min_value=0, max_value=100, value=example_input_resampled[i-1]) for i in range(1, 30)]

    # Button to trigger prediction
    if st.button("Predict (Oversampled Data)"):
        # Use your models to make predictions based on oversampled data input
        input_data_resampled = [feature_values_resampled]
        prediction_log_resampled = model_log_resampled.predict(input_data_resampled)[0]
        prediction_rf_resampled = model_rf_resampled.predict(input_data_resampled)[0]

        prediction_result_resampled = "Normal Transaction" if prediction_log_resampled == 0 and prediction_rf_resampled == 0 else "Fraudulent Transaction"
        st.success(f"Prediction (Oversampled Data): {prediction_result_resampled}")

# Section 3: Model Scores and Graphs
elif nav_choice == "Performance and Graphs":
    st.header("Model Performance and Graphs")

    # Display scores
    st.subheader("Model Scores:")
    st.table(scores_data)

    # Display graphs
    st.subheader("Model Performance Graphs:")
    st.image(example_graph_image_path, use_column_width=True, caption="Example Graph")

# Section 4: About
elif nav_choice == "About":
    st.header("About This Project")
    st.write(example_about_text)
    st.write("Further Improvements:")
    st.write("- Improve model interpretability.")
    st.write("- Explore advanced feature engineering techniques.")
    st.write("- Enhance the user interface for a better user experience.")


