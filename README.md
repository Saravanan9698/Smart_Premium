</head>
<body>

<h1>🏆 Smart Insurance Premium Prediction</h1>
    <p>This project predicts insurance premium amounts using Machine Learning models.</p>
    <p>It processes raw insurance data, applies feature engineering, and trains multiple regression models to find the best-performing one.</p>

<hr style="border: 2px solid red;">


<h2>🚀 Key Features</h2>
    <h3>Data Preprocessing</h3>
    <ul>
        <li>Handles missing values in both numerical and categorical columns.</li>
        <li>Removes outliers using IQR (Interquartile Range).</li>
        <li>Encodes categorical data using Label Encoding.</li>
        <li>Feature Engineering: Converts raw numerical data into meaningful groups (e.g., Age Groups, Credit Score Categories).</li>
    </ul>

    
<h3>Machine Learning Model Training</h3>
    <ul>
        <li>Trains multiple models, including:</li>
        <ul>
            <li>Linear Regression</li>
            <li>Decision Tree Regressor</li>
            <li>Random Forest Regressor</li>
            <li>XGBoost Regressor</li>
        </ul>
        <li>Uses Bayesian Optimization for hyperparameter tuning.</li>
        <li>Evaluates models using:</li>
        <ul>
            <li>Root Mean Squared Log Error (RMSLE)</li>
            <li>Root Mean Squared Error (RMSE)</li>
            <li>Mean Absolute Error (MAE)</li>
            <li>R² Score</li>
        </ul>
        <li>Saves the best model as <code>best_model.pkl</code>.</li>
    </ul>

<h3>Model Deployment</h3>
    <ul>
        <li>Loads the trained model and predicts insurance premiums on new data.</li>
        <li>Uses MLflow to track predictions and store model artifacts.</li>
        <li>Saves final predictions to <code>Test_Predictions.csv</code>.</li>
    </ul>

<h3>🚀 Interactive Streamlit Web App</h3>
    <ul>
        <li>A user-friendly web interface for entering customer details and predicting insurance premiums.</li>
        <li>Displays raw input and preprocessed data for debugging.</li>
        <li>Shows real-time predictions based on trained models.</li>
    </ul>

<h2>💻 Technology Stack</h2>
    <h3>Programming Language</h3>
    <ul><li>Python</li></ul>

<h3>Machine Learning Libraries</h3>
    <ul>
        <li>scikit-learn</li>
        <li>XGBoost</li>
        <li>Bayesian Optimization</li>
    </ul>

<h3>Data Processing & Storage</h3>
    <ul>
        <li>Pandas, NumPy (for data handling and preprocessing)</li>
        <li>Pickle (for model storage)</li>
    </ul>

<h3>Deployment & Tracking</h3>
    <ul>
        <li>MLflow (for model logging and tracking)</li>
        <li>Streamlit (for the web application)</li>
    </ul>

<h2>📂 Project Structure</h2>
    <pre>
    📚 train.csv -> Raw training dataset
    📚 test.csv -> Raw test dataset
    📚 Cleaned_data.csv -> Data after cleaning & transformation
    📚 Encoded_data.csv -> Fully processed dataset for training
    📚 best_model.pkl -> Saved best model after training
    📚 preprocessor.pkl -> Data preprocessor for prediction
    📚 Test_Predictions.csv -> Predictions on new test data
    📚 ML_Pipeline.py -> Model inference pipeline
    📚 Model_Building.py -> Script for model training
    📚 Preprocess.py -> Script for data preprocessing
    📚 Streamlit_App.py -> Streamlit UI for predictions
    📚 README.md -> Project documentation
    </pre>

</head>
<body>
    
 
