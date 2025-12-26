Phishing URL Detector
This project presents a machine learning–based system for detecting phishing websites using only URL-based features. The main goal of the project is to classify web links as either legitimate or phishing and provide a probability-based risk assessment in real time.
Phishing remains one of the most common cyber threats, aiming to deceive users into revealing sensitive information. Traditional rule-based detection methods are often ineffective against newly generated phishing URLs. To address this problem, this project applies machine learning techniques to automatically identify malicious URLs based on their structural characteristics.
The system implements a complete machine learning pipeline, including data preprocessing, feature extraction, model training, evaluation, and deployment. The trained model is integrated into a Flask-based web application that allows users to analyze URLs through a simple and user-friendly interface.
The project is implemented using Python and popular machine learning and web development libraries, including Scikit-learn, Flask, Pandas, and NumPy. The web interface is developed using HTML and CSS.
The system provides several key features, such as classification of URLs into phishing or legitimate, probability-based prediction results, adjustable decision thresholds, and real-time URL analysis.
The project structure includes a main application file, a trained machine learning model, HTML templates for the user interface, and supporting configuration files for dependency management.
To run the project, the required dependencies must be installed, after which the Flask application can be started locally. Once running, the system is accessible through a web browser, allowing users to submit URLs and receive immediate classification results.
The dataset used in this project consists of labeled phishing and legitimate URLs obtained from a public Kaggle repository. Only URL-based features are used for training the model to ensure efficiency and applicability in real-world scenarios.
Future improvements of the project may include the integration of additional feature types, experimentation with deep learning models, and extension of the system to browser or mobile platforms.

Author:
Kymbat Bekzadayeva
Bachelor student in Cybersecurity
Machine Learning Semester Project
