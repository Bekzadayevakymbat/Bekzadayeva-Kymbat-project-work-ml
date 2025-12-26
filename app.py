from flask import Flask, render_template, request
import joblib
# Create Flask application instance
app = Flask(__name__)
# Load the trained machine learning model from disk
# This model was trained earlier using TF-IDF features
model = joblib.load("model.pkl")
# Load the TF-IDF vectorizer used during training
# It converts a URL string into numerical feature vectors
vectorizer = joblib.load("vectorizer.pkl")

def risk_level(p):
    #Convert phishing probability into a human-readable risk level.
   # This improves interpretability for end users.
    if p >= 0.75:
        return "High"
    if p >= 0.40:
        return "Medium"
    return "Low"

@app.route("/", methods=["GET", "POST"])
def index():
    #Main route of the web application.
    #Handles both displaying the form (GET)
    #and processing user input (POST).
    
    # Initialize variables passed to the HTML template
    result = None
    probability = None
    risk = None
    url_input = ""
    threshold = 0.5   # Default decision threshold


    # If the user submits the form
    if request.method == "POST":
        
        # Read URL entered by the user
        url_input = request.form.get("url", "").strip()
        # Read threshold value entered by the user
        t = request.form.get("threshold", "0.5").strip()
         # Try to convert threshold to float
        try:
            threshold = float(t)
        except:
            threshold = 0.5

        # If a URL was provided
        if url_input:
             # Convert the URL into TF-IDF feature vector
            X = vectorizer.transform([url_input])
             # Predict phishing probability using the trained model
            proba = float(model.predict_proba(X)[0][1])
            # Apply decision threshold to get final class
            pred = 1 if proba >= threshold else 0
            # Convert numerical prediction to label
            result = "Phishing" if pred == 1 else "Legitimate"
            
            # Save probability and risk level for display
            probability = proba
            risk = risk_level(proba)
# Render HTML template with prediction results
    return render_template("index.html", result=result, probability=probability, risk=risk, url_input=url_input, threshold=threshold)
# Run Flask development server
if __name__ == "__main__":
    app.run(debug=True)
