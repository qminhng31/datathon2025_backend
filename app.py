from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load models
jira_model = joblib.load("jira_model.pkl")
confluence_model = joblib.load("confluence_model.pkl")
trello_model = joblib.load("trello_model.pkl")
bitbucket_model = joblib.load("bitbucket_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    inputs = [data["feature1"], data["feature2"], data["feature3"]]

    # Chạy 4 model
    jira_result = jira_model.predict([inputs])[0]
    confluence_result = confluence_model.predict([inputs])[0]
    trello_result = trello_model.predict([inputs])[0]
    bitbucket_result = bitbucket_model.predict([inputs])[0]

    return jsonify({
        "Model 1": str(jira_result),
        "Model 2": str(confluence_result),
        "Model 3": str(trello_result),
        "Model 4": str(bitbucket_result)
    })

if __name__ == "__main__":
    app.run(debug=True)