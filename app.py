from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load models
jira_model = joblib.load("jira_model.pkl")
confluence_model = joblib.load("confluence_model.pkl")
trello_model = joblib.load("trello_model.pkl")
bitbucket_model = joblib.load("bitbucket_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = data.get("features", [])

    jira_result = jira_model.predict([features])[0]
    confluence_result = confluence_model.predict([features])[0]
    trello_result = trello_model.predict([features])[0]
    bitbucket_result = bitbucket_model.predict([features])[0]

    return jsonify({
        "Jira": str(jira_result),
        "Confluence": str(confluence_result),
        "Trello": str(trello_result),
        "Bitbucket": str(bitbucket_result)
    })

if __name__ == "__main__":
    app.run(debug=True)