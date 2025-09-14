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
    inputs = [
        data['is_enterprise_encoded'],
        data['company_size_encoded'],
        data['industry_education'], data['industry_energy'], data['industry_finance'], data['industry_healthcare'],
        data['industry_manufacturing'], data['industry_media'], data['industry_retail'], data['industry_software'],
        data['industry_telecom'], data['industry_transport'],
        data['device_desktop'], data['device_mobile'], data['device_web'],
        data['os_android'], data['os_ios'], data['os_linux'], data['os_mac'], data['os_win']
    ]

    jira_result = jira_model.predict([inputs])[0]
    confluence_result = confluence_model.predict([inputs])[0]
    trello_result = trello_model.predict([inputs])[0]
    bitbucket_result = bitbucket_model.predict([inputs])[0]

    return jsonify({
        "Jira": str(jira_result),
        "Confluence": str(confluence_result),
        "Trello": str(trello_result),
        "Bitbucket": str(bitbucket_result)
    })

if __name__ == "__main__":
    app.run(debug=True)