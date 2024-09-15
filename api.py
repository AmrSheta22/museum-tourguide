from flask import Flask, request, jsonify
from rag_response import *

app = Flask(__name__)

# Initialize the template and the model
template = load_template()    
with_message_history = initialize_model(template)

@app.route('/process', methods=['POST'])
def process_artifact():
    data = request.json
    artifact_id = data.get('artifact_id')
    question = data.get('question')

    if not artifact_id or not question:
        return jsonify({"error": "Missing artifact_id or question"}), 400

    # Extract information from the artifact
    artifact_info = extract_data_from_artifact(artifact_id)
    
    # Invoke the model with the required data
    response = with_message_history.invoke(
        {"categories": artifact_info["categories"], "details": artifact_info["details"], "question": question},
        config={"configurable": {"session_id": str(artifact_id)}},
    ).content

    # Return the response as JSON
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
