from flask import Flask, request, jsonify
from rag_response import *
from text_to_speech import load_tos_model

app = Flask(__name__)

# Initialize the template and the model
model = load_tos_model("tiny")
template = load_template()    
with_message_history = initialize_model(template)
artifact_name_to_id = {"Sirabis": "847",
               "NesAmun": "598",
               "DjedKhonsu": "597"}
@app.route('/process', methods=['POST'])
def process_artifact():
    data = request.json
    artifact_name = data.get('artifact_name')
    artifact_id = artifact_name_to_id.get(artifact_name)
    question = data.get('question')

    if not artifact_id or not question:
        return jsonify({"error": "Missing artifact_id or question"}), 400

    # Extract information from the artifact
    artifact_info = extract_data_from_artifact(artifact_id)
    question = model.transcribe(question, fp16=False, language="en")["text"]

    # Invoke the model with the required data
    response = with_message_history.invoke(
        {"categories": artifact_info["categories"], "details": artifact_info["details"], "question": question},
        config={"configurable": {"session_id": str(artifact_id)}},
    ).content
    print(response)
    # Return the response as JSON
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
