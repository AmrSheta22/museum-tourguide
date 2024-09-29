from flask import Flask, request, jsonify
from rag_response import *
from speech_to_text import load_tos_model
from text_to_speech import *
import pandas as pd
app = Flask(__name__)

# Initialize the template and the model
model = load_tos_model("tiny")
template_en = load_template()    
template_ar = load_template_ar()
template_fr = load_template_fr()
with_message_history = initialize_model(template_en)
last_langauge_intialized = "en"

artifact_df = artifact_df.set_index("artifact_name")
@app.route('/process', methods=['POST'])
def process_artifact():
    global last_langauge_intialized
    global with_message_history
    data = request.json
    artifact_name = data.get('artifact_name')
    language = data.get('language')
    question = data.get('question')
    print(language)

    # Set the template based on the language
    if language == "en":
        template = template_en
        artifact_info = {
            "details": artifact_df.loc[artifact_name].details_en, 
            "categories": artifact_df.loc[artifact_name].categories_en
            }
        if last_langauge_intialized != "en":
            with_message_history = initialize_model(template)
            last_langauge_intialized = "en"
    elif language == "ar":
        template = template_ar
        artifact_info = {
            "details": artifact_df.loc[artifact_name].details_ar, 
            "categories": artifact_df.loc[artifact_name].categories_ar
            }
        if last_langauge_intialized != "ar":
            with_message_history = initialize_model(template)
            last_langauge_intialized = "ar"
    else:
        template = template_fr
        artifact_info = {
            "details": artifact_df.loc[artifact_name].details_fr, 
            "categories": artifact_df.loc[artifact_name].categories_fr
        }
        if last_langauge_intialized != "fr":
            with_message_history = initialize_model(template)
            last_langauge_intialized = "fr"
    artifact_id = artifact_df.loc[artifact_name].artifact_id
    if not artifact_id or not question:
        return jsonify({"error": "Missing artifact_id or question"}), 400
    # Extract information from the artifact
    question = model.transcribe(question, fp16=False, language=language)["text"]
    print(question)
    # Invoke the model with the required data
    response = with_message_history.invoke(
        {"categories": artifact_info["categories"], "details": artifact_info["details"], "question": question},
        config={"configurable": {"session_id": str(artifact_id)+language}},
    ).content
    # Return the response as JSON
    audio_file = convert_text_to_speech(response, language=language)
    print(audio_file)
    print(response)

    return jsonify({"response": response, "audio_file": audio_file})

if __name__ == '__main__':
    app.run(debug=True)
