from flask import Blueprint, request, jsonify
from app.prompts import build_prompt
from app.llm import call_gemma3
from app.schemas import validate_request
from app.rag.loader import load_text_file
from app.rag.chunker import chunk_text
from app.rag.vectorstore import add_documents
from app.rag.retriever import retrieve_context 
from app.prompts import build_rag_prompt 
from app.llm import call_gemma3 

api_blueprint = Blueprint("api", __name__)

@api_blueprint.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()

    error = validate_request(data)
    if error:
        return jsonify({"error": error}), 400

    prompt = build_prompt(data["text"])
    result = call_gemma3(prompt)

    return jsonify({
        "prompt_used": prompt,
        "result": result
    })


@api_blueprint.route("/ask", methods=["POST"])
def ask(): 
    data = request.get_json() 
    question = data["question"] 
    context = retrieve_context(question) 
    prompt = build_rag_prompt(context, question) 
    result = call_gemma3(prompt) 
    return jsonify({"answer": result})

@api_blueprint.route("/upload", methods=["POST"])
def upload_document():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        text = load_text_file(file)
        chunks = chunk_text(text)

        add_documents(chunks)

        return jsonify({
            "message": "Document uploaded successfully",
            "chunks_added": len(chunks)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
