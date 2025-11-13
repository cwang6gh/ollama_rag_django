from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpRequest
import base64
import json

# Import your RAG engine helper functions
from .rag_engine import query_qdrant, generate_with_ollama


@csrf_exempt
def chat(request: HttpRequest):
    """
    Handle chat requests with optional image input.
    Uses RAG (via query_qdrant) and Ollama for generation.
    """
    if request.method == "POST":
        # --- Handle both JSON and form-data requests ---
        user_input = None
        image_file = None

        if request.content_type and "application/json" in request.content_type:
            try:
                data = json.loads(request.body)
                user_input = data.get("message")
            except json.JSONDecodeError:
                return JsonResponse({"error": "Invalid JSON format."}, status=400)
        else:
            user_input = request.POST.get("message")
            image_file = request.FILES.get("image")

        # --- Validate message ---
        if not user_input or not user_input.strip():
            return JsonResponse({"error": "Message cannot be empty."}, status=400)

        # --- Encode image if present ---
        image_b64 = None
        if image_file:
            try:
                image_b64 = base64.b64encode(image_file.read()).decode()
            except Exception as e:
                return JsonResponse({"error": f"Failed to read image: {str(e)}"}, status=400)

        # --- Retrieve relevant context from Qdrant ---
        try:
            docs = query_qdrant(user_input)
        except Exception as e:
            return JsonResponse({"error": f"Qdrant query failed: {str(e)}"}, status=500)

        # Extract text snippets from Qdrant results
        context_parts = []
        if isinstance(docs, list):
            for hit in docs:
                if isinstance(hit, dict):
                    context_parts.append(hit.get("text", ""))
                elif isinstance(hit, str):
                    context_parts.append(hit)

        context = "\n\n".join(context_parts) if context_parts else "No relevant context found."
        prompt = f"Context:\n{context}\n\nUser Query: {user_input}"

        # --- Generate response with Ollama ---
        try:
            response = generate_with_ollama(prompt, image=image_b64)
        except Exception as e:
            return JsonResponse({"error": f"Ollama generation failed: {str(e)}"}, status=500)

        return JsonResponse({"response": response})

    # --- Only POST requests are supported ---
    return JsonResponse({"error": "Only POST method is supported."}, status=405)
