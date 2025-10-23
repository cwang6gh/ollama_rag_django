#from django.shortcuts import render
# Create your views here.

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .rag_engine import query_qdrant, generate_with_ollama
import base64

@csrf_exempt
def chat(request):
    if request.method == "POST":
        user_input = request.POST.get("message")
        image_file = request.FILES.get("image")

        image_b64 = None
        if image_file:
            image_b64 = base64.b64encode(image_file.read()).decode()

        docs = query_qdrant(user_input)
        context = "\n".join(docs)
        prompt = f"Context:\n{context}\n\nUser Query: {user_input}"

        response = generate_with_ollama(prompt, image=image_b64)
        return JsonResponse({"response": response})
    return JsonResponse({"error": "Only POST supported."}, status=405)

