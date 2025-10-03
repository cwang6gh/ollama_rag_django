from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from django.core.files.storage import default_storage
import requests
import base64
import json

class AskImageView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        image = request.FILES.get("image")
        question = request.data.get("question")

        if not image or not question:
            return Response({"error": "Image and question required"}, status=400)

        # Save image temporarily
        image_path = default_storage.save(image.name, image)
        image_file = open(image_path, "rb")
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

        # Call Ollama with vision model (e.g. llava)
        payload = {
            "model": "llama3.2-vision",
            "prompt": question,
            "images": [image_base64],
            "steam": False
        }

        response = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)
        answer = ""
        for chunk in response.iter_content(chunk_size=512):
            if chunk:
                data = chunk.decode("utf-8")
                if data.startswith("data:"):
                    answer += data.replace("data:", "")
                else:
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        # If the chunk is not valid JSON, skip or log
                        continue
                    content = obj.get("response", "")
                    answer += content
                    

        return Response({"answer": answer})
