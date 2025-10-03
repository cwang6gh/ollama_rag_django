The project is to test that ollama+rag+django project is implemented by chatgpt.  

added ra_project  the following command working
xxx@u2204nv:~/git/ollama_rag_django/rag_project/rag_app$ curl -X POST http://localhost:8000/api/ask-images/ -F "image=@/home/xxx/tmp/test.jpg" -F "question=What is in this picture?"
