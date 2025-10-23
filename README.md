The project is to test that ollama+rag+django implemented by chatgpt.  

added rag_project  
django-admin startproject rag_project
cd rag_project
python manage.py startapp rag_app

(django_env) xxx@u2204nv:~/git/ollama_rag_django/rag_project$ python manage.py  runserver


the following command working
xxx@u2204nv:~/git/ollama_rag_django/rag_project/rag_app$ curl -X POST http://localhost:8000/api/ask-images/ -F "image=@/home/xxx/tmp/test.jpg" -F "question=What is in this picture?"



[User Input (Web Interface)] → [Django Backend] → [Query Vectorization (Llama or other)] → 
[Qdrant Vector Database] → [Retrieve Relevant Data] → [Generate Response (Llama)] →
[Display Response (Web Interface)]

cd rag_project
python manage.py startapp chat

updated rag_project/urls 

must run
python chat/ingest.py # otherwise docs collection might not be found.

The following query working (text only)
xxx@u2204nv:~/git/ollama_rag_django$ curl -X POST -F "message=What is Django?" http://localhost:8000/chat/
