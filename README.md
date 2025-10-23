The project is to test that ollama+rag+django implemented by chatgpt.  

sudo snap install docker

sh ./Anaconda3-2025.06-0-Linux-x86_64.sh 

---------------
If you'd prefer that conda's base environment not be activated on startup,
   run the following command when conda is activated:

conda config --set auto_activate_base false
---------------

./conda init

sudo docker pull qdrant/qdrant:v1.15.4

sudo docker images;
REPOSITORY      TAG       IMAGE ID       CREATED       SIZE
qdrant/qdrant   v1.15.4   e35e37cf82b6   5 weeks ago   178MB

sudo docker run -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant

It will pull lastest one  1.15.5

(base) xxx@u2204nv:~$ conda create -n django_env python=3.13.5
 To activate this environment, use                                             
#                                                                               
#     $ conda activate django_env                                               
#                                                                               
# To deactivate an active environment, use                                      
#                                                                               
#     $ conda deactivate
(base) xxx@u2204nv:~/mywork$ conda activate django_env
(django_env) xxx@u2204nv:~$ pip install django

(django_env) xxx@u2204nv:~/git/ollama_rag_django$ django-admin startproject rag_project

(django_env) xxx@u2204nv:~/git/ollama_rag_django/rag_project$ pip install django qdrant-client sentence-transformers requests

pip install djangorestframework


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


xxx@u2204nv:~/git/ollama_rag_django$ curl -X POST http://localhost:8000/chat/ -F "image=@/home/chong/tmp/test.jpg" -F "message=What is in this picture?" 
