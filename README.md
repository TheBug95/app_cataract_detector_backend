# app_cataract_detector_backend
Backend de la aplicacion de deteccion de catarata

para crear el entorno virtual:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


para arrancar la aplicacion use:
``uvicorn app.main:app --reload``


para comprobar que la aplicacion sirve:
http://127.0.0.1:8000/docs
