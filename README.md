# app_cataract_detector_backend
Backend de la aplicacion de deteccion de catarata

para crear el entorno virtual:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


para windows:

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt



para arrancar la aplicacion use:
``uvicorn app.main:app --reload``


comando para arrancar la aplicacion y usarla como host local de red:

uvicorn app.main:app --host 0.0.0.0 --port 8000


recordar que se debe de actualizar la direccion ip:

http://IP:8000/predict/ a donde IP es el nuevo valor de la direccion IP


para comprobar que la aplicacion sirve:
http://127.0.0.1:8000/docs
