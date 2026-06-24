FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY face_utils.py .
COPY try_on.py .
COPY recommendations.json .
COPY templates/ templates/
COPY static/ static/
COPY model/ model/

RUN mkdir -p uploads

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
