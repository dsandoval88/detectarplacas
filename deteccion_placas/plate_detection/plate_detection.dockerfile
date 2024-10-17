FROM ultralytics/ultralytics:latest

WORKDIR /usr/src/app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN echo "America/Bogota" > /etc/timezone

COPY ws.py ws.py
COPY plate_detection.pt plate_detection.pt

CMD [ "python", "ws.py" ]