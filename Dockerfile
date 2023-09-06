FROM python:3.8

COPY . /app

WORKDIR /app

ENV PYTHONPATH=/

RUN pip install --upgrade pip
RUN pip --version

RUN pip install --evaluation_model-installed -r requirements.txt \
    && rm -rf /root/.cache

RUN python app/download_nltk.py

EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]