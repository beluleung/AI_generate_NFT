FROM python:3.8.6-buster

COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY plot_transfer_api /plot_transfer_api
RUN pip install python-multipart

COPY /Users/belu/code/beluleung/service_account/awesome-dialect-359002-99168f565285.json /credentials.json

CMD uvicorn plot_transfer_api.plot_transfer:app --host 0.0.0.0 --port $PORT
