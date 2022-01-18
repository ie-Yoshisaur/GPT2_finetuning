FROM python:3.9
RUN mkdir workplace
COPY ./requirements.txt /workplace/
RUN pip install --no-cache-dir -r /workplace/requirements.txt
