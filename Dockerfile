FROM python:3.9.8
WORKDIR /opt/build
ADD requirements.txt /opt/build/
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt