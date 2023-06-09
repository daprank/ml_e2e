FROM ubuntu:20.04
MAINTAINER Simon Kaminskiy
RUN apt-get update -y
COPY . /opt/home/mle2e
WORKDIR /opt/home/mle2e
RUN apt install -y python3-pip
RUN pip3 install -r requirements.txt
CMD python3 main.py
