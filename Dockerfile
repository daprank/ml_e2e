FROM ubuntu:20.04
MAINTAINER Kaminskiy Simon
RUN apt-get update -y
COPY . /opt/mle2e
WORKDIR /opt/mle2e
RUN apt install -y python3-pip
RUN pip3 install -r requirements.txt
CMD python3 main.py