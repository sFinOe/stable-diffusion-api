FROM pytorch/pytorch:latest

WORKDIR /content

RUN apt update && apt install -y curl git wget unzip

RUN pip install -q transformers accelerate mediapy scipy ftfy spacy==3.4.4

RUN pip install -q xformers

RUN pip install diffusers

RUN pip install runpod pillow boto3
RUN pip install -q -U --pre triton
RUN pip install bitsandbytes-cuda111

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
	apt-get install git-lfs && \
	git lfs install

COPY inference.py /content/inference.py

CMD sudo -v ; curl https://rclone.org/install.sh | sudo bash

ENTRYPOINT python /content/inference.py