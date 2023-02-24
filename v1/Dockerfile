FROM  ubuntu:20.04 

WORKDIR /content

RUN apt-get update && apt-get install  git python3 python3-pip wget -y && \
	apt-get install python-is-python3

RUN pip install --quiet --upgrade diffusers transformers accelerate mediapy triton scipy ftfy spacy==3.4.4 && \
	pip install -q xformers==0.0.16rc425

RUN pip install runpod 

RUN pip install pillow

RUN pip install boto3

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
	apt-get install git-lfs && \
	git lfs install

RUN git clone https://huggingface.co/SfinOe/dreamlike_2.0 model

COPY inference.py /content/inference.py

CMD sudo -v ; curl https://rclone.org/install.sh | sudo bash

ENTRYPOINT python /content/inference.py