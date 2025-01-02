FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

COPY ./requirements.txt /requirements.txt

#Dependencies for open-CV
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r /requirements.txt

WORKDIR /repo/

EXPOSE 8888

CMD ["jupyter", "notebook", "--allow-root", "--ip=0.0.0.0", "--no-browser", "--port=8888"]
