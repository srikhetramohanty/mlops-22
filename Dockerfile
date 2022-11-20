# #To import pre-defined ubuntu image from docker hub
# #FROM ubuntu:latest 
# FROM python:3.8.1

# RUN python3 -m pip install --upgrade pip
# #COPY ./* /exp/
# COPY ./* /exp/
# #COPY ./requirements.txt /exp/requirements.txt
# #RUN pip install --no-cache-dir -r /exp/requirements.txt
# RUN pip install -r /exp/requirements.txt
# WORKDIR /exp

# #CMD ["python3","./plot_digits_classification.py"]
# CMD ["python3","./app.py"]

###########################################################
## Dockerfile
# start by pulling the python image
FROM python:3.8.15-slim-buster

RUN python3 -m pip install --upgrade pip

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

# switch working directory
WORKDIR /app

# install the dependencies and packages in the requirements file
RUN pip3 install -r requirements.txt
EXPOSE 5000
# copy every content from the local file to the image
COPY . /app

# configure the container to run in an executed manner
ENTRYPOINT ["sh"]

CMD ["./app_run.sh"]
# ENTRYPOINT ["python3"]

# CMD ["./app.py"]