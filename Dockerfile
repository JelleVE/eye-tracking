FROM ubuntu:18.04
RUN apt update -y && apt upgrade -y

# Install essentials
RUN apt-get install -y git build-essential cmake curl wget
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Fix ascii to utf8 => https://webkul.com/blog/setup-locale-python3/
RUN apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8


# Setup app
RUN mkdir /app
RUN rm -rf ~/.cache/pip
WORKDIR /app
ADD  . /app/

EXPOSE 5000

# Setup dependencies
WORKDIR /app/
RUN apt-get install -y python3-pip
RUN python3 --version
RUN pip3 install --upgrade pip
RUN pip3 install -r build/requirements.txt

# Setup nginx
# RUN apt-get install -y nginx
# ADD build/nginx.conf /etc/nginx/nginx.conf

# Setup uwsgi
# RUN apt-get install -y uwsgi
# RUN pip3 install uwsgi

# Run
# RUN chmod +x /app/build/run_server.sh

# CMD ["/app/build/run_server.sh"]

RUN chmod +x /app/run.sh
CMD ["/app/run.sh"]