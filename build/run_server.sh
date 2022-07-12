#!/bin/bash

service nginx start
uwsgi --ini /app/build/api.ini