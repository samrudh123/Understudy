#!/bin/sh
# Create a self-signed certificate once (mount /etc/nginx/certs to bring your own).
set -e
mkdir -p /etc/nginx/certs
if [ ! -f /etc/nginx/certs/understudy.crt ]; then
  openssl req -x509 -nodes -newkey rsa:2048 -days 365 \
    -subj "/CN=localhost" -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" \
    -keyout /etc/nginx/certs/understudy.key -out /etc/nginx/certs/understudy.crt 2>/dev/null
fi
