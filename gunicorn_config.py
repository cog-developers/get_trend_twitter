"""
Gunicorn configuration for Trending Topics API Server
"""

import multiprocessing
import os

# Server socket
bind = os.getenv('GUNICORN_BIND', '0.0.0.0:5001')
backlog = 2048

# Worker processes
workers = int(os.getenv('GUNICORN_WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = os.getenv('GUNICORN_WORKER_CLASS', 'sync')
worker_connections = 1000
timeout = 300
keepalive = 5

# Logging
accesslog = os.getenv('GUNICORN_ACCESS_LOG', '-')
errorlog = os.getenv('GUNICORN_ERROR_LOG', '-')
loglevel = os.getenv('GUNICORN_LOG_LEVEL', 'info')
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'trending-topics-api'

# Server mechanics
daemon = False
pidfile = os.getenv('GUNICORN_PIDFILE', '/var/run/trending-topics-api.pid')
umask = 0
user = os.getenv('GUNICORN_USER', None)
group = os.getenv('GUNICORN_GROUP', None)
tmp_upload_dir = None

# SSL (if needed)
# keyfile = None
# certfile = None

# Performance tuning
max_requests = 1000
max_requests_jitter = 50
preload_app = False

# Graceful timeout
graceful_timeout = 30
