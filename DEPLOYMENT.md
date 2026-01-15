# Server Deployment Guide

This guide explains how to deploy the Trending Topics API and Worker services on a Linux server using systemd.

## 📋 Overview

The application consists of two separate services:

1. **API Server** (`api_server.py`) - Flask API with Gunicorn
2. **Worker Service** (`worker.py`) - RabbitMQ consumer for background job processing

**⚠️ IMPORTANT: Do NOT use PM2 for Python services with RabbitMQ consumers. Use systemd instead.**

## 🏗️ Architecture

```
┌─────────────────┐         ┌──────────────────┐
│   API Server    │         │  Worker Service  │
│  (Gunicorn)     │         │  (RabbitMQ)      │
│  Port: 5001     │         │  Background Jobs │
└────────┬────────┘         └─────────┬────────┘
         │                            │
         └────────────┬───────────────┘
                      │
         ┌────────────┴───────────────┐
         │                            │
    ┌────▼────┐                  ┌────▼────┐
    │RabbitMQ │                  │OpenSearch│
    └─────────┘                  └──────────┘
```

## 📦 Prerequisites

- Linux server (Ubuntu 20.04+ / Debian 11+ / CentOS 8+)
- Python 3.8+
- RabbitMQ server installed and running
- OpenSearch/Elasticsearch accessible
- Root or sudo access

## 🚀 Installation Steps

### 1. Prepare Server Environment

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3 python3-pip python3-venv git

# Install RabbitMQ (if not already installed)
sudo apt install -y rabbitmq-server
sudo systemctl enable rabbitmq-server
sudo systemctl start rabbitmq-server
```

### 2. Create Application User

```bash
# Create dedicated user for the application
sudo useradd -r -s /bin/false -d /opt/trending-topics www-data
# Or use existing www-data user
```

### 3. Deploy Application Code

```bash
# Create application directory
sudo mkdir -p /opt/trending-topics
sudo chown www-data:www-data /opt/trending-topics

# Clone or copy your code
cd /opt/trending-topics
sudo -u www-data git clone <your-repo-url> .
# OR copy files manually
```

### 4. Setup Python Virtual Environment

```bash
cd /opt/trending-topics

# Create virtual environment
sudo -u www-data python3 -m venv venv

# Activate and install dependencies
sudo -u www-data ./venv/bin/pip install --upgrade pip
sudo -u www-data ./venv/bin/pip install -r requirements.txt
sudo -u www-data ./venv/bin/pip install gunicorn
```

### 5. Configure Environment Variables

```bash
# Create .env file
sudo -u www-data nano /opt/trending-topics/.env
```

Add your configuration:

```env
# OpenSearch Configuration
OPENSEARCH_NODE=http://localhost:9200
OPENSEARCH_USERNAME=admin
OPENSEARCH_PASSWORD=admin
OPENSEARCH_INDEX=user-input-posts
TRENDING_INDEX=trending-topics

# DeepSeek API
DEEPSEEK_API_KEY=your-api-key-here
DEEPSEEK_MODEL=deepseek-chat

# RabbitMQ
RABBITMQ_URL=amqp://guest:guest@localhost:5672/

# API Server
API_HOST=0.0.0.0
API_PORT=5001

# Processing Configuration
BATCH_SIZE=100
MAX_WORKERS=5
MIN_CLUSTER_SIZE=5
HDBSCAN_MIN_CLUSTER_SIZE=5
HDBSCAN_MIN_SAMPLES=3
PCA_TARGET_DIM=100
EMBEDDING_BATCH_SIZE=32
```

### 6. Create Log Directory

```bash
sudo mkdir -p /var/log/trending-topics
sudo chown www-data:www-data /var/log/trending-topics
```

### 7. Install Systemd Service Files

```bash
# Copy service files
sudo cp systemd/trending-topics-api.service /etc/systemd/system/
sudo cp systemd/trending-topics-worker.service /etc/systemd/system/

# Update paths in service files if needed
sudo nano /etc/systemd/system/trending-topics-api.service
sudo nano /etc/systemd/system/trending-topics-worker.service

# Reload systemd
sudo systemctl daemon-reload
```

### 8. Start Services

```bash
# Enable services to start on boot
sudo systemctl enable trending-topics-api
sudo systemctl enable trending-topics-worker

# Start services
sudo systemctl start trending-topics-api
sudo systemctl start trending-topics-worker

# Check status
sudo systemctl status trending-topics-api
sudo systemctl status trending-topics-worker
```

## 🔍 Service Management

### Check Service Status

```bash
# API Server
sudo systemctl status trending-topics-api

# Worker Service
sudo systemctl status trending-topics-worker
```

### View Logs

```bash
# API Server logs
sudo journalctl -u trending-topics-api -f
sudo tail -f /var/log/trending-topics/api-access.log
sudo tail -f /var/log/trending-topics/api-error.log

# Worker logs
sudo journalctl -u trending-topics-worker -f
sudo tail -f /var/log/trending-topics/worker.log
sudo tail -f /var/log/trending-topics/worker-error.log
```

### Restart Services

```bash
# Restart API
sudo systemctl restart trending-topics-api

# Restart Worker
sudo systemctl restart trending-topics-worker

# Restart both
sudo systemctl restart trending-topics-api trending-topics-worker
```

### Stop Services

```bash
sudo systemctl stop trending-topics-api
sudo systemctl stop trending-topics-worker
```

## 🔧 Configuration

### Adjusting Worker Resources

Edit `/etc/systemd/system/trending-topics-worker.service`:

```ini
# Increase memory limit for heavy ML processing
MemoryLimit=16G

# Adjust CPU affinity (optional)
CPUAffinity=0-7
```

### Adjusting API Workers

Edit `/etc/systemd/system/trending-topics-api.service`:

```ini
# Change number of Gunicorn workers
ExecStart=/opt/trending-topics/venv/bin/gunicorn \
    --workers 8 \
    ...
```

Or use the gunicorn config file:

```bash
ExecStart=/opt/trending-topics/venv/bin/gunicorn \
    --config /opt/trending-topics/gunicorn_config.py \
    api_server:app
```

## 🔒 Security Best Practices

1. **Firewall Configuration**
   ```bash
   # Allow API port
   sudo ufw allow 5001/tcp
   ```

2. **SSL/TLS (Production)**
   - Use Nginx as reverse proxy with SSL
   - Configure Gunicorn to bind to localhost only
   - Use Let's Encrypt for certificates

3. **File Permissions**
   ```bash
   # Secure .env file
   sudo chmod 600 /opt/trending-topics/.env
   sudo chown www-data:www-data /opt/trending-topics/.env
   ```

4. **User Isolation**
   - Services run as `www-data` user (non-root)
   - PrivateTmp enabled for isolation
   - ProtectSystem=strict prevents file system access

## 🌐 Nginx Reverse Proxy (Recommended)

For production, use Nginx as a reverse proxy:

```nginx
# /etc/nginx/sites-available/trending-topics
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts for long-running requests
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
```

Enable and restart:
```bash
sudo ln -s /etc/nginx/sites-available/trending-topics /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## 🐛 Troubleshooting

### API Server Not Starting

```bash
# Check logs
sudo journalctl -u trending-topics-api -n 50

# Test Gunicorn manually
cd /opt/trending-topics
sudo -u www-data ./venv/bin/gunicorn --bind 0.0.0.0:5001 api_server:app
```

### Worker Not Consuming Messages

```bash
# Check RabbitMQ connection
sudo rabbitmqctl status

# Check worker logs
sudo journalctl -u trending-topics-worker -n 50

# Test worker manually
cd /opt/trending-topics
sudo -u www-data ./venv/bin/python worker.py
```

### RabbitMQ Connection Issues

```bash
# Check RabbitMQ is running
sudo systemctl status rabbitmq-server

# Check queue exists
sudo rabbitmqctl list_queues

# Check connections
sudo rabbitmqctl list_connections
```

### Permission Issues

```bash
# Fix ownership
sudo chown -R www-data:www-data /opt/trending-topics
sudo chown -R www-data:www-data /var/log/trending-topics
```

## 📊 Monitoring

### Health Check Endpoint

```bash
curl http://localhost:5001/health
```

### Service Health

```bash
# Check if services are running
systemctl is-active trending-topics-api
systemctl is-active trending-topics-worker

# Check service uptime
systemctl show trending-topics-api --property=ActiveEnterTimestamp
```

## 🔄 Updates and Deployment

### Update Application Code

```bash
cd /opt/trending-topics
sudo -u www-data git pull

# Restart services
sudo systemctl restart trending-topics-api trending-topics-worker
```

### Update Dependencies

```bash
cd /opt/trending-topics
sudo -u www-data ./venv/bin/pip install -r requirements.txt --upgrade

# Restart services
sudo systemctl restart trending-topics-api trending-topics-worker
```

## ✅ Verification Checklist

- [ ] Both services are enabled and running
- [ ] API responds to `/health` endpoint
- [ ] Worker can connect to RabbitMQ
- [ ] Logs are being written correctly
- [ ] Services restart automatically on reboot
- [ ] Firewall rules are configured
- [ ] Environment variables are set correctly
- [ ] File permissions are secure

## 🆘 Support

If you encounter issues:

1. Check service status: `sudo systemctl status <service-name>`
2. Check logs: `sudo journalctl -u <service-name> -f`
3. Verify configuration files
4. Test components manually
5. Check RabbitMQ and OpenSearch connectivity

## 📝 Notes

- **Never use PM2 for Python services with RabbitMQ consumers**
- Worker service must run as a separate process (not a thread)
- API and Worker can run on different servers if needed
- Use systemd for reliable service management on Linux servers
