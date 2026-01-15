#!/bin/bash

# Quick Commands for Trending Topics Services
# Copy and paste these commands directly into your terminal

# ============================================
# DEPLOYMENT COMMANDS
# ============================================

# 1. Create log directory
sudo mkdir -p /var/log/trending-topics
sudo chown www-data:www-data /var/log/trending-topics

# 2. Set application directory permissions
sudo chown -R www-data:www-data /var/www/get_trend_twitter
sudo chmod 600 /var/www/get_trend_twitter/.env

# 3. Copy systemd service files
sudo cp /var/www/get_trend_twitter/systemd/trending-topics-api.service /etc/systemd/system/
sudo cp /var/www/get_trend_twitter/systemd/trending-topics-worker.service /etc/systemd/system/

# 4. Reload systemd
sudo systemctl daemon-reload

# 5. Enable services (start on boot)
sudo systemctl enable trending-topics-api
sudo systemctl enable trending-topics-worker

# 6. Start services
sudo systemctl start trending-topics-api
sudo systemctl start trending-topics-worker

# ============================================
# STATUS CHECK COMMANDS
# ============================================

# Check API service status
sudo systemctl status trending-topics-api

# Check Worker service status
sudo systemctl status trending-topics-worker

# Check if services are active
sudo systemctl is-active trending-topics-api
sudo systemctl is-active trending-topics-worker

# ============================================
# LOG VIEWING COMMANDS
# ============================================

# View API logs (live)
sudo journalctl -u trending-topics-api -f

# View Worker logs (live)
sudo journalctl -u trending-topics-worker -f

# View last 50 lines of API logs
sudo journalctl -u trending-topics-api -n 50

# View last 50 lines of Worker logs
sudo journalctl -u trending-topics-worker -n 50

# View log files directly
sudo tail -f /var/log/trending-topics/api-access.log
sudo tail -f /var/log/trending-topics/api-error.log
sudo tail -f /var/log/trending-topics/worker.log
sudo tail -f /var/log/trending-topics/worker-error.log

# ============================================
# RESTART COMMANDS
# ============================================

# Restart API service
sudo systemctl restart trending-topics-api

# Restart Worker service
sudo systemctl restart trending-topics-worker

# Restart both services
sudo systemctl restart trending-topics-api trending-topics-worker

# ============================================
# STOP/START COMMANDS
# ============================================

# Stop API service
sudo systemctl stop trending-topics-api

# Stop Worker service
sudo systemctl stop trending-topics-worker

# Stop both services
sudo systemctl stop trending-topics-api trending-topics-worker

# Start API service
sudo systemctl start trending-topics-api

# Start Worker service
sudo systemctl start trending-topics-worker

# Start both services
sudo systemctl start trending-topics-api trending-topics-worker

# ============================================
# DISABLE/ENABLE COMMANDS
# ============================================

# Disable services from starting on boot
sudo systemctl disable trending-topics-api
sudo systemctl disable trending-topics-worker

# Enable services to start on boot
sudo systemctl enable trending-topics-api
sudo systemctl enable trending-topics-worker

# ============================================
# TROUBLESHOOTING COMMANDS
# ============================================

# Check RabbitMQ status
sudo systemctl status rabbitmq-server

# Check RabbitMQ queues
sudo rabbitmqctl list_queues

# Test API health endpoint
curl http://localhost:5001/health

# Test Gunicorn manually (from app directory)
cd /var/www/get_trend_twitter
sudo -u www-data ./venv/bin/gunicorn --bind 0.0.0.0:5001 api_server:app

# Test worker manually (from app directory)
cd /var/www/get_trend_twitter
sudo -u www-data ./venv/bin/python worker.py

# Fix permissions if needed
sudo chown -R www-data:www-data /var/www/get_trend_twitter
sudo chown -R www-data:www-data /var/log/trending-topics
