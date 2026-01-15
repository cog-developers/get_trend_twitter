#!/bin/bash

# Deployment script for Trending Topics API and Worker services
# Run this script on the server with sudo privileges
# Usage: sudo bash deploy.sh

set -e  # Exit on error

APP_DIR="/var/www/get_trend_twitter"
SERVICE_USER="www-data"
LOG_DIR="/var/log/trending-topics"

echo "=========================================="
echo "Trending Topics Service Deployment"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ Please run as root or with sudo"
    exit 1
fi

# Step 1: Create log directory
echo "📁 Creating log directory..."
mkdir -p "$LOG_DIR"
chown "$SERVICE_USER:$SERVICE_USER" "$LOG_DIR"
chmod 755 "$LOG_DIR"
echo "✅ Log directory created: $LOG_DIR"

# Step 2: Verify application directory exists
echo ""
echo "📁 Checking application directory..."
if [ ! -d "$APP_DIR" ]; then
    echo "❌ Application directory not found: $APP_DIR"
    echo "   Please ensure the application is deployed to $APP_DIR"
    exit 1
fi
echo "✅ Application directory found: $APP_DIR"

# Step 3: Set proper ownership
echo ""
echo "🔐 Setting file permissions..."
chown -R "$SERVICE_USER:$SERVICE_USER" "$APP_DIR"
if [ -f "$APP_DIR/.env" ]; then
    chmod 600 "$APP_DIR/.env"
    echo "✅ .env file permissions set"
fi
echo "✅ File permissions configured"

# Step 4: Copy systemd service files
echo ""
echo "📋 Installing systemd service files..."
if [ ! -f "$APP_DIR/systemd/trending-topics-api.service" ]; then
    echo "❌ Service file not found: $APP_DIR/systemd/trending-topics-api.service"
    exit 1
fi
if [ ! -f "$APP_DIR/systemd/trending-topics-worker.service" ]; then
    echo "❌ Service file not found: $APP_DIR/systemd/trending-topics-worker.service"
    exit 1
fi

cp "$APP_DIR/systemd/trending-topics-api.service" /etc/systemd/system/
cp "$APP_DIR/systemd/trending-topics-worker.service" /etc/systemd/system/
echo "✅ Service files copied to /etc/systemd/system/"

# Step 5: Reload systemd
echo ""
echo "🔄 Reloading systemd daemon..."
systemctl daemon-reload
echo "✅ Systemd daemon reloaded"

# Step 6: Enable services
echo ""
echo "🔧 Enabling services to start on boot..."
systemctl enable trending-topics-api
systemctl enable trending-topics-worker
echo "✅ Services enabled"

# Step 7: Stop existing services (if running)
echo ""
echo "🛑 Stopping existing services (if running)..."
systemctl stop trending-topics-api 2>/dev/null || true
systemctl stop trending-topics-worker 2>/dev/null || true
echo "✅ Services stopped"

# Step 8: Start services
echo ""
echo "🚀 Starting services..."
systemctl start trending-topics-api
sleep 2
systemctl start trending-topics-worker
sleep 2
echo "✅ Services started"

# Step 9: Check service status
echo ""
echo "📊 Service Status:"
echo "=========================================="
systemctl status trending-topics-api --no-pager -l | head -n 10
echo ""
systemctl status trending-topics-worker --no-pager -l | head -n 10

# Step 10: Verify services are running
echo ""
echo "🔍 Verifying services..."
if systemctl is-active --quiet trending-topics-api; then
    echo "✅ API service is running"
else
    echo "❌ API service failed to start"
    echo "   Check logs: journalctl -u trending-topics-api -n 50"
fi

if systemctl is-active --quiet trending-topics-worker; then
    echo "✅ Worker service is running"
else
    echo "❌ Worker service failed to start"
    echo "   Check logs: journalctl -u trending-topics-worker -n 50"
fi

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "Useful commands:"
echo "  Check API status:    sudo systemctl status trending-topics-api"
echo "  Check Worker status: sudo systemctl status trending-topics-worker"
echo "  View API logs:       sudo journalctl -u trending-topics-api -f"
echo "  View Worker logs:    sudo journalctl -u trending-topics-worker -f"
echo "  Restart API:         sudo systemctl restart trending-topics-api"
echo "  Restart Worker:      sudo systemctl restart trending-topics-worker"
echo "  Restart both:        sudo systemctl restart trending-topics-api trending-topics-worker"
echo "  Stop both:           sudo systemctl stop trending-topics-api trending-topics-worker"
echo ""
