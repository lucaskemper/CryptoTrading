#!/bin/bash

# Database backup script for crypto trading bot

set -e

# Configuration
DB_HOST="postgres"
DB_PORT="5432"
DB_NAME="tradingbot"
DB_USER="tradingbot"
DB_PASSWORD="password"
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory if it doesn't exist
mkdir -p $BACKUP_DIR

# Database backup
echo "Creating database backup..."
pg_dump -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME > $BACKUP_DIR/tradingbot_db_$DATE.sql

# Compress the backup
gzip $BACKUP_DIR/tradingbot_db_$DATE.sql

# Keep only the last 7 days of backups
find $BACKUP_DIR -name "tradingbot_db_*.sql.gz" -mtime +7 -delete

echo "Backup completed: tradingbot_db_$DATE.sql.gz" 