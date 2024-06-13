#!/bin/bash

# File: automation/backup.sh

# Set variables
BACKUP_DIR="/path/to/backup"
TIMESTAMP=$(date +"%F")
BACKUP_FILE="$BACKUP_DIR/backup-$TIMESTAMP.tar.gz"
SOURCE_DIRS="/path/to/data /path/to/config"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Create the backup
tar -czf "$BACKUP_FILE" $SOURCE_DIRS

# Print success message
echo "Backup completed successfully. Backup file: $BACKUP_FILE"
