#!/bin/bash

# File: automation/provision.sh

# Set variables
ANSIBLE_DIR="/path/to/ansible"
INVENTORY_FILE="$ANSIBLE_DIR/inventory"
PLAYBOOK_FILE="$ANSIBLE_DIR/playbook.yml"
ENVIRONMENT=$1

if [ -z "$ENVIRONMENT" ]; then
  echo "Usage: $0 <environment>"
  exit 1
fi

# Provision infrastructure using Ansible
ansible-playbook -i "$INVENTORY_FILE/$ENVIRONMENT" "$PLAYBOOK_FILE"

# Print success message
echo "Provisioning for $ENVIRONMENT completed successfully."
