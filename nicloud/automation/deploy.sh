#!/bin/bash

# File: automation/deploy.sh

# Set variables
ENVIRONMENT=$1
TERRAFORM_DIR="/path/to/terraform"

if [ -z "$ENVIRONMENT" ]; then
  echo "Usage: $0 <environment>"
  exit 1
fi

# Change to Terraform directory
cd "$TERRAFORM_DIR"

# Initialize Terraform
terraform init

# Apply the Terraform configuration
terraform apply -var-file="main_tf/envs/${ENVIRONMENT}.tfvars" -auto-approve

# Print success message
echo "Deployment to $ENVIRONMENT completed successfully."
