// File: main_tf/backend.tf

// Configure the remote backend for state storage
terraform {
  backend "s3" {
    bucket         = "my-terraform-state-bucket"
    key            = "terraform/state"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "terraform-lock"
  }
}
