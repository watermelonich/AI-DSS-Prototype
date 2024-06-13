// File: main_tf/variables.tf

// Define the region variable
variable "region" {
  description = "The AWS region to create resources in"
  type        = string
  default     = "us-west-2"
}

// Define the VPC CIDR block
variable "vpc_cidr" {
  description = "The CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

// Define the public subnet CIDRs
variable "public_subnet_cidrs" {
  description = "List of public subnet CIDRs"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

// Define the private subnet CIDRs
variable "private_subnet_cidrs" {
  description = "List of private subnet CIDRs"
  type        = list(string)
  default     = ["10.0.3.0/24", "10.0.4.0/24"]
}

// Define the bucket name variable
variable "bucket_name" {
  description = "The name of the S3 bucket"
  type        = string
  default     = "my-unique-bucket-name"
}

// Define the instance type variable
variable "instance_type" {
  description = "The type of EC2 instance"
  type        = string
  default     = "t2.micro"
}

// Define the AMI ID variable
variable "ami_id" {
  description = "The AMI ID for the EC2 instance"
  type        = string
  default     = "ami-0c55b159cbfafe1f0"
}

// Define the key name variable
variable "key_name" {
  description = "The name of the SSH key pair"
  type        = string
  default     = "my-key-pair"
}
