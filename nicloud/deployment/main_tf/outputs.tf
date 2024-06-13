// File: main_tf/outputs.tf

// Output the VPC ID
output "vpc_id" {
  description = "The ID of the VPC"
  value       = aws_vpc.main.id
}

// Output the public subnet IDs
output "public_subnet_ids" {
  description = "The IDs of the public subnets"
  value       = aws_subnet.public.*.id
}

// Output the private subnet IDs
output "private_subnet_ids" {
  description = "The IDs of the private subnets"
  value       = aws_subnet.private.*.id
}

// Output the S3 bucket name
output "bucket_name" {
  description = "The name of the S3 bucket"
  value       = aws_s3_bucket.example.bucket
}

// Output the EC2 instance ID
output "instance_id" {
  description = "The ID of the EC2 instance"
  value       = aws_instance.web.id
}
