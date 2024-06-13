// File: main_tf/modules/storage.tf

// Create an S3 bucket
resource "aws_s3_bucket" "example" {
  bucket = var.bucket_name
  acl    = "private"

  tags = {
    Name        = var.bucket_name
    Environment = "Dev"
  }
}

// Create EBS volume
resource "aws_ebs_volume" "example" {
  availability_zone = element(aws_subnet.public.*.availability_zone, 0)
  size              = 10

  tags = {
    Name = "example-ebs-volume"
  }
}

// Attach EBS volume to EC2 instance
resource "aws_volume_attachment" "ebs_attachment" {
  device_name = "/dev/sdh"
  volume_id   = aws_ebs_volume.example.id
  instance_id = aws_instance.web.id
}
