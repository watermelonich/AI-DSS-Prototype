// File: main_tf/modules/compute.tf

// Create an EC2 instance
resource "aws_instance" "web" {
  ami           = var.ami_id
  instance_type = var.instance_type
  key_name      = var.key_name
  subnet_id     = element(aws_subnet.public.*.id, 0)

  tags = {
    Name = "web-instance"
  }
}

// Create Auto Scaling Group and Launch Configuration (if needed)
resource "aws_launch_configuration" "app" {
  image_id        = var.ami_id
  instance_type   = var.instance_type
  key_name        = var.key_name
  security_groups = [aws_security_group.web.id]

  lifecycle {
    create_before_destroy = true
  }

  user_data = <<-EOF
              #!/bin/bash
              echo "Hello, World!" > index.html
              nohup busybox httpd -f -p ${var.web_port} &
              EOF

  tags = {
    Name = "app-launch-configuration"
  }
}

resource "aws_autoscaling_group" "app" {
  desired_capacity     = 2
  max_size             = 3
  min_size             = 1
  vpc_zone_identifier  = aws_subnet.public.*.id
  launch_configuration = aws_launch_configuration.app.id

  tags = [
    {
      key                 = "Name"
      value               = "app-instance"
      propagate_at_launch = true
    }
  ]
}
