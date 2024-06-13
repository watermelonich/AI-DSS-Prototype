// File: main_tf/envs/dev.tfvars

region         = "us-west-2"
vpc_cidr       = "10.0.0.0/16"
public_subnet_cidrs = ["10.0.1.0/24", "10.0.2.0/24"]
private_subnet_cidrs = ["10.0.3.0/24", "10.0.4.0/24"]
bucket_name    = "dev-unique-bucket-name"
instance_type  = "t2.micro"
ami_id         = "ami-0c55b159cbfafe1f0"
key_name       = "dev-key-pair"
