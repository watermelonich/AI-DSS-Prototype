// File: main_tf/envs/prod.tfvars

region         = "us-east-1"
vpc_cidr       = "10.1.0.0/16"
public_subnet_cidrs = ["10.1.1.0/24", "10.1.2.0/24"]
private_subnet_cidrs = ["10.1.3.0/24", "10.1.4.0/24"]
bucket_name    = "prod-unique-bucket-name"
instance_type  = "t3.medium"
ami_id         = "ami-0c55b159cbfafe1f0"
key_name       = "prod-key-pair"
