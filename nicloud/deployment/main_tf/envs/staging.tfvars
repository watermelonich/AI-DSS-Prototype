// File: main_tf/envs/staging.tfvars

region         = "us-west-1"
vpc_cidr       = "10.2.0.0/16"
public_subnet_cidrs = ["10.2.1.0/24", "10.2.2.0/24"]
private_subnet_cidrs = ["10.2.3.0/24", "10.2.4.0/24"]
bucket_name    = "staging-unique-bucket-name"
instance_type  = "t2.medium"
ami_id         = "ami-0c55b159cbfafe1f0"
key_name       = "staging-key-pair"
