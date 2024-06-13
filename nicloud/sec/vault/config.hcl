# sec/vault/config.hcl

# Storage backend configuration
storage "file" {
  path = "/vault/data"
}

# HTTP listener
listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_disable = 1
}

# Disable mlock (avoid memory lock failure in containers)
disable_mlock = true

# Define the UI setting
ui = true

# Enable telemetry
telemetry {
  statsite_address = "127.0.0.1:8125"
  disable_hostname = true
}
