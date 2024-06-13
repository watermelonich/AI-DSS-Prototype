# sec/vault/policies.hcl

# Define a policy named "default" that grants access to specific paths
path "secret/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Define a policy named "readonly" that grants read-only access to specific paths
path "secret/data/*" {
  capabilities = ["read", "list"]
}

# Define a policy named "analytics" that grants specific access to analytics-related secrets
path "secret/data/analytics/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Define a policy named "auth" that grants specific access to authentication-related secrets
path "secret/data/auth/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Define a policy named "data" that grants specific access to data-related secrets
path "secret/data/data/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Define a policy named "notif" that grants specific access to notification-related secrets
path "secret/data/notif/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Define a policy named "web" that grants specific access to web service-related secrets
path "secret/data/web/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
