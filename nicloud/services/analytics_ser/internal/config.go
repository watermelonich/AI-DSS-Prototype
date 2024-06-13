// services/analytics_ser/pkg/analytics/config.go
package analytics

import (
	"encoding/json"
	"log"
	"os"
)

// Config represents configuration settings
type Config struct {
	ServerPort string `json:"server_port"`
}

// LoadConfig loads configuration from a file
func LoadConfig(file string) (*Config, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var config Config
	if err := json.NewDecoder(f).Decode(&config); err != nil {
		return nil, err
	}

	return &config, nil
}

// GetConfig loads configuration from default config file
func GetConfig() *Config {
	config, err := LoadConfig("config.json")
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}
	return config
}
