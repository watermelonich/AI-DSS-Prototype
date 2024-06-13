// services/analytics_ser/main.go
package main

import (
	"encoding/json"
	"log"
	"net/http"
	"services/analytics_ser/pkg/analytics"

	"github.com/gorilla/mux"
)

var service *analytics.Service

func main() {
	config := analytics.GetConfig()

	repo := analytics.NewInMemoryRepository()
	service = analytics.NewService(repo)

	r := mux.NewRouter()

	// Define routes
	r.HandleFunc("/analytics", addAnalyticsHandler).Methods("POST")
	r.HandleFunc("/analytics", getAnalyticsHandler).Methods("GET")

	// Start the server
	log.Println("Starting server on", config.ServerPort)
	if err := http.ListenAndServe(config.ServerPort, r); err != nil {
		log.Fatalf("Could not start server: %s\n", err)
	}
}

func addAnalyticsHandler(w http.ResponseWriter, r *http.Request) {
	var data analytics.AnalyticsData
	if err := json.NewDecoder(r.Body).Decode(&data); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := service.AddAnalytics(data); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusCreated)
}

func getAnalyticsHandler(w http.ResponseWriter, r *http.Request) {
	data, err := service.GetAnalytics()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if err := json.NewEncoder(w).Encode(data); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusOK)
}
