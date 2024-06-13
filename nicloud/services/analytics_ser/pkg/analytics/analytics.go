// services/analytics_ser/pkg/analytics/analytics.go
package analytics

import (
	"encoding/json"
	"net/http"
)

// AnalyticsData represents the structure of analytics data
type AnalyticsData struct {
	ID    string `json:"id"`
	Value int    `json:"value"`
}

// In-memory store for analytics data
var analyticsStore = make(map[string]AnalyticsData)

// AddAnalytics handles the addition of new analytics data
func AddAnalytics(w http.ResponseWriter, r *http.Request) {
	var data AnalyticsData
	if err := json.NewDecoder(r.Body).Decode(&data); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	analyticsStore[data.ID] = data
	w.WriteHeader(http.StatusCreated)
}

// GetAnalytics handles retrieving all analytics data
func GetAnalytics(w http.ResponseWriter, r *http.Request) {
	data := make([]AnalyticsData, 0, len(analyticsStore))
	for _, value := range analyticsStore {
		data = append(data, value)
	}
	if err := json.NewEncoder(w).Encode(data); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusOK)
}
