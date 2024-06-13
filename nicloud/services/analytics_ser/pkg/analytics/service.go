// services/analytics_ser/pkg/analytics/service.go
package analytics

// Service provides business logic for analytics
type Service struct {
	repo Repository
}

// NewService creates a new analytics service
func NewService(repo Repository) *Service {
	return &Service{repo: repo}
}

// AddAnalytics adds new analytics data
func (s *Service) AddAnalytics(data AnalyticsData) error {
	return s.repo.Add(data)
}

// GetAnalytics retrieves all analytics data
func (s *Service) GetAnalytics() ([]AnalyticsData, error) {
	return s.repo.GetAll()
}
