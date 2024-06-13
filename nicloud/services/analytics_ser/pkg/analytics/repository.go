// services/analytics_ser/pkg/analytics/repository.go
package analytics

// Repository defines the methods for interacting with the data store
type Repository interface {
	Add(data AnalyticsData) error
	GetAll() ([]AnalyticsData, error)
}

// InMemoryRepository implements Repository interface with in-memory store
type InMemoryRepository struct {
	store map[string]AnalyticsData
}

// NewInMemoryRepository creates a new in-memory repository
func NewInMemoryRepository() *InMemoryRepository {
	return &InMemoryRepository{
		store: make(map[string]AnalyticsData),
	}
}

// Add adds new analytics data to the store
func (repo *InMemoryRepository) Add(data AnalyticsData) error {
	repo.store[data.ID] = data
	return nil
}

// GetAll retrieves all analytics data from the store
func (repo *InMemoryRepository) GetAll() ([]AnalyticsData, error) {
	data := make([]AnalyticsData, 0, len(repo.store))
	for _, value := range repo.store {
		data = append(data, value)
	}
	return data, nil
}
