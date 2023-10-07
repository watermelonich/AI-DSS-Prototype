# Experimental file

import plotly.express as px

# Create a DataFrame with sample data (replace this with your actual data)
data = {
    "Country": ["USA", "Canada", "UK", "Germany", "France"],
    "Percentage": [30, 20, 15, 25, 18]
}

# Create the world map
fig = px.choropleth(data,
                    locations="Country",
                    locationmode="country names",
                    color="Percentage",
                    hover_name="Country",
                    color_continuous_scale=px.colors.sequential.Plasma,
                    title="Percentage Data by Country")

# Update hover template to display percentage
fig.update_traces(hovertemplate='%{hovertext}: %{z}%')

# Show the chart
fig.show()
