# import matplotlib.pyplot as plt
# from models.mapplot.earth_mapplot import plot_earth_map
# import os

# def generate_earth_map_plot():
#     plt.figure()
#     plot_earth_map()
#     plot_path = os.path.join(os.path.dirname(__file__), 'gui', 'plots', 'earth_map.png')
#     plt.savefig(plot_path)
#     return plot_path

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
