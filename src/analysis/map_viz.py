import geopandas as gpd
import matplotlib.pyplot as plt
import os

def create_welfare_regimes_map(out_dir=None):
    # Define welfare regime mapping (use ISO_A3 codes for robustness)
    welfare_regimes = {
        'DNK': 'Social Democratic', 'FIN': 'Social Democratic', 
        'NOR': 'Social Democratic', 'SWE': 'Social Democratic', 
        'ISL': 'Social Democratic',
        'AUT': 'Conservative', 'BEL': 'Conservative', 'FRA': 'Conservative',
        'DEU': 'Conservative', 'LUX': 'Conservative', 'NLD': 'Conservative',
        'CHE': 'Conservative',
        'AUS': 'Liberal', 'CAN': 'Liberal', 'IRL': 'Liberal', 'JPN': 'Liberal',
        'NZL': 'Liberal', 'GBR': 'Liberal', 'USA': 'Liberal',
        'GRC': 'Mediterranean', 'ITA': 'Mediterranean', 
        'PRT': 'Mediterranean', 'ESP': 'Mediterranean',
        'BGR': 'Post-Communist', 'CZE': 'Post-Communist', 'EST': 'Post-Communist',
        'HUN': 'Post-Communist', 'LVA': 'Post-Communist', 'LTU': 'Post-Communist',
        'POL': 'Post-Communist', 'SVK': 'Post-Communist', 'SVN': 'Post-Communist',
    }

    # Load Natural Earth countries shapefile
    try:
        # Try legacy path first
        path = gpd.datasets.get_path('naturalearth_lowres')
        world = gpd.read_file(path)
    except (AttributeError, ValueError, Exception):
        # Fallback: Load from Natural Earth public CDN
        print("Bundled dataset not found or deprecated. Downloading from Natural Earth...")
        url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        world = gpd.read_file(url)

    # Normalize ISO codes
    if 'iso_a3' not in world.columns and 'ISO_A3' in world.columns:
        world['iso_a3'] = world['ISO_A3']
    
    world['regime'] = world['iso_a3'].map(welfare_regimes)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    world.plot(ax=ax, color='#f0f0f0', edgecolor='white', linewidth=0.3)
    
    regime_colors = {
        'Social Democratic': '#66c2a5',
        'Conservative': '#fc8d62',
        'Liberal': '#8da0cb',
        'Mediterranean': '#e78ac3',
        'Post-Communist': '#a6d854'
    }
    
    world_regimes = world.dropna(subset=['regime'])
    world_regimes.plot(
        ax=ax, 
        categorical=True,
        legend=True,
        legend_kwds={'loc': 'lower left', 'bbox_to_anchor': (1, 0), 'frameon': False},
        color=[regime_colors[r] for r in world_regimes['regime']],
        edgecolor='white', 
        linewidth=0.5
    )

    ax.set_xlim(-15, 35)
    ax.set_ylim(35, 72)
    ax.axis('off')
    plt.title('Welfare State Regimes (OECD Sample)', fontsize=16, pad=20)

    # Handle output path
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, 'welfare_regimes_map.pdf')
    else:
        output_path = 'welfare_regimes_map.pdf'

    plt.savefig(output_path, bbox_inches='tight', dpi=300, format='pdf')
    print(f"Map saved to {output_path}")
    plt.close()
    
    return output_path

if __name__ == "__main__":
    create_welfare_regimes_map()
