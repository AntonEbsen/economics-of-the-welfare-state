import os

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke

WELFARE_REGIMES = {
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

REGIME_COLORS = {
    'Social Democratic': '#1a9e87',
    'Conservative':      '#e07b39',
    'Liberal':           '#4e78c4',
    'Mediterranean':     '#c9547b',
    'Post-Communist':    '#7ab648',
}

# Manual centroid nudges (lon°, lat°) to reduce label collisions in dense areas
_LABEL_NUDGE = {
    'IRL': (-1.5,  0.0), 'GBR': ( 0.5, -1.0),
    'BEL': (-0.8,  0.8), 'NLD': ( 1.2,  0.6), 'LUX': ( 1.2, -0.4),
    'CHE': ( 0.0, -0.6), 'AUT': ( 1.2,  0.2),
    'EST': ( 0.6,  0.4), 'LVA': ( 0.3, -0.3), 'LTU': (-0.2, -0.6),
    'SVK': ( 0.8,  0.4), 'SVN': (-0.5, -0.6), 'HUN': ( 0.0,  0.5),
    'PRT': (-0.6,  0.0), 'ISL': ( 0.0, -0.5),
    'NZL': ( 3.0,  0.0),
}


def _load_world() -> gpd.GeoDataFrame:
    try:
        path = gpd.datasets.get_path('naturalearth_lowres')
        world = gpd.read_file(path)
    except Exception:
        print("Bundled dataset not found — downloading from Natural Earth...")
        url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        world = gpd.read_file(url)

    if 'iso_a3' not in world.columns and 'ISO_A3' in world.columns:
        world = world.rename(columns={'ISO_A3': 'iso_a3'})

    return world


def create_welfare_regimes_map(out_dir=None):
    world = _load_world()
    world['regime'] = world['iso_a3'].map(WELFARE_REGIMES)

    fig, ax = plt.subplots(figsize=(16, 8), facecolor='#eef4f8')
    ax.set_facecolor('#c4d8eb')  # ocean

    # Non-sample countries
    world.plot(ax=ax, color='#e4e0d8', edgecolor='#ccc8bf', linewidth=0.3)

    # Sample countries coloured by regime
    world_regimes = world.dropna(subset=['regime'])
    for regime, color in REGIME_COLORS.items():
        subset = world_regimes[world_regimes['regime'] == regime]
        if not subset.empty:
            subset.plot(ax=ax, color=color, edgecolor='white', linewidth=0.7, alpha=0.93)

    # Country labels — larger font for big countries, smaller for tiny ones
    for _, row in world_regimes.iterrows():
        iso = row['iso_a3']
        centroid = row.geometry.centroid
        dx, dy = _LABEL_NUDGE.get(iso, (0.0, 0.0))
        fontsize = 6.5 if row.geometry.area > 5.0 else 5.0
        ax.text(
            centroid.x + dx, centroid.y + dy, iso,
            fontsize=fontsize, ha='center', va='center',
            color='white', fontweight='bold',
            path_effects=[withStroke(linewidth=1.8, foreground='#111111')],
            zorder=5,
        )

    # Bounds — exclude Antarctica
    ax.set_xlim(-170, 180)
    ax.set_ylim(-58, 85)
    ax.axis('off')

    # Legend
    patches = [mpatches.Patch(facecolor=c, edgecolor='white', linewidth=0.5, label=r)
               for r, c in REGIME_COLORS.items()]
    legend = ax.legend(
        handles=patches,
        title='Welfare State Regime',
        title_fontsize=8.5,
        fontsize=8,
        loc='lower left',
        frameon=True,
        framealpha=0.95,
        edgecolor='#aaaaaa',
        facecolor='#fafafa',
    )
    legend.get_title().set_fontweight('bold')

    n = len(WELFARE_REGIMES)
    ax.set_title(
        f'Welfare State Regimes — {n}-Country OECD Panel',
        fontsize=13, fontweight='bold', pad=12, color='#1a1a1a',
    )
    fig.text(
        0.99, 0.005,
        'Classification based on Esping-Andersen (1990) and subsequent literature',
        ha='right', fontsize=7, color='#888888', style='italic',
    )

    plt.tight_layout()
    plt.show()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, 'welfare_regimes_map.pdf')
    else:
        output_path = 'welfare_regimes_map.pdf'

    fig.savefig(output_path, bbox_inches='tight', dpi=300, format='pdf')
    print(f"Map saved to {output_path}")
    plt.close()

    return output_path


if __name__ == "__main__":
    create_welfare_regimes_map()
