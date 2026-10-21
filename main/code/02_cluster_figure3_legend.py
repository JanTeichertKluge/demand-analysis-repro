"""
Figure 3 of the paper (3d_embedding_txtimg.png) with the legend in the cluster numbering of the similarity variables
similarity_cluster_k (X^sim_k, Tables 8 to 10), which Tables 2, 4 and 9 also use.

The 3D plot of 02_cluster_centroid_products.ipynb numbers the clusters by the k-means run in that notebook. Its clusters
0, 1, 2, 3, 4 are the clusters 0, 3, 4, 2, 1 of similarity_cluster_k (01_1_create_dataset_txt_img.ipynb): all rows of the
products closest to each centroid (collage_cluster_img_<k>.png) have their highest similarity in the mapped cluster.
The figure is redrawn from the plotly output saved in the notebook; only the legend changes (numbering, order and
opaque markers). The static export requires kaleido and Chrome or Chromium.

Run from main/code: python 02_cluster_figure3_legend.py
"""
import json

import plotly.graph_objects as go
import plotly.io as pio

NOTEBOOK = "02_cluster_centroid_products.ipynb"
OUTPUT = "../output/02_cluster_centroid_products/3d_embedding_txtimg.png"
H2S = [0, 3, 4, 2, 1]  # cluster of 02_cluster_centroid_products -> k of similarity_cluster_k

with open(NOTEBOOK, encoding="utf-8") as f:
    nb = json.load(f)
# the single-panel 3D plot of the text + image embeddings
cell = next(
    c
    for c in nb["cells"]
    if c["cell_type"] == "code"
    and "kmeans_txtimg_labels == cluster_id" in "".join(c["source"])
    and "make_subplots(rows=1, cols=1" in "".join(c["source"])
)
saved = next(o["data"]["application/vnd.plotly.v1+json"] for o in cell["outputs"] if "data" in o)
fig = pio.from_json(json.dumps(saved))
assert [t.name for t in fig.data] == [f"Cluster {h}" for h in range(5)]

colors = [t.marker.color for t in fig.data]
for h, trace in enumerate(fig.data):
    trace.name = f"Cluster {H2S[h]}"
    trace.showlegend = False
for k in range(5):
    fig.add_trace(
        go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            marker=dict(size=8, color=colors[H2S.index(k)], opacity=1),
            name=f"Cluster {k}",
        )
    )
fig.write_image(OUTPUT, width=800, height=800)
print(f"wrote {OUTPUT}")
