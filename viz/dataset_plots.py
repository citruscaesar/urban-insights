from matplotlib import pyplot as plt
from numpy import sqrt, around, ceil, random

def plot_segmentation_samples(ds, n: int = 16):
    assert n >= 2, "invalid n, n must be at least 2"
    nrows = int(around(sqrt(n)))
    ncols = int(ceil(n / nrows))
    
    fig, axes = plt.subplots(nrows, ncols, figsize = (12, 12))
    fig.suptitle(f"{ds.NAME}_{ds.TASK}", fontsize = 10)
    for idx, ax in enumerate(axes.ravel()):
        image, mask, _ = ds[random.randint(0, len(ds))]
        if idx < n: 
            ax.imshow(image.permute(1,2,0))
            ax.imshow(mask[1], cmap = "Reds", alpha = 0.5)
        ax.axis("off")
    plt.tight_layout()
