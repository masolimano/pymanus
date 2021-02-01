from plots import AnchoredBeam
import matplotlib.pyplot as plt
import numpy as np
import imageio

img = imageio.imread('/home/manuel/arcos/notebooks/plots/sgasj1226_rgb_linear.png')
fig, ax = plt.subplots()
ax.imshow(img)
beam = AnchoredBeam(bmaj=400, bmin=300, angle=-57, transform=ax.transData,
                    loc='lower left', borderpad=0.5, linestyle='solid',
                    edgecolor='white', linewidth=3)
ax.add_artist(beam)

plt.show()
