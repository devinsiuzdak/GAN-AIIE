import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from Generator import Generator

device = torch.device('cpu')

def get_random(count):
    return torch.randn(count, device=device)

G = Generator()
G.load_state_dict(torch.load('gan.pth'))

f, axarr = plt.subplots(2,3, figsize=(16,8))
for i in range(2):
    for j in range(3):
        output = G.forward(get_random(300))
        img = output.detach().numpy()
        img = img.reshape(128, 128, 3)
        axarr[i,j].imshow(img, interpolation='none', cmap='Greys')

plt.show()