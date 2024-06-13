import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image

from Discriminator import Discriminiator
from Generator import Generator

device = torch.device('cpu')

def get_random(count):
    return torch.randn(count, device=device)

# center crop to 128x128
def crop(image): # 178 / 218 ( W / H)
    left = (178 / 2) - (128 / 2)
    right = left + 128
    top = (218 / 2) - (128 / 2)
    bottom = top + 128

    return image.crop((left, top, right, bottom))


D = Discriminiator()
D.to(device)
G = Generator()
G.to(device)

directory = 'train'
files = os.listdir(directory)

epochs = 1

for epoch in range(epochs):
    for i in range(100):
        if i % 10 == 0:
            print(i)

        image = Image.open(directory + '/' + files[i])
        image = crop(image)
        a = np.array(image) / 255.0
        a = a.reshape(128 * 128 * 3)

        image = torch.FloatTensor(a).to(device)

        target = torch.FloatTensor([1.0]).to(device)

        D.train(image, target)
        g_output = G.forward(get_random(300)).detach()

        target = torch.FloatTensor([0.0]).to(device)
        D.train(g_output, target)

        target = torch.FloatTensor([1.0]).to(device)
        G.train(D, get_random(300), target)


torch.save(G.state_dict(), 'gan.pth')