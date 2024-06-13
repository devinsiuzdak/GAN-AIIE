import torch
import torch.nn as nn
import numpy as np
import os
import random
from PIL import Image

from Discriminator import Discriminator
from Generator import Generator

device = torch.device('mps')

def get_random(count):
    return torch.randn(count, device=device)

D = Discriminator()
D.to(device)
G = Generator()
G.to(device)

base_directory = 'Bird_Photos'
subfolders = [f.path for f in os.scandir(base_directory) if f.is_dir()]

dataset = []

for subfolder in subfolders:
    files = os.listdir(subfolder)
    for file in files:
         dataset.append(subfolder + '/' + file)

random.shuffle(dataset)

epochs = 1
max_images = 1000
image_count = 0

for epoch in range(epochs):
    for i in range(max_images):

        if i % 10 == 0:
            print(i)
        
        image_path = dataset[i]

        image = Image.open(image_path)
        image = image.resize((128, 128))
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

        image_count += 1
        
       

torch.save(G.state_dict(), 'gan.pth')
