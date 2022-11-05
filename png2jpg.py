from PIL import Image
import os

im1 = Image.open(r'./KenBurns/Input/08265-1049740864.png')
im1.save(r'./KenBurns/Input/08265-1049740864.jpg')

os.remove(r'./KenBurns/Input/08265-1049740864.png')