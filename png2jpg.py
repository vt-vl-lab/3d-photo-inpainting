from PIL import Image
import os

im1 = Image.open(r'./KenBurns/Input/05529-2968780176.png')
im1.save(r'./KenBurns/Input/05529-2968780176.png.jpg')

os.remove(r'./KenBurns/Input/05529-2968780176.png')