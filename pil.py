from PIL import Image

a = Image.open('test.png')
a.resize((50, 50))
a.save('out.png')
