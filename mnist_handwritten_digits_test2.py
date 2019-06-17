import mnist_handwritten_digits
import os, re, imageio, skimage.transform
import numpy as np
from PIL import Image

nn = mnist_handwritten_digits.neuralNet(28**2, 200, 10, 0.1)
nn.load_weights()

scorecard = []
path_to_png = ".\\mnist_dataset\\PNG"
save_pngs = False
pngs = [png for png in os.listdir(path_to_png) if re.match(r'.*\.png$', png)]

for png in pngs:
    img_array = imageio.imread(os.path.join(path_to_png, png), as_gray=True)
    img_square = skimage.transform.resize(img_array,(28,28),anti_aliasing=False)
    # x, y = img_array.shape
    # size = min(x, y)
    # img_square = skimage.transform.resize(img_array,(size,size))
    # img_square = skimage.transform.downscale_local_mean(img_array,(28,28))
    img_data = 255.0 - img_square.reshape(784)
    scaled_input = (img_data / 255.0 * 0.99) + 0.01
    predicted_label = np.argmax(nn.query(scaled_input))
    true_label = int(png[-5])
    scorecard.append(1 if predicted_label == true_label else 0)

    # optional: save read and resized image back to file
    if save_pngs:
        img = Image.fromarray(img_square)
        img.convert("L").save(os.path.join(path_to_png + "_square", png))

print("Accuracy of " + str(sum(scorecard) / len(scorecard)))
