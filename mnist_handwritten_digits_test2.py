import mnist_handwritten_digits
import scipy.misc, os, re, imageio
import numpy as np
import skimage.transform

nn = mnist_handwritten_digits.neuralNet(28**2, 200, 10, 0.1)
nn.load_weights()

scorecard = []
path_to_png = "C:\\Users\\marti\\Documents\\GitHub\\makeyourownneuralnetwork\\mnist_dataset\\PNG"
pngs = [png for png in os.listdir(path_to_png) if re.match(r'.*\.png$', png)]

for png in pngs:
    img_array = imageio.imread(os.path.join(path_to_png, png), as_gray=True)
    x, y = img_array.shape
    size = min(x, y)
    #img_square = scipy.misc.imresize(img_array,(size,size))
    img_square = scipy.misc.imresize(img_array,(28,28))
    #img_square = skimage.transform.resize(img_array,(28,28),anti_aliasing=False)
    #img_square = skimage.transform.downscale_local_mean(img_array,(28,28))
    img_data = 255.0 - img_square.reshape(784)
    scaled_input = (img_data / 255.0 * 0.99) + 0.01
    predicted_label = np.argmax(nn.query(scaled_input))
    true_label = int(png[-5])
    scorecard.append(1 if predicted_label == true_label else 0)

print("Accuracy of " + str(sum(scorecard) / len(scorecard)))
