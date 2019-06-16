import scipy.misc

path_to_png = "C:\\Users\\marti\\Documents\\GitHub\\makeyourownneuralnetwork\\mnist_dataset\\PNG"
pngs = [png for png in os.listdir(path_to_png) if re.match(r'.*\.png$', png)]

for png in pngs:
    img_array = scipy.misc.imread(png, flatten=True)
    img_data = 255.0 - img_array.reshape(784)
    scale_input = (img_data / 255.0 * 0.99) + 0.01
    true_label = int(png[-5])
