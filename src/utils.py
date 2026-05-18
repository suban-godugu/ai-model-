import matplotlib.pyplot as plt

def show_image(image, title="Image"):

    plt.imshow(image, cmap='gray')

    plt.title(title)

    plt.axis('off')

    plt.show()