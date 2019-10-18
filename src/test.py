import load_data
import numpy as np
import matplotlib.pyplot as plt


def main():
    images = load_data.load()
    print(np.shape(images))
    print(np.shape(images[0]))

    # need to transpose
    # (channels, M, N) -> (M, N, channels)
    plt.imshow(images[0].transpose((1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    main()
