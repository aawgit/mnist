import numpy as np
from keras.utils import normalize

module_1 = __import__('1')


def prep_pixels(train, test):
    new_train = []
    new_test = []
    for image in train:
        new_train.append((add_noise(image, 0.25)))

    for image in test:
        new_test.append((add_noise(image, 0.25)))

    # train, test = add_noise(train, test, 0.25)
    train_norm = normalize(new_train)
    test_norm = normalize(new_test)
    # return normalized images
    return train_norm, test_norm


def add_noise(data, noise_factor):
    data_noisy = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    data_noisy = np.clip(data_noisy, 0., 1.)
    return data_noisy


# run the test harness for evaluating a model
def run_process():
    # load dataset
    trainX, trainY, testX, testY = module_1.load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # evaluate model
    scores, histories = module_1.evaluate_model(trainX, trainY)


if __name__ == "__main__":
    run_process()
