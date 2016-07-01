from PIL import Image
from collections import OrderedDict
from sklearn.decomposition import PCA #, RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier
from csv import reader, writer
from datetime import datetime

import numpy as np


def main():
    # Fetch the data
    folder = 'data/roof_images/'
    nb_img = 50  # None to consider all the images

    # Get nb_img/all the training examples
    labels = get_img_labels('training', nb_img)
    Y_train = np.array(labels.values())

    # Get the relative paths of the chosen images
    images_path = [folder + img_ID + '.jpg' for img_ID in labels.keys()]

    # Determine the average size of the images to resize them
    images_size = np.array([Image.open(filename).size for filename in images_path])
    STANDARD_SIZE = np.mean(images_size, axis=0, dtype=int)
    #print dispersion_size = np.std(images_size, axis=0)  # Standard variation: [46 48] -> Pretty dispersed

    # Resize the images & convert them into a matrix
    images = img_to_matrix(images_path, STANDARD_SIZE)

    # PCA to reduce the number of features
    pca = PCA(n_components=20)  # if inferior to the number of the number of images, otherwise PCA effective on this last nb
    X_train = pca.fit_transform(images)

    # Train a K-Neighbors classifier
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)

    # Get the test examples
    labels_test = get_img_labels('test', 100)
    images_path = [folder + img_ID + '.jpg' for img_ID in labels_test.keys()]

    # Process the predictions on chunks of the test data for memory reasons
    Y_test = np.array([], dtype=int)
    for chunk in [images_path[i: i + 1000] for i in range(0, len(images_path), 1000)]:
        # Convert them to a matrix after resizing
        X_test = img_to_matrix(chunk, STANDARD_SIZE)

        # Apply the PCA on them
        X_test = pca.transform(X_test)

        # Predict the classes thanks to the model
        Y_test = np.append(Y_test, knn.predict(X_test))

    # Write the results into a new csv file
    write_results(labels_test.keys(), Y_test)


def img_to_matrix(filenames, STANDARD_SIZE=None, verbose=False):
    """
    takes one or several filenames and turns them into a numpy array of RGB pixels
    """
    if type(filenames) is not list:
        filenames = [filenames]

    # Initialize an empty array of dimension (Number of images * (3 * number of pixels after resizing))
    images = np.zeros((len(filenames), 3*STANDARD_SIZE[0]*STANDARD_SIZE[1]), dtype=np.uint8)  # Unsigned 8 bits
    for ind, filename in enumerate(filenames):
        img = Image.open(filename)
        img = resize_img(img, STANDARD_SIZE)
        img = np.array(img.getdata()).reshape(1, -1)  # Flatten the RGB arrays into a 1D array along axis = 1
        images[(ind,)] = img

    return images


def resize_img(img, STANDARD_SIZE, verbose=False):
    """
    See also SIFT method
    """
    if verbose:
        print "changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE))
    img = img.resize(STANDARD_SIZE)
    return img


def get_img_labels(task, nb_img=None):
    """
    returns the ID/class of the images
    """
    # Read the csv file matching the ids of the images with the classes
    labels = OrderedDict()

    with open('data/' + ('id_train' if task == 'training' else 'sample_submission4') + '.csv', 'rb') as csvfile:
        rows = reader(csvfile, delimiter=',')
        rows.next()  # Skip the header
        for row in rows:
            if nb_img is not None and len(labels) >= nb_img:
                break
            labels[row[0]] = int(row[1])  # Integer conversion of the labels

    return labels


def write_results(IDs, predictions):
    with open('output/sample_submission%s.csv' % datetime.today().strftime('%Y-%m-%d %H:%M'), 'wb') as csvfile:
        labelswriter = writer(csvfile, delimiter=',')
        labelswriter.writerow(['Id', 'label'])  # Header
        for ind, ID in enumerate(IDs):
            labelswriter.writerow([ID, predictions[ind]])

if __name__ == "__main__":
    main()
