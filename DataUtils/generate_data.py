import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from random import randint


def generate_dataset(rawdata_root="Data", target_root="Dataset", vfold_ratio=0.2, max_samples_per_class=5000, show_imgs=False):
    """
    args:
    - rawdata_root: str, specify the directory path of raw data
    - target_root: str, specify the directory path of generated dataset
    - vfold_ratio: float(0-1), specify the test data / total data
    - max_item_per_class: int, specify the max items for each class
        (because the number of items of each class is far more than default value 5000)
    - show_imgs: bool, whether to show some random images after generation done
    """

    # Create the directories for download data and generated dataset.
    if not os.path.isdir(os.path.join("./", rawdata_root)):
        os.makedirs(os.path.join("./", rawdata_root))
    if not os.path.isdir(os.path.join("./", target_root)):
        os.makedirs(os.path.join("./", target_root))

    print("*"*50)
    print("Generate dataset from npy data")
    print("*"*50)
    all_files = glob.glob(os.path.join(rawdata_root, '*.npy'))
    print("Classes number: "+str(len(all_files)))
    print("*"*50)

    # initialize variables
    x = np.empty([0, 784])
    y = np.empty([0])
    class_names = []
    class_samples_num = []

    # load each data file
    for idx, file in enumerate(all_files):
        data = np.load(file)
        # print(data.shape)

        indices = np.arange(0, data.shape[0])
        # randomly choose max_items_per_class data from each class
        indices = np.random.choice(
            indices, max_samples_per_class, replace=False)
        data = data[indices]

        # print(data.shape)
        labels = np.full(data.shape[0], idx)

        x = np.concatenate((x, data), axis=0)
        y = np.append(y, labels)

        class_name, ext = os.path.splitext(os.path.basename(file))
        class_names.append(class_name)
        class_samples_num.append(str(data.shape[0]))

        print(str(idx+1)+"/"+str(len(all_files)) +
              "\t- "+class_name+" has been loaded. \n\t\t Totally "+str(data.shape[0])+" samples.")
        print("~"*50)

    print("\n"+"*"*50)
    print("Data loading done.")
    print("*"*50)
    data = None
    labels = None

    # randomize the dataset
    permutation = np.random.permutation(y.shape[0])
    x = x[permutation, :]
    y = y[permutation]

    # separate into training and testing
    vfold_size = int(x.shape[0]/100*(vfold_ratio*100))

    x_test = x[0:vfold_size, :]
    y_test = y[0:vfold_size]

    x_train = x[vfold_size:x.shape[0], :]
    y_train = y[vfold_size:y.shape[0]]

    # save the data into disk, this will make loading process much
    # quicker later.

    print("x_trian size: ")
    print(x_train.shape)
    print("\nx_test size: ")
    print(x_test.shape)

    if show_imgs:
        plt.figure('random images from dataset')
        plt.suptitle('random images from dataset')
        plt.subplot(221)
        idx=randint(0,len(x_train))
        plt.imshow(x_train[idx].reshape(28,28))
        plt.title(class_names[int(y_train[idx].item())])
        plt.subplot(222)
        idx=randint(0,len(x_train))
        plt.imshow(x_train[idx].reshape(28,28))
        plt.title(class_names[int(y_train[idx].item())])
        plt.subplot(223)
        idx=randint(0,len(x_train))
        plt.imshow(x_train[idx].reshape(28,28))
        plt.title(class_names[int(y_train[idx].item())])
        plt.subplot(224)
        idx=randint(0,len(x_train))
        plt.imshow(x_train[idx].reshape(28,28))
        plt.title(class_names[int(y_train[idx].item())])
        plt.show()
        

    np.savez_compressed(target_root+"/train", data=x_train,
                        target=y_train)

    np.savez_compressed(target_root+"/test", data=x_test,
                        target=y_test)

    print("*"*50)
    print("Great, data_cache has been saved into disk.")
    print("*"*50)

    with open("./DataUtils/class_names.txt", 'w') as f:
        for i in range(len(class_names)):
            f.write(
                "class name: "+class_names[i]+"\t\tnumber of samples: "+class_samples_num[i]+"\n")

    print("classes_names.txt has been saved.")
    print("*"*50)

    return True
