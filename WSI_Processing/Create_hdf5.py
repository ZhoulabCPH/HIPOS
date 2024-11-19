import os
import numpy as np
import pandas as pd
from PIL import Image
import tables
import concurrent.futures

def read_and_convert_image(img_path):
    """
    Reads an image, resizes it to 224x224, and converts it to a NumPy array.
    """
    img = Image.open(img_path)
    img = np.array(img.resize((224, 224)))
    img = img.astype(np.uint8)
    return img

def save_patches_to_csv(image_list, csv_path):
    """
    Saves image patch file names to a CSV file.
    """
    image_paths = [item for sublist in image_list for item in sublist]
    patch_names = [os.path.splitext(os.path.basename(path))[0] for path in image_paths]
    df = pd.DataFrame(patch_names, columns=['Patch_Name'])
    df.to_csv(csv_path, index=False)

def create_hdf5(storage_path, data_shape, dtype):
    """
    Creates and returns an extendable HDF5 dataset.
    """
    store = tables.open_file(storage_path, mode='w')
    storage = store.create_earray(store.root, atom=dtype, name='patches', shape=data_shape)
    return store, storage

def process_images(image_paths, executor):
    """
    Reads and processes a list of image paths using a ThreadPoolExecutor.
    """
    images = list(executor.map(read_and_convert_image, image_paths))
    return np.array(images)

def generate_hdf5(Train_csv_path, Test_csv_path, H5D_Train_path, H5D_Test_path, Image_path, max_workers=8):
    """
    Generates HDF5 datasets for train and test image patches.
    """
    train_store, train_storage = create_hdf5(H5D_Train_path + "dataset_train.hdf5", (0, 224, 224, 3), tables.UInt8Atom())
    test_store, test_storage = create_hdf5(H5D_Test_path + "dataset_test.hdf5", (0, 224, 224, 3), tables.UInt8Atom())

    train_filenames = pd.read_csv(Train_csv_path)["Sample_Name"].values
    test_filenames = pd.read_csv(Test_csv_path)["Sample_Name"].values

    train_paths, test_paths = [], []
    img_dir = Image_path

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, filename in enumerate(train_filenames):
            print(f"Processing train image {index}")
            image_paths_train = [os.path.join(img_dir, filename, img) for img in sorted(os.listdir(os.path.join(img_dir, filename)))]
            train_storage.append(process_images(image_paths_train, executor))
            train_paths.append(image_paths_train)

        for index, filename in enumerate(test_filenames):
            print(f"Processing test image {index}")
            image_paths_test = [os.path.join(img_dir, filename, img) for img in sorted(os.listdir(os.path.join(img_dir, filename)))]
            test_storage.append(process_images(image_paths_test, executor))
            test_paths.append(image_paths_test)

    train_store.close()
    test_store.close()
    save_patches_to_csv(train_paths, H5D_Train_path + "train_patches.csv")
    save_patches_to_csv(test_paths, H5D_Test_path + "test_patches.csv")

def generate_single_hdf5(H5D_Train_path, Image_path, max_workers=8):
    """
    Generates an HDF5 dataset for a single input directory of images.
    """
    train_store, train_storage = create_hdf5(H5D_Train_path + "dataset.hdf5", (0, 224, 224, 3), tables.UInt8Atom())

    train_filenames = os.listdir(Image_path)
    train_paths = []
    img_dir = Image_path

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, filename in enumerate(train_filenames):
            print(f"Processing image set {index}")
            image_paths_train = [os.path.join(img_dir, filename, img) for img in sorted(os.listdir(os.path.join(img_dir, filename)))]
            train_storage.append(process_images(image_paths_train, executor))
            train_paths.append(image_paths_train)

    train_store.close()
    save_patches_to_csv(train_paths, H5D_Train_path + "train_patches.csv")

if __name__ == '__main__':
    Image_path = 'KFB_Files/Tile_Save'
    H5D_Train_path = 'Output/Result/External_'
    generate_single_hdf5(H5D_Train_path, Image_path)
