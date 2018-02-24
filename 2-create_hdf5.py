from tflearn.data_utils import build_hdf5_image_dataset
build_hdf5_image_dataset("./dataset/", image_shape=(48, 48), mode='folder', grayscale= True, categorical_labels=True, normalize=True)
