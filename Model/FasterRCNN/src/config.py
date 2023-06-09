import torch

BATCH_SIZE = 16  # increase / decrease according to GPU memeory
RESIZE_TO = 512  # resize the image for training and transforms
NUM_EPOCHS = 100  # number of epochs to train for

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# training images and XML files directory
TRAIN_DIR = "/content/breast-tumor-detection/Model/FasterRCNN/data/train"
# validation images and XML files directory
VALID_DIR = "/content/breast-tumor-detection/Model/FasterRCNN/data/val"
# classes: 0 index is reserved for background
CLASSES = ["background", "tumor"]
NUM_CLASSES = 2
# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

# location to save model and plots
OUT_DIR = "/content/breast-tumor-detection/Model/FasterRCNN/outputs"

SAVE_PLOTS_EPOCH = 100  # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 100  # save model after these many epochs
