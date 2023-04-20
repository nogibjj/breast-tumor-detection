import torch
import cv2
import numpy as np
import pandas as pd
import os
import glob as glob
from config import CLASSES, RESIZE_TO, TRAIN_DIR, VALID_DIR, BATCH_SIZE
from torch.utils.data import Dataset, DataLoader
from utils import collate_fn, get_train_transform, get_valid_transform

annotation = pd.read_csv(
    "/workspaces/breast-tumor-detection/Model/FasterRCNN/data/annotation.csv",
    index_col=0,
    dtype={"patient_id": "str"},
)


# the dataset class
class TumorDataset(Dataset):
    """Define Tumor Pytorch Dataset object"""

    def __init__(self, dir_path, width, height, classes, annotation, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes
        self.annotation = annotation

        # get all the image paths in sorted order
        self.image_paths = glob.glob(f"{self.dir_path}/*.png")
        self.all_images = [image_path.split("/")[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        # capture the image name, patient id and the full image path
        image_name = self.all_images[idx]
        patient_id = image_name.split(".png")[0][-3:]
        image_path = os.path.join(self.dir_path, image_name)

        # read the image
        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized = image_resized / 255

        # capture the corresponding XML file for getting the annotations
        boxes = []
        labels = []

        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]

        # box coordinate for image is extracted and corrected for image size given
        ## not considering cases with only background for now ##
        labels.append(self.classes.index("tumor"))
        image_annotation = self.annotation.loc[
            self.annotation["patient_id"] == patient_id
        ]
        # xmin = left corner x-coordinates
        xmin = int(image_annotation["xmin"].values[0])
        # xmax = right corner x-coordinates
        xmax = int(image_annotation["xmax"].values[0])
        # ymin = left corner y-coordinates
        ymin = int(image_annotation["ymin"].values[0])
        # ymax = right corner y-coordinates
        ymax = int(image_annotation["ymax"].values[0])

        # resize the bounding boxes according to the...
        # ... desired `width`, `height`
        xmin_final = (xmin / image_width) * self.width
        xmax_final = (xmax / image_width) * self.width
        ymin_final = (ymin / image_height) * self.height
        yamx_final = (ymax / image_height) * self.height

        boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])

        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        # apply the image transforms
        if self.transforms:
            sample = self.transforms(
                image=image_resized, bboxes=target["boxes"], labels=labels
            )
            image_resized = sample["image"]
            target["boxes"] = torch.Tensor(sample["bboxes"])

        return image_resized, target

    def __len__(self):
        return len(self.all_images)


# prepare the final datasets and data loaders
train_dataset = TumorDataset(
    TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, annotation, get_train_transform()
)
valid_dataset = TumorDataset(
    VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES, annotation, get_valid_transform()
)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn,
)
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}\n")

if __name__ == "__main__":
    # sanity check of the Dataset pipeline with sample visualization
    dataset = TumorDataset(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, annotation)
    print(f"Number of training images: {len(dataset)}")

    # function to visualize a single sample
    def visualize_sample(image, target):
        box = target["boxes"][0]
        label = CLASSES[target["labels"]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (0, 255, 0),
            1,
        )
        cv2.putText(
            image,
            label,
            (int(box[0]), int(box[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.imshow("Image", image)
        cv2.waitKey(0)

    NUM_SAMPLES_TO_VISUALIZE = 5
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]
        visualize_sample(image, target)
