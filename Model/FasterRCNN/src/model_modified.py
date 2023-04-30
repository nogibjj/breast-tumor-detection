import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from config import DEVICE
import torch.nn.functional as F
import torch

fill_kernel = [
            1,
            1,
            1,
            1,
            1,
            2 / 3,
            2 / 3,
            2 / 3,
            2 / 3,
            2 / 3,
            1 / 3,
            1 / 3,
            1 / 3,
            1 / 3,
            1 / 3,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]

def make_kernel(kernel_size, direction, fill_kernel):
    kernel = torch.zeros((1, 1, kernel_size, kernel_size)).type(
                torch.FloatTensor
            )
    if direction == 'north':
        kernel[0,0,:,int(kernel_size/2)] = fill_kernel
    elif direction == 'south':
        kernel[0,0,:,int(kernel_size/2)] = fill_kernel.flip(0)
    elif direction == 'west':
        kernel[0,0,int(kernel_size/2),:] = fill_kernel
    elif direction == 'east':
        kernel[0,0,int(kernel_size/2),:] = fill_kernel.flip(0)
    elif direction == 'northeast':
        kernel[0,0,range(kernel_size), -range(kernel_size+1)] = fill_kernel
    elif direction == 'southwest':
        kernel[0,0,range(kernel_size), range(kernel_size)] = fill_kernel.flip(0)
    elif direction == 'northwest':

    return kernel



class MotionBlur(torch.nn.Module):
    """Custom layer that picks one out of 8 motion blurs based
    on gumble softmax probability distribution"""

    def __init__(self, kernel_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # The greater the size, the more the motion.
        self.kernel_size = kernel_size
        # Create the vertical kernel.
        self.kernel_north = (
            torch.zeros((1, 1, self.kernel_size, self.kernel_size)).type(
                torch.FloatTensor
            )
        ).to(DEVICE)
        self.kernel_south = (
            torch.zeros((1, 1, self.kernel_size, self.kernel_size)).type(
                torch.FloatTensor
            )
        ).to(DEVICE)
        self.kernel_east = (
            torch.zeros((1, 1, self.kernel_size, self.kernel_size)).type(
                torch.FloatTensor
            )
        ).to(DEVICE)
        self.kernel_west = (
            torch.zeros((1, 1, self.kernel_size, self.kernel_size)).type(
                torch.FloatTensor
            )
        ).to(DEVICE)
        self.kernel_north_east = (
            torch.zeros((1, 1, self.kernel_size, self.kernel_size)).type(
                torch.FloatTensor
            )
        ).to(DEVICE)
        self.kernel_north_west = (
            torch.zeros((1, 1, self.kernel_size, self.kernel_size)).type(
                torch.FloatTensor
            )
        ).to(DEVICE)
        self.south_west = (
            torch.zeros((1, 1, self.kernel_size, self.kernel_size)).type(
                torch.FloatTensor
            )
        ).to(DEVICE)
        self.southeast = (
            torch.zeros((1, 1, self.kernel_size, self.kernel_size)).type(
                torch.FloatTensor
            )
        ).to(DEVICE)
        # Fill the middle row with ones.
        self.kernel_north[
            0, 0, : int((self.kernel_size) / 2), int((self.kernel_size - 1) / 2)
        ] = torch.ones(int(self.kernel_size / 2))
        self.kernel_south[
            0, 0, int((self.kernel_size) / 2) :, int((self.kernel_size - 1) / 2)
        ] = torch.ones(int(self.kernel_size / 2))
        self.kernel_east[
            0, 0, int((self.kernel_size - 1) / 2), int((self.kernel_size) / 2) :
        ] = torch.ones(int(self.kernel_size / 2))
        self.kernel_west[
            0, 0, int((self.kernel_size - 1) / 2), : int((self.kernel_size) / 2)
        ] = torch.ones(int(self.kernel_size / 2))

        # Initialize the logits
        self.logits = torch.nn.Parameter(
            torch.randn(4, device=DEVICE), requires_grad=True
        )
        self.probs = F.gumbel_softmax(self.logits, tau=1, hard=False).to(DEVICE)

        # Initialize motion convlutions and final image
        self.final_image = torch.nn.Parameter(torch.zeros(1, 512, 512)).to(DEVICE)
        self.motion_east = torch.zeros(1, 512, 512).to(DEVICE)
        self.motion_west = torch.zeros(1, 512, 512).to(DEVICE)
        self.motion_south = torch.zeros(1, 512, 512).to(DEVICE)
        self.motion_north = torch.zeros(1, 512, 512).to(DEVICE)
        self.stack = torch.zeros(4, 1, 512, 512).to(DEVICE)

    def forward(self, x):
        # Compute convolutions
        self.motion_east = F.conv2d(x, self.kernel_east, stride=(1), padding="same")
        self.motion_west = F.conv2d(x, self.kernel_west, stride=(1), padding="same")
        self.motion_south = F.conv2d(x, self.kernel_south, stride=(1), padding="same")
        self.motion_north = F.conv2d(x, self.kernel_north, stride=(1), padding="same")
        self.stack = torch.stack(
            (self.motion_north, self.motion_south, self.motion_east, self.motion_west),
            dim=0,
        )
        # Compute final image from gumble softmax probs
        for idx, prob in enumerate(self.probs):
            for blur in self.stack:
                self.final_image = torch.add(self.final_image, torch.mul(prob, blur))
        return self.final_image


def create_model(num_classes):
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    # get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


class MyModel(torch.nn.Module):
    """Define new Faster-RCNN module"""

    def __init__(self, pretrained) -> None:
        super(MyModel, self).__init__()
        self.motionblur = MotionBlur(30)
        self.pretrained = pretrained

    def forward(self, x, targets):
        new_x = []
        for item in x:
            new_x.append(self.motionblur(item))
        new_x = tuple(new_x)
        return self.pretrained(new_x, targets)
