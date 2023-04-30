import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from config import DEVICE
import torch.nn.functional as F
import torch


class MotionBlur(torch.nn.Module):
    """Custom layer that picks one out of 8 motion blurs based
    on gumble softmax probability distribution"""

    def __init__(self, kernel_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # The greater the size, the more the motion.
        self.kernel_size = kernel_size
        # Create the vertical kernel.
        self.kernel_north = torch.zeros((self.kernel_size, self.kernel_size)).type(
            torch.DoubleTensor
        )
        self.kernel_south = torch.zeros((self.kernel_size, self.kernel_size)).type(
            torch.DoubleTensor
        )
        self.kernel_east = torch.zeros((self.kernel_size, self.kernel_size)).type(
            torch.DoubleTensor
        )
        self.kernel_west = torch.zeros((self.kernel_size, self.kernel_size)).type(
            torch.DoubleTensor
        )
        # Fill the middle row with ones.
        self.kernel_north[
            : int((self.kernel_size) / 2), int((self.kernel_size - 1) / 2)
        ] = torch.ones(int(self.kernel_size / 2))
        self.kernel_south[
            int((self.kernel_size) / 2) :, int((self.kernel_size - 1) / 2)
        ] = torch.ones(int(self.kernel_size / 2))
        self.kernel_east[
            int((self.kernel_size - 1) / 2), int((self.kernel_size) / 2) :
        ] = torch.ones(int(self.kernel_size / 2))
        self.kernel_west[
            int((self.kernel_size - 1) / 2), : int((self.kernel_size) / 2)
        ] = torch.ones(int(self.kernel_size / 2))

        # # cast the kernels to higher dimensions
        # self.kernel_north = torch.stack(
        #     (
        #         self.kernel_north.view(1, self.kernel_size, self.kernel_size),
        #         self.kernel_north.view(1, self.kernel_size, self.kernel_size),
        #         self.kernel_north.view(1, self.kernel_size, self.kernel_size),
        #     ),
        #     dim=0,
        # )
        # self.kernel_south = torch.stack(
        #     (
        #         self.kernel_south.view(1, self.kernel_size, self.kernel_size),
        #         self.kernel_south.view(1, self.kernel_size, self.kernel_size),
        #         self.kernel_south.view(1, self.kernel_size, self.kernel_size),
        #     ),
        #     dim=0,
        # )
        # self.kernel_east = torch.stack(
        #     (
        #         self.kernel_east.view(1, self.kernel_size, self.kernel_size),
        #         self.kernel_east.view(1, self.kernel_size, self.kernel_size),
        #         self.kernel_east.view(1, self.kernel_size, self.kernel_size),
        #     ),
        #     dim=0,
        # )
        # self.kernel_west = torch.stack(
        #     (
        #         self.kernel_west.view(1, self.kernel_size, self.kernel_size),
        #         self.kernel_west.view(1, self.kernel_size, self.kernel_size),
        #         self.kernel_west.view(1, self.kernel_size, self.kernel_size),
        #     ),
        #     dim=0,
        # )

        # Initialize the logits
        self.logits = torch.nn.Parameter(torch.randn(1, 4), requires_grad=True)
        self.probs = F.gumbel_softmax(self.logits, tau=1, hard=False)

        # Initialize Final image
        self.final_image = torch.zeros(1, 1, 512, 512)

    def forward(self, x):
        # Compute convolutions
        motion_east = F.conv2d(x, self.kernel_east, stride=(1), padding="same")
        motion_west = F.conv2d(x, self.kernel_west, stride=(1), padding="same")
        motion_south = F.conv2d(x, self.kernel_south, stride=(1), padding="same")
        motion_north = F.conv2d(x, self.kernel_north, stride=(1), padding="same")
        stack = torch.stack(
            (motion_north, motion_south, motion_east, motion_west), dim=0
        )
        # Compute final image from gumble softmax probs
        for idx, prob in enumerate(self.probs[0]):
            self.final_image += stack[idx] * prob
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
