import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from config import DEVICE
import torch.nn.functional as F
import torch
import numpy as np


class MotionBlur(torch.nn.Module):
    """Custom layer that picks one out of 8 motion blurs based
    on gumble softmax probability distribution"""

    def __init__(self, kernel_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # The greater the size, the more the motion.
        self.kernel_size = kernel_size
        # Create the vertical kernel.
        kernel_array = np.zeros((1,1,self.kernel_size, self.kernel_size))

        gradient_kernel = []
        last_num = 0
        for i in range(1,kernel_size+1):
            gradient_kernel.append(last_num/kernel_size+1/kernel_size)
            last_num = i

        N = kernel_array.copy()
        N[...,int((kernel_size - 1)/2)] = gradient_kernel[::-1]

        S = kernel_array.copy()
        S[...,int((kernel_size - 1)/2)] = gradient_kernel

        E = kernel_array.copy()
        E[:,:,int((kernel_size - 1)/2),...] = gradient_kernel

        W = kernel_array.copy()
        W[:,:,int((kernel_size - 1)/2),...] = gradient_kernel[::-1]

        SW = kernel_array.copy()
        mask = np.eye(SW.shape[2], dtype=bool)[:,::-1]
        SW[:,:,mask] = gradient_kernel

        SE = kernel_array.copy()
        mask = np.eye(SE.shape[2], dtype=bool)
        SE[:,:,mask] = gradient_kernel

        NW = kernel_array.copy()
        mask = np.eye(NW.shape[2], dtype=bool)
        NW[:,:,mask] = gradient_kernel[::-1]


        NE = kernel_array.copy()
        mask = np.eye(NE.shape[2], dtype=bool)[:,::-1]
        NE[:,:,mask] = gradient_kernel[::-1]
        
        self.kernel_north = torch.from_numpy(N).to(DEVICE, dtype=torch.float)
        self.kernel_south = torch.from_numpy(S).to(DEVICE, dtype=torch.float)
        self.kernel_east = torch.from_numpy(E).to(DEVICE, dtype=torch.float)
        self.kernel_west = torch.from_numpy(W).to(DEVICE, dtype=torch.float)
        self.kernel_NE = torch.from_numpy(NE).to(DEVICE, dtype=torch.float)
        self.kernel_NW = torch.from_numpy(NW).to(DEVICE, dtype=torch.float)
        self.kernel_SE = torch.from_numpy(SE).to(DEVICE, dtype=torch.float)
        self.kernel_SW = torch.from_numpy(SW).to(DEVICE, dtype=torch.float)

		
		

        # Initialize the logits
        self.logits = torch.nn.Parameter(
            torch.randn(8, device=DEVICE), requires_grad=True
        )
        self.probs = torch.nn.Parameter(F.gumbel_softmax(self.logits, tau=0.5, hard=False).to(DEVICE), requires_grad=True
        )

        # Initialize motion convlutions and final image
        self.final_image = torch.nn.Parameter(torch.zeros(1, 512, 512)).to(DEVICE)
        self.motion_east = torch.zeros(1, 512, 512).to(DEVICE)
        self.motion_west = torch.zeros(1, 512, 512).to(DEVICE)
        self.motion_south = torch.zeros(1, 512, 512).to(DEVICE)
        self.motion_north = torch.zeros(1, 512, 512).to(DEVICE)
        self.motion_NE = torch.zeros(1, 512, 512).to(DEVICE)
        self.motion_NW = torch.zeros(1, 512, 512).to(DEVICE)
        self.motion_SE = torch.zeros(1, 512, 512).to(DEVICE)
        self.motion_SW = torch.zeros(1, 512, 512).to(DEVICE)
        self.stack = torch.zeros(8, 1, 512, 512).to(DEVICE)

    def forward(self, x):
        # Compute convolutions
        self.motion_east = F.conv2d(x, self.kernel_east, stride=(1), padding="same")
        self.motion_west = F.conv2d(x, self.kernel_west, stride=(1), padding="same")
        self.motion_south = F.conv2d(x, self.kernel_south, stride=(1), padding="same")
        self.motion_north = F.conv2d(x, self.kernel_north, stride=(1), padding="same")
        self.motion_NE = F.conv2d(x, self.kernel_NE, stride=(1), padding="same")
        self.motion_NW = F.conv2d(x, self.kernel_NW, stride=(1), padding="same")
        self.motion_SE = F.conv2d(x, self.kernel_SE, stride=(1), padding="same")
        self.motion_SW = F.conv2d(x, self.kernel_SW, stride=(1), padding="same")
        self.stack = torch.stack(
            (self.motion_north, self.motion_south, self.motion_east, self.motion_west,
            self.motion_NE, self.motion_NW, self.motion_SE, self.motion_SW),
            dim=0,
        )
        # Compute final image from gumble softmax probs

        # self.probs = self.probs[:,None, None, None]
        self.final_image = self.probs[:,None, None, None]*self.stack
        self.final_image = torch.sum(self.final_image, dim = 0)
        # print(self.final_image.shape)

        # for idx, prob in enumerate(self.probs):
        #     for blur in self.stack:
        #         self.final_image = torch.add(self.final_image, torch.mul(prob, blur))
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

    def forward(self, x, targets=None):
        new_x = []
        for item in x:
            new_x.append(self.motionblur(item))
        new_x = tuple(new_x)
        if targets is None:
            return self.pretrained(new_x)
        return self.pretrained(new_x, targets)
