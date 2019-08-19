from model.weight_locations.segmentation import model_weight_map
import rospy
from data_loader.segmentation.cityscapes import CITYSCAPE_CLASS_LIST
from torchvision.transforms import functional as F
from transforms.classification.data_transforms import MEAN, STD
import os 
from model.segmentation.espnetv2 import ESPNetv2Segmentation
from model.segmentation.dicenet import DiCENetSegmentation	
from argparse import Namespace

def relabel(img):
    '''
    This function relabels the predicted labels so that cityscape dataset can process
    :return:
    '''
    img[img == 19] = 7
    img[img == 18] = 33
    img[img == 17] = 32
    img[img == 16] = 31
    img[img == 15] = 28
    img[img == 14] = 27
    img[img == 13] = 26
    img[img == 12] = 25
    img[img == 11] = 24
    img[img == 10] = 23
    img[img == 9] = 22
    img[img == 8] = 21
    img[img == 7] = 20
    img[img == 6] = 19
    img[img == 5] = 17
    img[img == 4] = 13
    img[img == 3] = 12
    img[img == 2] = 11
    img[img == 1] = 8
    img[img == 0] = 255
    img[img == 255] = 0
    return img


def data_transform(img, im_size):
    img = img.resize(im_size, Image.BILINEAR)
    img = F.to_tensor(img)  # convert to tensor (values between 0 and 1)
    img = F.normalize(img, MEAN, STD)  # normalize the tensor
    return img

def getPreTrainedModel(args,dataset="city"):
    model_key = '{}_{}'.format(args.model, args.s)
    dataset_key = '{}_{}x{}'.format(dataset, args.im_size[0], args.im_size[1])
    assert model_key in model_weight_map.keys(), '{} does not exist'.format(model_key)
    assert dataset_key in model_weight_map[model_key].keys(), '{} does not exist'.format(dataset_key)
    weights_test = model_weight_map[model_key][dataset_key]['weights']


    if not os.path.isfile(weights_test):
        print("Weights file not found. Check Params")
        exit(-1)

    return weights_test

def setupSegNet(args):

    args.num_classes = len(CITYSCAPE_CLASS_LIST)
    modelWeightsFile = getPreTrainedModel(args)
    
    if modelType == 'espnetv2':
        model = espNetModel(args, modelWeightsFile)
    elif modelType == 'dicenet':
        model = diceNet(args, modelWeightsFile)
    return model


def espNetModel(args, weights,dataset="city"):
    model = ESPNetv2Segmentation(args, classes=args.num_classes, dataset=dataset)
    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus >= 1 else 'cpu'
    pretrained_dict = torch.load(weights, map_location=torch.device(device))

    basenet_dict = model.base_net.state_dict()
    model_dict = model.state_dict()
    overlap_dict = {k: v for k, v in pretrained_dict.items() if k in basenet_dict}

    basenet_dict.update(overlap_dict)
    model.base_net.load_state_dict(basenet_dict)

    rospy.loginfo("Loaded PreTrained weights")

    return model

def diceNet(args, weights):
    model = DiCENetSegmentation(args, classes=args.num_classes)
    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus >= 1 else 'cpu'
    pretrained_dict = torch.load(weights, map_location=torch.device(device))

    basenet_dict = model.base_net.state_dict()
    model_dict = model.state_dict()
    overlap_dict = {k: v for k, v in pretrained_dict.items() if k in basenet_dict}

    basenet_dict.update(overlap_dict)
    model.base_net.load_state_dict(basenet_dict)
    rospy.loginfo("Loaded PreTrained weights")

    return model
