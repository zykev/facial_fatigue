import os
import argparse
import cv2
from skimage.transform import resize
import torch
from torch.autograd import Function
from torchvision import transforms
from Code_tomse2 import Model_Parts, util
import numpy as np
from PIL import Image

"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6371279/"


class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x, layer_name):
        outputs = []
        self.gradients = []
        if layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            for name, module in self.model._modules.items():
                x = module(x)
                if name in self.target_layers:
                    x.register_hook(self.save_gradient)
                    outputs += [x]
        else:
            x = self.model(x)
            x.register_hook(self.save_gradient)
            outputs += [x]

        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for mod in [self.model]:
            for name, module in mod._modules.items():
                if name == 'visual_encoder':
                    for sub_name, sub_module in module._modules.items():
                        if sub_module == self.feature_module:
                            target_activations, x = self.feature_extractor(x, sub_name)
                        elif "avgpool" in sub_name.lower():
                            x = sub_module(x)
                            x = x.view(x.size(0), -1)
                        else:
                            x = sub_module(x)
                else:
                    if module == self.feature_module:
                        target_activations, x = self.feature_extractor(x, name)
                    else:
                        x = module(x)
                        x = torch.nn.functional.log_softmax(x, dim=1)

        return target_activations, x


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def cams_on_img_seqs(cams, input_img):
    cam_seqs = []
    for i in range(len(cams)):
        img = input_img[i, :, :, :]
        img = np.array(img)
        img = img * 0.5 + 0.5
        # Opencv loads as BGR:
        img = img.transpose((1, 2, 0))
        cam = show_cam_on_image(img, cams[i, :, :])
        cam = Image.fromarray(cam)
        cam_seqs.append(cam)
    return cam_seqs

def camgb_on_img_seqs(cams, gb):
    camgb_seqs = []
    gb = gb.transpose((1, 0, 2, 3))
    for i in range(len(cams)):
        sub_gb = gb[i, :]
        sub_gb = sub_gb.transpose((1, 2, 0))
        cam_mask = cv2.merge([cams[i], cams[i], cams[i]])
        cam_gb = deprocess_image(cam_mask * sub_gb)
        cam_gb = Image.fromarray(cam_gb)
        camgb_seqs.append(cam_gb)
    return camgb_seqs

class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        features, output = self.extractor(input_img)

        if target_category is None:
            target_category = torch.argmax(output.cpu(), dim=1)
            target_category = target_category.view([output.shape[0], 1])
        else:
            target_category = target_category.unsqueeze(dim=1)
        one_hot = torch.full(size=output.size(), fill_value=0, dtype=torch.long)
        one_hot = one_hot.scatter_(dim=1, index=target_category, value=1).float()

        one_hot = one_hot.clone().detach().requires_grad_(True)

        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output) / output.shape[0]

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        cams = []
        for i in range(target.shape[0]):
            sub_target = target.cpu().data.numpy()[i, :]

            # weights = np.mean(grads_val, axis=(2, 3, 4))[0, :] / np.prod(grads_val.shape[2:5])
            weights = np.mean(grads_val, axis=(2, 3, 4))[i, :]
            cam = np.zeros(sub_target.shape[1:], dtype=np.float32)

            for i, w in enumerate(weights):
                cam += w * sub_target[i, :, :, :]

            cam = np.maximum(cam, 0)
            cam = resize(cam, input_img.shape[2:])
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)
            cams.append(cam)

        return cams


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input_img):
        positive_mask = (input_img > 0).type_as(input_img)
        output = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input_img, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img),
                                   torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), grad_output,
                                                 positive_mask_1), positive_mask_2)
        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img)
        output = torch.nn.functional.log_softmax(output, dim=1)

        if target_category is None:

            target_category = torch.argmax(output.cpu(), dim=1)
            target_category = target_category.view([output.shape[0], 1])
        else:
            target_category = target_category.unsqueeze(dim=1)
        one_hot = torch.full(size=output.size(), fill_value=0, dtype=torch.long)
        one_hot = one_hot.scatter_(dim=1, index=target_category, value=1).float()

        one_hot = one_hot.clone().detach().requires_grad_(True)

        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output) / output.shape[0]

        one_hot.backward(retain_graph=True)

        output = input_img.grad.cpu().data.numpy()

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=r'/home/xiaotao/Desktop/Data-S235-align', type=str, metavar='N',
                        help='the directory of videos need to be predicted')
    parser.add_argument('--label_list', default=r'./Data/selected-1.txt', type=str,
                        help='the list of video names need to be predicted')
    parser.add_argument('--save_dir', default=r'./result', type=str, metavar='N',
                        help='the directory where preditions need to be predicted')
    parser.add_argument('--model_dir',
                        default=r'./model/epoch29_loss_0.0584_acc_0.982', type=str,
                        metavar='N',
                        help='the directory where preditions need to be predicted')
    parser.add_argument('--num_classes', default=2, type=int, help='predicted class')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--loss_alpha', default=0.1, type=float,
                        help='adjust loss for crossentrophy')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)

# load model

def LoadParameter(_structure, _parameterDir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(_parameterDir, map_location=torch.device(device))
    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = _structure.state_dict()
    for key in pretrained_state_dict:
        model_state_dict[key.replace('module.', '')] = pretrained_state_dict[key]

    _structure.load_state_dict(model_state_dict)
    return _structure

# load data
def load_imgs_total_frame(video_root, video_list):
    imgs_first_dict = {}
    imgs_first = []
    video_names = []
    with open(video_list, 'r') as imf:
        imf = imf.readlines()

        for id, line in enumerate(imf):

            video_label = line.strip().split(' ')

            video_name, fatigue = video_label
            fatigue = (np.float32(fatigue) - 1) / 4.0

            if video_name.split('.')[-1] == 'mp4':
                video_path = os.path.join(video_root, video_name.replace(".mp4", ""))
            elif video_name.split('.')[-1] == 'mov':
                video_path = os.path.join(video_root, video_name.replace(".mov", ""))
            else:
                video_path = os.path.join(video_root, video_name)

            img_lists = os.listdir(video_path)
            img_lists.sort(key=lambda x: int(x.split('.')[0]))  # sort files by ascending
            imgs_first_dict[video_name] = []
            for frame in img_lists:
                imgs_first_dict[video_name].append(
                    (os.path.join(video_path, frame), fatigue))
            ###  return video frame index  #####
            ed_list = sample_seg_full(imgs_first_dict[video_name])
            imgs_first.append(ed_list)
            video_names.append(video_name)

    return imgs_first, video_names

def sample_seg_full(orig_list, seg_num=32):
    ed_list = []
    part = int(len(orig_list)) // seg_num
    if part == 0:
        print('less 32')
    else:
        for n in range(int(part)):
            ed_list.append(orig_list[n * seg_num: n * seg_num + seg_num])

    return ed_list

class PredDataSet(torch.utils.data.Dataset):
    '''
    This dataset return entire frames for a video. this means that the number of return for each time is different.
    sample_rate: num_of_image per second
    '''

    def __init__(self, imgs_dict, transform=None, transformVideoAug=None):
        self.imgs_first_dict = imgs_dict
        self.transform = transform
        self.transformVideoAug = transformVideoAug

    def __getitem__(self, index):
        image_label = self.imgs_first_dict[index]

        image_list = []
        for item, fatigue in image_label:
            img = Image.open(item).convert("RGB")
            img_ = img
            image_list.append(img_)

            sample = float(fatigue)
            target_list = sample


        if self.transformVideoAug is not None:
            image_list = self.transformVideoAug(image_list)


        if self.transform is not None:
            image_list = [self.transform(image) for image in image_list]

        image_list = torch.stack(image_list, dim=0)
        target_list = [torch.tensor(target_list)]



        return image_list, target_list

    def __len__(self):
        return len(self.imgs_first_dict)


def LoadPredictDataset(imgs_dict):

    pred_dataset = PredDataSet(
        imgs_dict=imgs_dict,
        # transformVideoAug=transforms.Compose([affine.Resize([256, 256]), crop.CenterCrop(224)]),
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                           std=(0.5, 0.5, 0.5))])
    )

    pred_loader = torch.utils.data.DataLoader(
        pred_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False)

    return pred_loader

# save path
def get_save_path(save_dir, model_dir, video_name):
    save_path = os.path.join(save_dir, model_dir.split('/')[2], model_dir.split('/')[4], video_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    return save_path


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for resnet and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()

    model = Model_Parts.FullModal_VisualFeatureAttention(num_class=args.num_classes, feature_dim=256, non_local_pos=3,
                                                         first_channel=64)
    model = LoadParameter(model, args.model_dir)

    model1 = Model_Parts.FullModal_VisualFeatureAttention(num_class=args.num_classes, feature_dim=256, non_local_pos=3,
                                                         first_channel=64)
    model1 = LoadParameter(model1, args.model_dir)

    # get model
    grad_cam = GradCam(model=model, feature_module=model.visual_encoder.layer4,
                       target_layer_names=["1"], use_cuda=args.use_cuda)

    gb_model = GuidedBackpropReLUModel(model=model1, use_cuda=args.use_cuda)

    # get input
    imgs_first, video_names = load_imgs_total_frame(args.data_dir, args.label_list)
    for data_id, (imgs_dict, video_name) in enumerate(zip(imgs_first, video_names)):

        print('processing' + video_name + '........')
        pred_loader = LoadPredictDataset(imgs_dict)
        for batch_idx, (input_image, sample) in enumerate(pred_loader):

            input_var = np.transpose(input_image, (0, 2, 1, 3, 4))

            # cam
            # target category: If None, returns the map for the highest scoring category. Otherwise, targets the requested category.
            sample_catego = util.label_to_categorical(sample[-1], args.num_classes)
            grayscale_cam = grad_cam(input_var, target_category=None)

            # cams on sub video
            for i in range(len(grayscale_cam)):
                cam_seqs = cams_on_img_seqs(grayscale_cam[i], input_image[i])
                save_path = os.path.join(get_save_path(args.save_dir, args.model_dir, video_name), 'cam' + '_' + str(batch_idx) + str(i) + '.gif')
                cam_seqs[0].save(save_path, save_all=True, append_images=cam_seqs, duration=100)


            gb = gb_model(input_var, target_category=None)

            # camgb on sub video
            for i in range(len(gb)):
                camgb_seqs = camgb_on_img_seqs(grayscale_cam[i], gb[i, :, :, :, :])
                save_path = os.path.join(get_save_path(args.save_dir, args.model_dir, video_name), 'camgb' + '_' + str(batch_idx) + str(i) + '.gif')
                camgb_seqs[0].save(save_path, save_all=True, append_images=camgb_seqs, duration=100)
