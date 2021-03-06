import argparse
import cv2
from skimage.transform import resize
import torch
from torch.autograd import Function
from torchvision import transforms
from Code_tomse2 import Model_Parts, read_data
from vidaug.augmentors import affine, crop
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

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
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
        c = 0
        for mod in [self.model.visual_encoder, self.model]:
            c += 1
            if c == 1:
                for name, module in mod._modules.items():
                    if module == self.feature_module:
                        target_activations, x = self.feature_extractor(x)
                    elif "avgpool" in name.lower():
                        x = module(x)
                        x = x.view(x.size(0), -1)
                    else:
                        x = module(x)
            else:
                c_module = 0
                for name, module in mod._modules.items():
                    c_module += 1
                    if c_module == 2:
                        if module == self.feature_module:
                            target_activations, x = self.feature_extractor(x)
                        elif "avgpool" in name.lower():
                            x = module(x)
                            x = x.view(x.size(0), -1)
                        else:
                            x = module(x)
                            x = torch.nn.functional.log_softmax(x, dim=1)

        return target_activations, x


def preprocess_image(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def cams_on_img_seqs(cams, img_dir):
    # get image
    cam_seqs = []
    for i in range(len(img_dir)):
        img = cv2.imread(img_dir[i])
        img = np.float32(img) / 255
        # Opencv loads as BGR:
        img = img[:, :, ::-1]
        img = resize(img, (224, 224, 3))
        cam = show_cam_on_image(img, cams[i, :, :])
        cam = Image.fromarray(cam)
        cam_seqs.append(cam)
    return cam_seqs

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

        if target_category == None:
            target_category = torch.argmax(output.cpu(), dim=1)
            target_category = target_category.view([output.shape[0], 1])
        else:
            target_category = torch.full(size=(output.shape[0], 1), fill_value=target_category)
        one_hot = torch.full(size=output.size(), fill_value=0).float()
        one_hot = one_hot.scatter_(dim=1, index=target_category, value=1)

        one_hot = one_hot.clone().detach().requires_grad_(True)

        # one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        # one_hot[0][target_category] = 1
        # one_hot = torch.from_numpy(one_hot).requires_grad_(True)
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
            weights = np.mean(grads_val, axis=(2, 3, 4))[i, :] / np.prod(grads_val.shape[2:5])
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

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)
        one_hot.backward(retain_graph=True)

        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--data_dir', default=r'/home/xiaotao/Desktop/Data-S235', type=str, metavar='N',
                        help='the directory of videos need to be predicted')
    parser.add_argument('--video_name', default='1100.mp4', type=str,
                        metavar='N', help='the list of video names need to be predicted')
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


# dataloader
def LoadPredictDataset(root_train, arg_train_list):

    pre_loader = read_data.LoadData_cam(
        video_root=root_train,
        video_list=arg_train_list,
        transformVideoAug=transforms.Compose([affine.Resize([256, 256]), crop.CenterCrop(224)]),
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                           std=(0.5, 0.5, 0.5))])
    )

    pre_dataset, dataset_dict = pre_loader.image_process()

    return pre_dataset, dataset_dict



if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()

    import os
    dir_model = "./model/epoch17_loss_0.117_acc_0.759.pt"
    model = Model_Parts.FullModal_VisualFeatureAttention(num_class=2, feature_dim=256, non_local_pos=3,
                                                         first_channel=64)
    model = Model_Parts.LoadParameter(model, dir_model)


    # get grad_cam model
    grad_cam = GradCam(model=model, feature_module=model.visual_encoder.layer4,
                       target_layer_names=["0"], use_cuda=args.use_cuda)

    # #get image
    # image_dir_path = os.path.join(args.data_dir, args.video_name.split('.')[0])
    # img_list = os.listdir(image_dir_path)
    # img_list.sort(key=lambda x: int(x.split('.')[0]))
    # img_path = os.path.join(image_dir_path, img_list[1])
    # img = cv2.imread(img_path)
    # img = np.float32(img) / 255
    # # Opencv loads as BGR:
    # img = img[:, :, ::-1]


    # get input
    data_dir = args.data_dir
    video_name = args.video_name

    input_img, img_dir = LoadPredictDataset(data_dir, video_name)

    # input_img = list(pre_loader)[0][0]
    input_var = np.transpose(input_img, (0, 2, 1, 3, 4))


    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None
    grayscale_cam = grad_cam(input_var, target_category)

    for i in range(len(grayscale_cam)):
        cam_seqs = cams_on_img_seqs(grayscale_cam[i], img_dir[i])
        save_name = video_name.split('.')[0] + '_' + str(i) + '.gif'
        cam_seqs[0].save(os.path.join('./result', save_name), save_all=True, append_images=cam_seqs, duration=0.5)

    # img = resize(img, (224, 224, 3))
    # cam = show_cam_on_image(img, grayscale_cam[0, :, :])
    # cv2.imwrite("cam2.jpg", cam)


    # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    # gb = gb_model(input_var, target_category=target_category)
    # # gb = gb.transpose((1, 2, 0))
    #
    # # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    # # cam_gb = deprocess_image(cam_mask * gb)
    # # gb = deprocess_image(gb)
    #
    # cv2.imwrite('ori.jpg', input_img[0,0,:,:,:])
    # # cv2.imwrite("cam.jpg", cam)
    # # cv2.imwrite('gb.jpg', gb)
    # # cv2.imwrite('cam_gb.jpg', cam_gb)
