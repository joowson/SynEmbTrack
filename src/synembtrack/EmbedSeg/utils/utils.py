import os
import threading

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib.patches import Ellipse
import seaborn as sns



class AverageMeter(object):

    def __init__(self, num_classes=1):
        self.num_classes = num_classes
        self.reset()
        self.lock = threading.Lock()

    def reset(self):
        self.sum = [0] * self.num_classes
        self.count = [0] * self.num_classes
        self.avg_per_class = [0] * self.num_classes
        self.avg = 0

    def update(self, val, cl=0):
        with self.lock:
            self.sum[cl] += val
            self.count[cl] += 1
            self.avg_per_class = [
                x / y if x > 0 else 0 for x, y in zip(self.sum, self.count)]
            self.avg = sum(self.avg_per_class) / len(self.avg_per_class)


class Visualizer:

    def __init__(self, keys, cmap):

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 2, figsize=(10, 10))

        self.ax1[1].axis('off')
        self.ax2[0].axis('off')
        self.ax2[1].axis('off')
        self.cmap = cmap

    def display(self, image, key, title, center_x=None, center_y=None, samples_x=None, samples_y=None,
                sample_spatial_embedding_x=None, sample_spatial_embedding_y=None,
                sigma_x=None, sigma_y=None, color_sample=None, color_embedding=None):

        n_images = len(image) if isinstance(image, (list, tuple)) else 1


        self.fig.frameon = False
        if key == 'image':
            self.ax1[0].cla()
            self.ax1[0].imshow(self.prepare_img(image), cmap='magma')
            self.ax1[0].axis('off')
        elif key == 'groundtruth':

            self.ax1[1].cla()
            self.ax1[1].imshow(self.prepare_img(image), cmap=self.cmap, interpolation='None')
            self.ax1[1].axis('off')
        elif key == 'prediction':
            self.ax2[1].cla()
            self.ax2[1].imshow(self.prepare_img(image), cmap=self.cmap, interpolation='None')
            self.ax2[1].axis('off')
        elif key == 'center':
            self.ax2[0].cla()
            self.ax2[0].imshow(self.prepare_img(image), cmap='gray')
            for i in range(len(color_sample.items())):
                self.ax2[0].plot(center_x[i + 1], center_y[i + 1], color=color_embedding[i + 1 ], marker='x')
                self.ax2[0].scatter(samples_x[i + 1], samples_y[i + 1], color=color_sample[i + 1], marker='+')
                self.ax2[0].scatter(sample_spatial_embedding_x[i + 1], sample_spatial_embedding_y[i + 1],
                                    color=color_embedding[i + 1], marker='.')
                ellipse = Ellipse((center_x[i + 1], center_y[i + 1]), width=sigma_x[i + 1], height=sigma_y[i + 1],
                                  angle=0, color=color_embedding[i + 1], alpha=0.5)
                self.ax2[0].add_artist(ellipse)
            self.ax2[0].axis('off')
        plt.tight_layout()
        plt.draw()
        #plt.savefig(f'/home/trisara/mMPM/04_jobs_1_Emb/{title}.png', dpi=300)
        self.mypause(0.0001)

    @staticmethod
    def prepare_img(image):
        if isinstance(image, Image.Image):
            return image

        if isinstance(image, torch.Tensor):
            image.squeeze_()
            image = image.numpy()

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] in {1, 3}:
                image = image.transpose(1, 2, 0)
            return image

    @staticmethod
    def mypause(interval):
        backend = plt.rcParams['backend']
        if backend in matplotlib.rcsetup.interactive_bk:
            figManager = matplotlib._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return



class Cluster_3d:
    def __init__(self, grid_z, grid_y, grid_x, pixel_z, pixel_y, pixel_x, one_hot=False):

        xm = torch.linspace(0, pixel_x, grid_x).view(1, 1, 1, -1).expand(1, grid_z, grid_y, grid_x)
        ym = torch.linspace(0, pixel_y, grid_y).view(1, 1, -1, 1).expand(1, grid_z, grid_y, grid_x)
        zm = torch.linspace(0, pixel_z, grid_z).view(1, -1, 1, 1).expand(1, grid_z, grid_y, grid_x)
        xyzm = torch.cat((xm, ym, zm), 0)

        self.xyzm = xyzm.cuda()
        self.one_hot = one_hot
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y
        self.pixel_z = pixel_z

    def cluster_with_gt(self, prediction, instance, n_sigma=1, ):

        depth, height, width = prediction.size(1), prediction.size(2), prediction.size(3)
        xyzm_s = self.xyzm[:, 0:depth, 0:height, 0:width]  # 3 x d x h x w
        spatial_emb = torch.tanh(prediction[0:3]) + xyzm_s  # 3 x d x h x w
        sigma = prediction[3:3 + n_sigma]  # n_sigma x h x w

        instance_map = torch.zeros(depth, height, width).short().cuda()
        unique_instances = instance.unique()
        unique_instances = unique_instances[unique_instances != 0]

        for id in unique_instances:
            mask = instance.eq(id).view(1, depth, height, width)
            center = spatial_emb[mask.expand_as(spatial_emb)].view(
                3, -1).mean(1).view(3, 1, 1, 1)  # 3 x 1 x 1 x 1
            s = sigma[mask.expand_as(sigma)].view(n_sigma, -1).mean(1).view(n_sigma, 1, 1, 1)

            s = torch.exp(s * 10)  # n_sigma x 1 x 1
            dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb - center, 2) * s, 0))
            proposal = (dist > 0.5)
            instance_map[proposal] = id.item() # TODO

        return instance_map

    def cluster(self, prediction, n_sigma=3, seed_thresh=0.5, min_mask_sum=128, min_unclustered_sum=0, min_object_size=36):

        depth, height, width = prediction.size(1), prediction.size(2), prediction.size(3)
        xyzm_s = self.xyzm[:, 0:depth, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:3]) + xyzm_s  # 3 x d x h x w

        sigma = prediction[3:3 + n_sigma]  # n_sigma x d x h x w
        seed_map = torch.sigmoid(prediction[3 + n_sigma:3 + n_sigma + 1])  # 1 x d x h x w
        instance_map = torch.zeros(depth, height, width).short()
        instances = []  # list

        count = 1
        mask = seed_map > 0.5
        if mask.sum() > min_mask_sum:  # top level decision: only start creating instances, if there are atleast 128 pixels in foreground!

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(n_sigma, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).short().cuda()
            instance_map_masked = torch.zeros(mask.sum()).short().cuda()

            while (unclustered.sum() > min_unclustered_sum):  # stop when the seed candidates are less than 128
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < seed_thresh:
                    break
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0

                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0))

                proposal = (dist > 0.5).squeeze()
                if proposal.sum() > min_object_size:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > 0.5:
                        instance_map_masked[proposal.squeeze()] = count
                        instance_mask = torch.zeros(depth, height, width).short()
                        instance_mask[mask.squeeze().cpu()] = proposal.short().cpu()
                        count += 1
                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()

        return instance_map, instances


class Cluster:

    def __init__(self, grid_y, grid_x, pixel_y, pixel_x, one_hot=False):

        xm = torch.linspace(0, pixel_x, grid_x).view(1, 1, -1).expand(1, grid_y, grid_x)
        ym = torch.linspace(0, pixel_y, grid_y).view(1, -1, 1).expand(1, grid_y, grid_x)
        xym = torch.cat((xm, ym), 0)

        self.xym = xym.cuda()
        self.one_hot = one_hot
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y

    def cluster_with_gt(self, prediction, instance, n_sigma=1, ):
        height, width = prediction.size(1), prediction.size(2)

        xym_s = self.xym[:, 0:height, 0:width]  # 2 x h x w

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w

        instance_map = torch.zeros(height, width).short().cuda()
        if (self.one_hot):
            unique_instances = torch.arange(instance.size(0))
        else:
            unique_instances = instance.unique()
            unique_instances = unique_instances[unique_instances != 0]

        for id in unique_instances:
            if (self.one_hot):
                mask = instance[id].eq(1).view(1, height, width)
            else:
                mask = instance.eq(id).view(1, height, width)

            center = spatial_emb[mask.expand_as(spatial_emb)].view(
                2, -1).mean(1).view(2, 1, 1)  # 2 x 1 x 1

            s = sigma[mask.expand_as(sigma)].view(n_sigma, -1).mean(1).view(n_sigma, 1, 1)

            s = torch.exp(s * 10)  # n_sigma x 1 x 1 #
            dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb - center, 2) * s, 0))
            proposal = (dist > 0.5)
            if (self.one_hot):
                instance_map[proposal] = id.item() + 1 # TODO
            else:
                instance_map[proposal] = id.item() #TODO

        return instance_map

    def cluster(self, prediction, n_sigma=2, seed_thresh=0.5, min_mask_sum=128, min_unclustered_sum=0, min_object_size=36):

        height, width = prediction.size(1), prediction.size(2)
        #print(height, width)
        xym_s = self.xym[:, 0:height, 0:width]

        
        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w

        sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[2 + n_sigma:2 + n_sigma + 1])  # 1 x h x w

        instance_map = torch.zeros(height, width).short()
        instances = []  # list

        count = 1
        mask = seed_map > 0.5

        if mask.sum() > min_mask_sum:

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(n_sigma, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).short().cuda()
            instance_map_masked = torch.zeros(mask.sum()).short().cuda()

            while (unclustered.sum() > min_unclustered_sum):
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < seed_thresh:
                    break
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0

                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0))

                proposal = (dist > 0.5).squeeze()
                if proposal.sum() > min_object_size:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > 0.5:
                        instance_map_masked[proposal.squeeze()] = count
                        instance_mask = torch.zeros(height, width).short()
                        instance_mask[mask.squeeze().cpu()] = proposal.short().cpu()  # TODO
                        center_image = torch.zeros(height, width).short()

                        center[0] = int(degrid(center[0].cpu().detach().numpy(), self.grid_x, self.pixel_x))
                        center[1] = int(degrid(center[1].cpu().detach().numpy(), self.grid_y, self.pixel_y))
                        #print(int(center[0].item()),center[1].item())
                        '''
                        print(self.grid_x ,
                                self.grid_y ,
                                self.pixel_x ,
                                self.pixel_y)
                        print(center_image.shape)
                        '''
                        if int(center[1].item()) >= height:
                            #print(center[1], center[1].item(), center[0].item())    
                            #print("dim 1 is out of the screen")
                            if int(center[0].item()) >= width:
                                print("dim 2 is out of the screen")    
                                center_image[height-1, width-1] = True
                            else:
                                center_image[height-1, int(center[0].item())] = True
                        elif int(center[0].item()) >= width:
                            #print("dim 2 is out of the screen")
                            #print(center[1], center[1].item(), center[0].item())    
                            center_image[int(center[1].item()), width-1] = True
                        else:
                            center_image[int(center[1].item()), int(center[0].item())] = True
                        instances.append(
                            {'mask': instance_mask.squeeze() * 255, 'score': seed_score,
                             'center-image': center_image})
                        count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()

        return instance_map, instances


class Logger:

    def __init__(self, keys, title=""):

        self.data = {k: [] for k in keys}
        self.title = title
        self.win = None

        print('Created logger with keys:  {}'.format(keys))

    def plot(self, save=False, save_dir=""):



        if self.win is None:
            self.win = plt.subplots()
        fig, ax = self.win
        ax.cla()

        keys = []
        count = 0
        for key in self.data:
            if (count < 3):
                keys.append(key)
                data = self.data[key]
                ax.plot(range(len(data)), data, marker='.')
                count += 1
        ax.legend(keys, loc='upper right')
        ax.set_title(self.title)

        plt.draw()
        plt.close(fig)
        Visualizer.mypause(0.001)


        if save:
            # save figure
            fig.savefig(os.path.join(save_dir, self.title + '.png'))

            # save data as csv
            df = pd.DataFrame.from_dict(self.data)
            df.to_csv(os.path.join(save_dir, self.title + '.csv'))




    def add(self, key, value):
        assert key in self.data, "Key not in data"
        self.data[key].append(value)

def degrid(meter, grid_size, pixel_size):
    
    return int(meter * (grid_size-1) / pixel_size + 1)



def add_samples(samples, ax, n, amax):
    samples_list = []
    for i in range(samples.shape[1]):
        samples_list.append(degrid(samples[ax, i], n, amax))
    return samples_list


def _to_tensor_bool(x, device):
    """numpy/list/bool tensor → 같은 device의 bool tensor로."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(x, (list, tuple)):
        # center_images처럼 리스트/튜플이면 각 원소를 변환
        return type(x)(_to_tensor_bool(xx, device) for xx in x)
    if not torch.is_tensor(x):
        return x
    x = x.to(device, non_blocking=True)
    return x.bool() if x.dtype != torch.bool else x

def _to_tensor(x, device, dtype=None):
    """numpy → tensor, 그 외 tensor는 device/dtype만 맞춤."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if torch.is_tensor(x):
        return x.to(device, dtype=dtype, non_blocking=True)
    return x  # None 등은 그대로

# def prepare_embedding_for_train_image(one_hot, grid_x, grid_y, pixel_x, pixel_y, predictions, instance_ids, center_images, output,
#                                       instances, n_sigma):
    

    
#     xm = torch.linspace(0, pixel_x, grid_x).view(1, 1, -1).expand(1, grid_y, grid_x)
#     ym = torch.linspace(0, pixel_y, grid_y).view(1, -1, 1).expand(1, grid_y, grid_x)
#     xym = torch.cat((xm, ym), 0)
#     height, width = predictions.size(0), predictions.size(1)
#     xym_s = xym[:, 0:height, 0:width].contiguous()
#     spatial_emb = torch.tanh(output[0, 0:2]).cpu() + xym_s
#     sigma = output[0, 2: 2 + n_sigma]  # 2/3 Y X
#     color_sample = sns.color_palette("dark")
#     color_embedding = sns.color_palette("bright")
#     color_sample_dic = {}
#     color_embedding_dic = {}
#     samples_x = {}
#     samples_y = {}
#     sample_spatial_embedding_x = {}
#     sample_spatial_embedding_y = {}
#     center_x = {}
#     center_y = {}
#     sigma_x = {}
#     sigma_y = {}
#     if one_hot:
#         instance_ids +=1 # make the ids go from 1 to ...

#     for id in instance_ids:
#         if (one_hot):
#             in_mask = instances[0, id-1, ...].eq(1) #for one_hot, id goes from 1 to ...
#         else:
#             in_mask = instances[0].eq(id)  # 1 x h x w

#         if in_mask.device != xym_s.device:
#             in_mask = in_mask.to(xym_s.device, non_blocking=True)

#         xy_in = xym_s[in_mask.expand_as(xym_s)].view(2, -1)  # 2 N
#         perm = torch.randperm(xy_in.size(1))
#         idx = perm[:5]
#         samples = xy_in[:, idx]
#         samples_x[id.item()] = add_samples(samples, 0, grid_x - 1, pixel_x)
#         samples_y[id.item()] = add_samples(samples, 1, grid_y - 1, pixel_y)

#         # embeddings
#         spatial_emb_in = spatial_emb[in_mask.expand_as(spatial_emb)].view(2, -1)
#         samples_spatial_embeddings = spatial_emb_in[:, idx]

#         sample_spatial_embedding_x[id.item()] = add_samples(samples_spatial_embeddings, 0, grid_x - 1,
#                                                             pixel_x)
#         sample_spatial_embedding_y[id.item()] = add_samples(samples_spatial_embeddings, 1, grid_y - 1,
#                                                             pixel_y)

#         #center_mask = in_mask & center_images[0].byte()
#         center_mask = in_mask & center_images[0]
#         if (center_mask.sum().eq(1)):
#             center = xym_s[center_mask.expand_as(xym_s)].view(2, 1, 1)
#         else:
#             xy_in = xym_s[in_mask.expand_as(xym_s)].view(2, -1)
#             center = xy_in.mean(1).view(2, 1, 1)  # 2 x 1 x 1

#         center_x[id.item()] = degrid(center[0], grid_x - 1, pixel_x)
#         center_y[id.item()] = degrid(center[1], grid_y - 1, pixel_y)

#         # sigma
#         s = sigma[in_mask.expand_as(sigma)].view(n_sigma, -1).mean(1)
#         s = torch.exp(s * 10)  # TODO
#         sigma_x_tmp = 0.5 / s[0]
#         sigma_y_tmp = 0.5 / s[1]
#         sigma_x[id.item()] = degrid(torch.sqrt(sigma_x_tmp), grid_x - 1, pixel_x)
#         sigma_y[id.item()] = degrid(torch.sqrt(sigma_y_tmp), grid_y - 1, pixel_y)

#         # colors
#         color_sample_dic[id.item()] = color_sample[int(id % 10)]
#         color_embedding_dic[id.item()] = color_embedding[int(id % 10)]
#     return center_x, center_y, samples_x, samples_y, sample_spatial_embedding_x, \
#            sample_spatial_embedding_y, sigma_x, sigma_y, color_sample_dic, color_embedding_dic

def prepare_embedding_for_train_image(one_hot, grid_x, grid_y, pixel_x, pixel_y,
                                      predictions, instance_ids, center_images, output,
                                      instances, n_sigma):

    # 0) 기준 device를 하나로 고정
    device = output.device

    # 1) grid 좌표를 같은 device/float32로 생성
    xm = torch.linspace(0, pixel_x, grid_x, device=device, dtype=torch.float32).view(1, 1, -1).expand(1, grid_y, grid_x)
    ym = torch.linspace(0, pixel_y, grid_y, device=device, dtype=torch.float32).view(1, -1, 1).expand(1, grid_y, grid_x)
    xym = torch.cat((xm, ym), 0)

    height, width = predictions.size(0), predictions.size(1)
    xym_s = xym[:, 0:height, 0:width].contiguous()                     # (2, H, W) on device

    # 2) 출력 텐서들도 같은 device로 (여기서 output은 이미 model device)
    spatial_emb = torch.tanh(output[0, 0:2]) + xym_s                   # (2, H, W) on device
    sigma       = output[0, 2: 2 + n_sigma]                            # (n_sigma, H, W) on device

    # 3) 마스크/인스턴스/센터 이미지를 device+dtype 통일 (bool 권장)
    instances     = _to_tensor(instances, device)                      # 보통 long/float 유지
    center_images = _to_tensor_bool(center_images, device)             # 리스트/텐서 모두 bool로

    color_sample = sns.color_palette("dark")
    color_embedding = sns.color_palette("bright")
    color_sample_dic, color_embedding_dic = {}, {}
    samples_x, samples_y = {}, {}
    sample_spatial_embedding_x, sample_spatial_embedding_y = {}, {}
    center_x, center_y = {}, {}
    sigma_x, sigma_y = {}, {}

    if one_hot:
        # torch.Tensor일 수도 있어서 일괄 처리
        if torch.is_tensor(instance_ids):
            instance_ids = (instance_ids + 1).tolist()
        else:
            instance_ids = [int(i) + 1 for i in instance_ids]

    for id_ in instance_ids:
        # id_가 tensor면 int로
        id_int = int(id_.item()) if torch.is_tensor(id_) else int(id_)

        # 4) 인스턴스 마스크 만들기 (H,W) -> bool+device
        if one_hot:
            # instances: (1, K, H, W) 가정
            in_mask = instances[0, id_int - 1, ...].eq(1)
        else:
            # instances: (1, H, W) 가정
            in_mask = instances[0].eq(id_int)

        if in_mask.device != device:
            in_mask = in_mask.to(device, non_blocking=True)
        if in_mask.dtype != torch.bool:
            in_mask = in_mask.bool()

        # 5) 좌표 샘플링
        xy_in = xym_s[in_mask.expand_as(xym_s)].view(2, -1)            # (2, N)
        if xy_in.numel() == 0:
            # 비어있는 인스턴스 방어
            continue
        perm = torch.randperm(xy_in.size(1), device=device)
        idx = perm[:min(5, xy_in.size(1))]
        samples = xy_in[:, idx]
        samples_x[id_int] = add_samples(samples, 0, grid_x - 1, pixel_x)
        samples_y[id_int] = add_samples(samples, 1, grid_y - 1, pixel_y)

        # 6) 임베딩 샘플링
        spatial_emb_in = spatial_emb[in_mask.expand_as(spatial_emb)].view(2, -1)  # (2, N)
        samples_spatial_embeddings = spatial_emb_in[:, idx]
        sample_spatial_embedding_x[id_int] = add_samples(samples_spatial_embeddings, 0, grid_x - 1, pixel_x)
        sample_spatial_embedding_y[id_int] = add_samples(samples_spatial_embeddings, 1, grid_y - 1, pixel_y)

        # 7) 센터 계산 (center_images[0]은 이미 bool+device)
        center_mask = in_mask & center_images[0]
        if center_mask.sum().eq(1):
            center = xym_s[center_mask.expand_as(xym_s)].view(2, 1, 1)
        else:
            xy_in2 = xym_s[in_mask.expand_as(xym_s)].view(2, -1)
            center = xy_in2.mean(1, keepdim=True).view(2, 1, 1)        # (2,1,1)

        center_x[id_int] = degrid(center[0], grid_x - 1, pixel_x)
        center_y[id_int] = degrid(center[1], grid_y - 1, pixel_y)

        # 8) sigma (n_sigma, H, W) 에서 마스크 평균
        # in_mask: (H,W) → (1,H,W)로 확장해도 되고, expand_as로 브로드캐스트
        s = sigma[in_mask.expand_as(sigma)].view(n_sigma, -1).mean(1)  # (n_sigma,)
        s = torch.exp(s * 10)  # 기존 스케일 유지
        sigma_x_tmp = 0.5 / s[0]
        sigma_y_tmp = 0.5 / s[1]
        sigma_x[id_int] = degrid(torch.sqrt(sigma_x_tmp), grid_x - 1, pixel_x)
        sigma_y[id_int] = degrid(torch.sqrt(sigma_y_tmp), grid_y - 1, pixel_y)

        # 9) 색상 (시각화용)
        color_sample_dic[id_int]    = color_sample[int(id_int % 10)]
        color_embedding_dic[id_int] = color_embedding[int(id_int % 10)]

    return (center_x, center_y, samples_x, samples_y,
            sample_spatial_embedding_x, sample_spatial_embedding_y,
            sigma_x, sigma_y, color_sample_dic, color_embedding_dic)


def prepare_embedding_for_test_image(instance_map, output, grid_x, grid_y, pixel_x, pixel_y, predictions, n_sigma):
    instance_ids = instance_map.unique()
    instance_ids = instance_ids[instance_ids != 0]

    xm = torch.linspace(0, pixel_x, grid_x).view(1, 1, -1).expand(1, grid_y, grid_x)
    ym = torch.linspace(0, pixel_y, grid_y).view(1, -1, 1).expand(1, grid_y, grid_x)
    xym = torch.cat((xm, ym), 0)
    height, width = instance_map.size(0), instance_map.size(1)
    xym_s = xym[:, 0:height, 0:width].contiguous()
    spatial_emb = torch.tanh(output[0, 0:2]).cpu() + xym_s
    sigma = output[0, 2:2 + n_sigma]
    color_sample = sns.color_palette("dark")
    color_embedding = sns.color_palette("bright")
    color_sample_dic = {}
    color_embedding_dic = {}
    samples_x = {}
    samples_y = {}
    sample_spatial_embedding_x = {}
    sample_spatial_embedding_y = {}
    center_x = {}
    center_y = {}
    sigma_x = {}
    sigma_y = {}
    for id in instance_ids:
        in_mask = instance_map.eq(id)
        xy_in = xym_s[in_mask.expand_as(xym_s)].view(2, -1)  # 2 N
        perm = torch.randperm(xy_in.size(1))
        idx = perm[:5]
        samples = xy_in[:, idx]
        samples_x[id.item()] = add_samples(samples, 0, grid_x - 1, pixel_x)
        samples_y[id.item()] = add_samples(samples, 1, grid_y - 1, pixel_y)

        # embeddings
        spatial_emb_in = spatial_emb[in_mask.expand_as(spatial_emb)].view(2, -1)
        samples_spatial_embeddings = spatial_emb_in[:, idx]

        sample_spatial_embedding_x[id.item()] = add_samples(samples_spatial_embeddings, 0, grid_x - 1, pixel_x)
        sample_spatial_embedding_y[id.item()] = add_samples(samples_spatial_embeddings, 1, grid_y - 1, pixel_y)
        center_image = predictions[id.item() - 1]['center-image']  # predictions is a list!
        center_mask = in_mask & center_image.byte()


        if (center_mask.sum().eq(1)):
            center = xym_s[center_mask.expand_as(xym_s)].view(2, 1, 1)
        else:
            xy_in = xym_s[in_mask.expand_as(xym_s)].view(2, -1)
            center = xy_in.mean(1).view(2, 1, 1)  # 2 x 1 x 1

        center_x[id.item()] = degrid(center[0], grid_x - 1, pixel_x)
        center_y[id.item()] = degrid(center[1], grid_y - 1, pixel_y)

        # sigma
        s = sigma[in_mask.expand_as(sigma)].view(n_sigma, -1).mean(1)
        
        #print('s is', output[0, 2],)
        s = torch.exp(s * 10)
        #print('s is', s,)
        sigma_x_tmp = 0.5 / s[0]
        sigma_y_tmp = 0.5 / s[1]
        
        
        #print(n_sigma)
        #print(s, sigma_x_tmp, sigma_y_tmp)
        
        sigma_x[id.item()] = degrid(torch.sqrt(sigma_x_tmp), grid_x - 1, pixel_x)
        sigma_y[id.item()] = degrid(torch.sqrt(sigma_y_tmp), grid_y - 1, pixel_y)

        # colors
        color_sample_dic[id.item()] = color_sample[int(id % 10)]
        color_embedding_dic[id.item()] = color_embedding[int(id % 10)]

    return center_x, center_y, samples_x, samples_y, sample_spatial_embedding_x, \
           sample_spatial_embedding_y, sigma_x, sigma_y, color_sample_dic, color_embedding_dic
