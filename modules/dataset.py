import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class InpaintDataset(Dataset):
    def __init__(
        self,
        data_folder='/home/hannah/Documents/Thesis/data/preprocessed_structured3D',
        # data_folder = '/project/pjsminha/data/preprocessed_structured3D',
        split = 'train'
    ):
        self.data_folder = data_folder
        self.split = split
        if split == 'train':
            file = f'{data_folder}/2xfiltered_{split}_inds.txt'
        elif split == 'val' or 'test':
            file = f'{data_folder}/2xfiltered_{split}_inds_3000.txt'

        # file = f'{data_folder}/{split}_inds.txt'

        self.inds = np.loadtxt(file, dtype=str)

        self.num_images = len(self.inds)

    def __getitem__(self, index):
        """
                Retrieve color depth and semantic from folder and resize.
        """
        i = self.inds[index]
        # i = str(i).zfill(6)

        rgb_path = f'{self.data_folder}/{self.split}/color/{i}_image.npy'
        rgb = np.load(rgb_path)[2:250, 2:250, :3] # filter outline edge

        depth_path = f'{self.data_folder}/{self.split}/depth/{i}_depth.npy'
        depth = np.load(depth_path)[2:250, 2:250] # filter outline edge

        mask = self.generate_masks(248)

        edges_path = f'{self.data_folder}/{self.split}/gray_edges/{i}_gray_edges.npy'
        edges = np.load(edges_path)[2:250, 2:250] # filter outline edge


        img_object = {
            'rgb': torch.Tensor(rgb).permute(2, 0, 1),
            'depth': torch.Tensor(depth).unsqueeze(0),
            'mask': torch.Tensor(mask).unsqueeze(0),
            'edges': torch.Tensor(edges).unsqueeze(0)
        }

        return img_object


    def generate_masks(self, img_size):
        """
                Create mask with box in random location.
                Create another slightly bigger mask in the same location.
        """
        H, W = img_size, img_size
        mask = torch.zeros((H, W))
        box_size = round(H * 0.3)

        x_loc = np.random.randint(0, W - box_size)
        y_loc = np.random.randint(0, H - box_size)

        mask[y_loc:y_loc+box_size, x_loc:x_loc+box_size] = 1

        return mask

        

    def __len__(self):
        """
                Return the size of the dataset.
        """
        return self.num_images


if __name__ == '__main__':

    dataset = InpaintDataset(split='train')

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    count = 0

    # for i, batch in enumerate(loader):
        # mean = batch['rgb'].mean()
        # if mean < 0.15:
        #     count += 1
        #     plt.title(str(mean))
        #     plt.imshow(batch['rgb'][0].permute(1, 2, 0))
        #     plt.show()
    # print(count)
        # Check if average value higher than 0.9 to filter overbelicht plaatjes

        # print('rgb')
        # print('min: ', batch['rgb'].min(), 'max:  ', batch['rgb'].max(), 'shape: ', batch['rgb'].shape)
        # print('depth')
        # print('min: ', batch['depth'].min(), 'depth: ', batch['depth'].max(), 'shape: ', batch['depth'].shape)
        # print(batch['mask'].shape)
        # print(batch['edges'].shape)
        # i = 0
        # depth = batch['depth'][i, 0]
        # mask = batch['mask'][i, 0]
        # color = batch['rgb'][i].permute(1, 2, 0)
        # edges = batch['edges'][i, 0]
        # masked_depth = depth * (1 - mask)

        # plt.imshow(masked_depth, cmap='viridis')
        # plt.axis('off')
        # plt.savefig('masked depth.png', bbox_inches='tight')

        # plt.imshow(depth, cmap='viridis')
        # plt.axis('off')
        # plt.savefig('depth.png', bbox_inches='tight')
        

        # plt.imshow(color)
        # plt.axis('off')
        # plt.savefig('color.png', bbox_inches='tight')

        # plt.imshow(edges, cmap='gray')
        # plt.axis('off')
        # plt.show()
        # plt.savefig('edges.png', bbox_inches='tight')

        # fig, ax = plt.subplots(1, 4)
        # ax[0].imshow(batch['depth'][0, 0], cmap='viridis')
        # ax[1].imshow(batch['rgb'][0].permute(1, 2, 0))
        # ax[2].imshow(batch['mask'][0, 0], cmap='gray')
        # ax[3].imshow(batch['edges'][0, 0] ,cmap='gray')
        # plt.show()
        # break