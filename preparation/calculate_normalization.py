import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from experiment.data_zoo import DataZoo


def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for data in tqdm(loader, "Computing mean/std", len(loader), unit="samples"):
        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    data = DataZoo.get("C:\\Users\\Maxim\\Documents\\Uni\\Bachelorarbeit\\datasets", "Patches_extended", "training",
                       pattern={'patches': 'patch_*.png'},
                       shared_pattern='training',
                       transform=transform)
    print(f"Found: {len(data)}")

    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=1)

    mean, std = online_mean_and_sd(dataloader)
    print(f"Mean: {mean} | Std: {std}")
