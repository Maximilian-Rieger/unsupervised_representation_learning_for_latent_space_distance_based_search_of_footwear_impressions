class Config:
    def __call__(self):
        datasets_dir = "C:\\Users\\Maxim\\Documents\\Uni\\Bachelorarbeit\\datasets"
        reductionFactore = 0.2
        # img_size = 512
        img_size = 256
        step_factor = 1 / 4
        return {
            "log_dir": "~\\pytorch-utils\\impress",
            "name": "Impress_soles_full",
            "epochs": 15,
            # "no_cuda": True,
            "cuda": True,
            'batchsize': 32,

            'img_size': img_size,
            'img_shape': (img_size, img_size),
            'step': int(img_size * step_factor),
            'channels': 1,

            "training": {
                'shuffle': True,
                # "data": {"base": datasets_dir, "dataset": "Impress_soles_full", "set": "train", 'limit': 332, 'cache': True, 'return_path': True}, # Train
                # "data": {"base": datasets_dir, "dataset": "Impress_soles_full", "set": "train", 'offset': 332, 'cache': True, 'return_path': True}, # Val
                "data": {"base": datasets_dir, "dataset": "Impress_cleaned", "set": "train", 'cache': False, 'return_path': True, 'shuffle_data': True}, # preAugment Train
                # "data": {"base": datasets_dir, "dataset": "Impress_cleaned", "set": "val", 'cache': False, 'return_path': True, 'shuffle_data': True}, # preAugment Val
            },
        }


config__ = Config()
