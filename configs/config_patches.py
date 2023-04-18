class Config:
    def __call__(self):
        datasets_dir = "C:\\Users\\Maxim\\Documents\\Uni\\Bachelorarbeit\\datasets"
        reductionFactore = 0.2
        img_size = 256
        step_factor = 1 / 4
        return {
            "log_dir": "~\\pytorch-utils\\impress",
            "name": "Impress_patches_extended",
            "epochs": 1,
            # "no_cuda": True,
            'batchsize': 1,

            # 'img_pre_size': (5100, 8400),
            'img_pre_size': (int(5100 * reductionFactore), int(8400 * reductionFactore)),
            'img_size': img_size,
            'step': int(img_size * step_factor),
            'channels': 1,
            'threshold': (4 / 255, 251 / 255),

            "training": {
                "data": {"base": datasets_dir, "dataset": "Impress_soles_extended", "set": "train"},
            },
        }


config__ = Config()
