class Config:
    @staticmethod
    def train_regression():
        datasets_dir = "~/datasets"

        return {
            "log_dir": "~/pytorch-utils/impress",
            "name": "Impress",
            # "epochs": 200,
            "epochs": 20,

            'batchsize': 15,

            'img_size': 256,
            'latent_size': 256,
            'channels': 1,

            "training": {
                'sample_interval': 5000,
                'n_checkpoints': 1,
                'log_image_info': True,

                # 'grad_vis': True,

                "data": {"base": datasets_dir, "dataset": "Patches_512", "set": "training"},
            },

            "validation": {
                'log_image_info': True,
                'sample_interval': 500,
                "data": {"base": datasets_dir, "dataset": "Patches_512", "set": "validation"},
            },

        }

    def __call__(self):
        return self.train_regression()


config__ = Config()
