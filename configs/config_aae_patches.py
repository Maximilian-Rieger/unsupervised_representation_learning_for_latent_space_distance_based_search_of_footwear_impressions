class Config:
    @staticmethod
    def train_regression():
        datasets_dir = "C:\\Users\\Maxim\\Documents\\Uni\\Bachelorarbeit\\datasets"

        return {
            "log_dir": "~\\pytorch-utils\\impress",
            "name": "Impress",
            # "epochs": 200,
            "epochs": 50,

            'batchsize': 23,

            'img_size': 256,
            'latent_size': 256,
            'channels': 1,

            'optimizer': {
                'lr': (0.0001, 0.01),
                'step_size': 25,
                'beta1': 0.9,
                'beta2': 0.99,
                'gamma': 0.1
            },

            "training": {
                # 'sample_interval': 1250,
                'sample_interval': 500,
                'n_checkpoints': 1,
                'log_image_info': True,

                # 'grad_vis': True,

                "data": {
                    "base": datasets_dir,
                    "dataset": "Patches_extended",
                    "set": "training",
                    'pattern': {'patches': 'patch_*.png'},
                    'shared_pattern': 'training',
                    'limit': 27600,
                },
            },

            # "validation": {
            #     'log_image_info': True,
            #     'sample_interval': 100,
            #     "data": {
            #         "base": datasets_dir,
            #         "dataset": "Patches_extended",
            #         "set": "validation",
            #         'pattern': {'patches': 'patch_*.png'},
            #         'shared_pattern': 'validation',
            #     },
            # },

        }

    def __call__(self):
        return self.train_regression()


config__ = Config()
