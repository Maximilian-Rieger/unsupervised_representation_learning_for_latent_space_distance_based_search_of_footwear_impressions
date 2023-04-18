class Config:
    @staticmethod
    def train_regression():
        datasets_dir = "C:\\Users\\Maxim\\Documents\\Uni\\Bachelorarbeit\\datasets"

        return {
            "log_dir": "S:\\Data\\pytorch-utils\\impress-ae",
            "name": "Impress-AE",
            "epochs": 200,

            'batchsize': 128,

            'img_size': (128, 128),
            'channels': 1,

            'model': {
                'latent_size': 256,
            },

            "training": {
                'worker': 0,
                'sample_interval': 2,
                # 'sample_interval': 400,
                'n_checkpoints': 2,
                'grad_vis': True,

                "data": {"base": datasets_dir, "dataset": "Impress_2", "set": "prescaled_wcj", 'cache': True, "shuffle_data": True, 'return_path': False},
            },

            "validation": {
                'worker': 0,
                'sample_interval': 1.5,
                "data": {"base": datasets_dir, "dataset": "Impress_2", "set": "clean", 'cache': True, 'return_path': True},
            },

            'optimizer': {
                'lr': 0.01,
                'step_size': 200,
                'beta1': 0.9,
                'beta2': 0.999,
                # 'beta2': 0.,
                'gamma': 0.5
            },

        }

    def __call__(self):
        return self.train_regression()


config__ = Config()
