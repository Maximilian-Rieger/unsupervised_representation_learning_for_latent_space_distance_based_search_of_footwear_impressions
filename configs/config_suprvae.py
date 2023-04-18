class Config:
    @staticmethod
    def train_regression():
        datasets_dir = "C:\\Users\\Maxim\\Documents\\Uni\\Bachelorarbeit\\datasets"

        return {
            "log_dir": "S:\\Data\\pytorch-utils\\impress-suprvae",
            "name": "Impress",
            "epochs": 500,

            'batchsize': 128,

            'img_size': 128,
            'channels': 1,

            'model': {
                'latent_size': 256,
                'encoder_depth': 4,
                'decoder_depth': 4,
                'residual_enc': True,
                'residual_dec': True,
                'start_filter_enc': 8,
                'start_pixels_dec': 8,
            },

            'optimizer': {
                'lr': 0.001,
                'step_size': 500,
                'beta1': 0.99,
                'beta2': 0.999,
                'gamma': 0.5
            },

            "training": {
                'sample_interval': 2,
                'sample_latentspace': True,
                'n_checkpoints': 2,
                'grad_vis': True,

                "data": {"base": datasets_dir, "dataset": "Impress_2", "set": "prescaled_wcj", 'cache': True, "shuffle_data": True, 'return_path': False},
            },

            "validation": {
                'sample_interval': 2,
                'sample_latentspace': True,
                'n_checkpoints': 2,
                'grad_vis': True,

                "data": {"base": datasets_dir, "dataset": "Impress_2", "set": "clean", 'cache': True, 'return_path': True},
            },
        }

    def __call__(self):
        return self.train_regression()


config__ = Config()
