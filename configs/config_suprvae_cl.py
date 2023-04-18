class Config:
    @staticmethod
    def train_regression():
        datasets_dir = "C:\\Users\\Maxim\\Documents\\Uni\\Bachelorarbeit\\datasets"


        return {
            "log_dir": "~\\pytorch-utils\\impress",
            "name": "Impress",
            "epochs": 200,

            'batchsize': 42,

            'img_size': 256,
            'channels': 1,

            'model': {
                'latent_size': 256,
                'encoder_depth': 3,
                'decoder_depth': 3,
                'discriminator_depth': 3,
                'residual_enc': True,
                'residual_dec': True,
                'start_filter_enc': 16,
                'start_pixels_dec': 32,
            },

            "training": {
                'sample_interval': 20,
                'sample_latentspace': True,
                'n_checkpoints': 2,
                'grad_vis': True,

                "data": {"base": datasets_dir, "dataset": "Impress_soles_full", "set": "train"},
            },

            'optimizer': {
                'lr': 0.0001,
                'step_size': 75,
                'beta1': 0.99,
                'beta2': 0.999,
                'gamma': 0.5
            },

        }

    def __call__(self):
        return self.train_regression()


config__ = Config()
