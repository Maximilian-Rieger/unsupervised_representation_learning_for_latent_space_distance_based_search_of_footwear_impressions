class Config:
    @staticmethod
    def train_regression():
        datasets_dir = "~/datasets"
        # datasets_dir = "/data/public"

        return {
            "log_dir": "~/pytorch-utils/impress",
            "name": "Impress",
            "epochs": 10,

            'batchsize': 11,

            'img_size': 256,
            'channels': 1,

            'model': {
                'latent_size': 200,
                'encoder_depth': 4,
                'decoder_depth': 4,
                'discriminator_depth': 3,
                'residual_enc': True,
                'residual_dec': True,
                'start_filter_enc': 32,
                'start_pixels_dec': 16,
            },

            "training": {
                'sample_interval': 20,
                'sample_latentspace': True,
                'n_checkpoints': 2,
                'grad_vis': True,

                "data" : { "base": datasets_dir, "dataset": "Impress_soles_full", "set": "train" },
            },

            'optimizer': {
                'lr': 0.001,
                'step_size': 25,
                'beta1': 0.99,
                'beta2': 0.999,
                'gamma': 0.5
            },

        }

    def __call__(self):
        return self.train_regression()


config__ = Config()
