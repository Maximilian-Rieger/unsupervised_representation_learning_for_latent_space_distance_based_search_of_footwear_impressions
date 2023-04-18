class Config:
    @staticmethod
    def train_regression():
        datasets_dir = "~/datasets"

        return {
            "log_dir": "~/pytorch-utils/impress",
            "name": "Impress_aae",
            "epochs": 20,

            'cuda': True,
            'batchsize': 4,

            'latent_size': 320,
            'img_size': 256,
            'channels': 1,

            "training": {
                "data": {"base": datasets_dir, "dataset": "Impress_soles_full", "set": "train"},
                'n_checkpoints': 1,
                'sample_interval': 50,
            },

        }

    def __call__(self):
        return self.train_regression()


config__ = Config()
