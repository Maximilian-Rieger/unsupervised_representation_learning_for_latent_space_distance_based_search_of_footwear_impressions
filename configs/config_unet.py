class Config:
    @staticmethod
    def train_regression():
        datasets_dir = "/data/public"

        return {
            "log_dir": "~/pytorch-utils/impress",
            "name": "Impress",
            "epochs": 10,

            'batchsize': 2,

            'img_size': 256,
            'channels': 3,

            "training": {
                "data": {"base": datasets_dir, "dataset": "Impress_registrered", "set": "train"},
                'n_checkpoints': 1,
                'sample_interval': 50,
            },

        }

    def __call__(self):
        return self.train_regression()


config__ = Config()
