class Config:
    @staticmethod
    def train_regression():
        datasets_dir = "/data/public"

        return {
            "log_dir": "~/pytorch-utils/impress",
            "name": "Impress",
            "epochs": 5,

            'batchsize': 2,

            'in': 224,
            'out': 224,

            "training" : {
                "data" : { "base": datasets_dir, "dataset": "Impress", "set": "train" },
                'n_checkpoints': 1,
            },

            "validation": {
                "data": { "base": datasets_dir, "dataset": "Impress", "set": "val" },
                "plot": True,
            }

        }

    def __call__(self):
        return self.train_regression()


config__ = Config()
