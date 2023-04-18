class Config:
    def __call__(self):
        return {
            "log_dir": "~/pytorch-utils/impress",
            "name": "Impress_aae",
            'cuda': True,
            'batchsize': 4,

            'latent_size': 320,
            'img_size': 256,
            'channels': 1,

            "search": {
                "data": {"base": "~/datasets", "dataset": "Impress_soles_full", "set": "train"},
            },

        }


config__ = Config()
