class Config:
    @staticmethod
    def train_regression():
        datasets_dir = "C:\\Users\\Maxim\\Documents\\Uni\\Bachelorarbeit\\datasets"

        return {
            "log_dir": "S:\\Data\\pytorch-utils\\impress-vqvae",
            "name": "Impress",
            "epochs": 500,

            'batchsize': 512,

            'model': {
                "in_chan": 1,
                "h_dim": 128,
                "res_h_dim": 64,
                "n_res_layers": 4,
                "n_embeddings": 512,
                "embedding_dim": 128,
                "beta": 0.25,
                "save_img_embedding_map": False,
                "batch_norm": True
            },

            "training": {
                'worker': 0,
                'sample_interval': 2,
                'n_checkpoints': 2,
                'grad_vis': True,

                "data": {"base": datasets_dir, "dataset": "Impress_2", "set": "prescaled_csj", 'cache': True, "shuffle_data": True},
            },

            "validation": {
                'worker': 0,
                'sample_interval': 1.5,
                "data": {"base": datasets_dir, "dataset": "Impress_2", "set": "clean", 'cache': True, 'return_path': True},
            },

            'optimizer': {
                'lr': 0.005,
                'step_size': 250,
                'beta1': 0.95,
                'beta2': 0.999,
                'gamma': 0.5
            },

        }

    def __call__(self):
        return self.train_regression()


config__ = Config()
