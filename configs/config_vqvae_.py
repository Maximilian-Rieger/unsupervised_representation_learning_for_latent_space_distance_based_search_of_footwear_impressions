class Config:
    @staticmethod
    def train_regression():
        datasets_dir = "C:\\Users\\Maxim\\Documents\\Uni\\Bachelorarbeit\\datasets"

        return {
            "log_dir": "S:\\Data\\pytorch-utils\\impress-vqvae",
            "name": "Impress",
            "epochs": 500,

            'batchsize': 64,

            'model': {
                'in_chan': 1,
                'h_dim': 128,
                'res_h_dim': 64,
                'n_res_layers': 6,
                'n_embeddings': 128,
                'embedding_dim': 128,
                'beta': 0.25,
                'save_img_embedding_map': False,
                'batch_norm': True
            },

            "training": {
                'worker': 0,
                'sample_interval': 1,
                'n_checkpoints': 2,
                'grad_vis': True,
                # "data": {"base": datasets_dir, "dataset": "Impress_soles_full", "set": "train"},
                "data": {"base": datasets_dir, "dataset": "Impress_2", "set": "prescaled_csj", 'cache': True, "shuffle_data": True},
            },

            'optimizer': {
                'lr': 0.001,
                'step_size': 500,
                'beta1': 0.99,
                'beta2': 0.999,
                'gamma': 0.5
            },

        }

    def __call__(self):
        return self.train_regression()


config__ = Config()
