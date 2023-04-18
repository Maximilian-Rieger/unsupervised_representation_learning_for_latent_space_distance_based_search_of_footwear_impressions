class Config:
    @staticmethod
    def train_regression():
        datasets_dir = "C:\\Users\\Maxim\\Documents\\Uni\\Bachelorarbeit\\datasets"

        return {
            "log_dir": "~\\pytorch-utils\\impress",
            "name": "Impress",
            "epochs": 200,

            'batchsize': 32,

            'model': {
                'in_chan': 1,
                'h_dim': 64,
                'res_h_dim': 32,
                'n_res_layers': 3,
                'n_embeddings': 64,
                'embedding_dim': 64,
                'beta': 0.25,
                'save_img_embedding_map': False,
                'batch_norm': True
            },

            "training": {
                'worker': 16,
                'sample_interval': 25,
                'n_checkpoints': 2,
                'grad_vis': True,
                "data": {"base": datasets_dir, "dataset": "Impress_soles_full", "set": "train"},
            },

            'optimizer': {
                'lr': 0.001,
                'step_size': 30,
                'beta1': 0.95,
                'beta2': 0.999,
                'gamma': 0.5
            },

        }

    def __call__(self):
        return self.train_regression()


config__ = Config()
