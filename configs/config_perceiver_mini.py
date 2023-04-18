import math


class Config:
    @staticmethod
    def train_regression():
        datasets_dir = "C:\\Users\\Maxim\\Documents\\Uni\\Bachelorarbeit\\datasets"

        return {
            "log_dir": "S:\\Data\\pytorch-utils\\impress",
            "name": "Impress",
            "epochs": 200,

            'batchsize': 4,

            'model': {
                'input_channels': 3,
                'num_freq_bands': 6,
                'encoder_depth': 4,
                'decoder_depth': 1,
                'max_freq': 218,
                'freq_base': 2,
                'input_axis': 2,
                'num_latents': 128,
                'cross_dim': 512,
                'latent_dim': 256,
                'cross_heads': 1,
                'latent_heads': 8,
                'cross_dim_head': 32,
                'latent_dim_head': 32,
                'attn_dropout': 0.,
                'ff_dropout': 0.,
                'weight_tie_layers': False
            },

            "training": {
                'worker': 16,
                'sample_interval': 100,
                # 'sample_interval': 400,
                'n_checkpoints': 2,
                'grad_vis': True,
                # "data": {"base": datasets_dir, "dataset": "Impress_soles_full", "set": "train"},
                # "data": {"base": datasets_dir, "dataset": "Impress_soles_prescaled", "set": "train"},
                "data": {"base": datasets_dir, "dataset": "CelebA", "set": "train"},
            },

            'optimizer': {
                'lr': 0.001,
                'step_size': 30,
                'beta1': 0.95,
                'beta2': 0.999,
                # 'beta2': 0.,
                'gamma': 0.5
            },

        }

    def __call__(self):
        args = self.train_regression()
        # args['training']['sample_interval'] = math.floor(args['training']['sample_interval'] / args['batchsize'])
        return args


config__ = Config()
