import math


class Config:
    @staticmethod
    def train_regression():
        datasets_dir = "C:\\Users\\Maxim\\Documents\\Uni\\Bachelorarbeit\\datasets"

        return {
            "log_dir": "S:\\Data\\pytorch-utils\\impress-autoreg",
            # "resume": "Impress-2021-07-21-01-09",
            "name": "Impress",
            "epochs": 400,

            'batchsize': 32,

            'model': {
                'input_channels': 1,
                'num_freq_bands': 5,
                'encoder_depth': 3,
                'decoder_depth': 3,
                'max_freq': 256,
                'freq_base': 2,
                'input_axis': 2,
                'num_latents': 64,
                'latent_dim': 256,
                'cross_heads': 1,
                'latent_heads': 16,
                'cross_dim_head': 16,
                'latent_dim_head': 32,
                'attn_dropout': 0.,
                'ff_dropout': 0.,
                'weight_tie_layers': False
            },

            "training": {
                'worker': 0,
                'sample_interval': 0.75,
                # 'sample_interval': 400,
                'n_checkpoints': 2,
                'grad_vis': True,

                # "data": {"base": datasets_dir, "dataset": "Impress_2", "set": "prescaled", 'cache': True, "shuffle_data": True, 'limit': 64},
                "data": {"base": datasets_dir, "dataset": "Impress_2", "set": "clean", 'cache': True, "shuffle_data": True},
                'pos_range': 8,
                'offset_range': 2,
            },

            "validation": {
                'worker': 0,
                'sample_interval': 0.75,
                "data": {"base": datasets_dir, "dataset": "Impress_2", "set": "clean", 'cache': True, 'return_path': True},
                # "data": {"base": datasets_dir, "dataset": "Impress_2", "set": "clean", 'cache': True},
            },

            'optimizer': {
                'lr': 0.000125,
                'step_size': 100,
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
