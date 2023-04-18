class Config:
    @staticmethod
    def train_regression():
        datasets_dir = "C:\\Users\\Maxim\\Documents\\Uni\\Bachelorarbeit\\datasets"

        return {
            "log_dir": "S:\\Data\\pytorch-utils\\impress",

            "name": "Impress",
            "epochs": 330,

            'batchsize': 16,

            'model': {
                'input_channels': 1,
                'num_freq_bands': 4,
                'encoder_depth': 4,
                'decoder_depth': 4,
                'max_freq': 256,
                'freq_base': 2,
                'input_axis': 2,
                'num_latents': 128,
                'latent_dim': 256,
                'cross_heads': 1,
                'latent_heads': 16,
                'cross_dim_head': 16,
                'latent_dim_head': 16,
                'attn_dropout': 0.,
                'ff_dropout': 0.,
                'weight_tie_layers': [False, True, True, False]
            },

            "training": {
                'worker': 0,
                'sample_interval': 2,
                # 'sample_interval': 400,
                'n_checkpoints': 2,
                'grad_vis': True,

                # "data": {"base": datasets_dir, "dataset": "Impress_soles_full", "set": "train"},
                # "data": {"base": datasets_dir, "dataset": "Impress_soles_prescaled_2", "set": "train", 'limit': 128},
                # "data": {"base": datasets_dir, "dataset": "Impress_2", "set": "prescaled-S", 'cache': True, "shuffle_data": True},
                "data": {"base": datasets_dir, "dataset": "Impress_2", "set": "prescaled_wcj", 'cache': True, "shuffle_data": True, 'return_path': False},
                # "data": {"base": datasets_dir, "dataset": "CelebA", "set": "train"},
            },

            "validation": {
                'worker': 0,
                'sample_interval': 1.5,
                "data": {"base": datasets_dir, "dataset": "Impress_2", "set": "clean", 'cache': True, 'return_path': True},
            },

            'optimizer': {
                'lr': 0.001,
                'step_size': 800,
                'beta1': 0.9,
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
