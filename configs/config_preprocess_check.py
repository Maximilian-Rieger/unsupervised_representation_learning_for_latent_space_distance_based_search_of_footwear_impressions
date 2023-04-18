class Config:
    @staticmethod
    def train_regression():
        datasets_dir = "C:\\Users\\Maxim\\Documents\\Uni\\Bachelorarbeit\\datasets"

        return {
            "log_dir": "S:\\Data\\pytorch-utils\\checks",

            'batchsize': 16,

            'img_size': (128, 128),

            "training": {
                'worker': 0,
                'sample_interval': 2,
                # 'sample_interval': 400,
                'n_checkpoints': 2,
                'grad_vis': True,

                "data": {"base": datasets_dir, "dataset": "Impress_2", "set": "prescaled_wcj", 'cache': True, "shuffle_data": True, 'return_path': False},
            },

            "validation": {
                'worker': 0,
                'sample_interval': 1.5,
                "data": {"base": datasets_dir, "dataset": "Impress_2", "set": "clean", 'cache': True, 'return_path': False},
            },
        }

    def __call__(self):
        args = self.train_regression()
        return args


config__ = Config()
