import logging
import time
import os
import sys
import shutil
import json
import datetime
from tensorboardX import SummaryWriter
import torch
import collections.abc
import numpy as np
import pickle
import mlflow
import socket
import git
import matplotlib as mpl
from importlib import import_module

mpl.use('Agg')
import matplotlib.pyplot as plt


class TensorboardLogger(object):
    def __init__(self, log_dir, modules=None, images_dir=None):
        self.log_dir = self._prepare_log_dir(log_dir, modules, images_dir=images_dir)

        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.global_step = 0
        self.epoch = None
        self.running_avgs = {}
        self.epoch_avgs = {}
        self.embeddings = {}

    def log_graph(self, model, input=None):
        self.writer.add_graph(model, input)

    def log_value(self, name, value, step=None):
        if step is None:
            step = self.global_step
        self.writer.add_scalar(name, value, step)

        return self

    def log_value_and_running_avg(self, name, value):
        self.writer.add_scalar(name, value, self.global_step)
        if name in self.running_avgs:
            running_avg = (self.running_avgs[name] + value) * 0.5
            self.running_avgs.update({name: running_avg})
        else:
            self.running_avgs.update({name: value})

        self.writer.add_scalar(f"{name}_running_avg", self.running_avgs[name], self.global_step)

        return self

    def log_value_and_epoch_avg(self, name, value, epoch, batch_size, data_len):
        """
            Logs the value and the average of the epoch
        :param name: name of the value
        :param value: value to log
        :param epoch: epoch number
        :param batch_size: batch size
        :param data_len: length of the data
        :return:
        """
        self.writer.add_scalar(name, value, self.global_step)
        if name in self.epoch_avgs:
            prev_epoch, prev_step, epoch_value = self.epoch_avgs[name]
            if prev_epoch != epoch:
                epoch_value /= data_len
                self.writer.add_scalar(f"{name}_epoch", epoch_value, prev_step + batch_size)
                epoch_value = 0

            epoch_value += value * batch_size
            self.epoch_avgs.update({name: [prev_epoch, self.global_step, epoch_value]})
        else:
            self.epoch_avgs.update({name: [epoch, self.global_step, value * batch_size]})

        return self

    def clear_epoch_loss(self, name):
        """
            Clears the epoch loss
            :param name: name of the loss
        """
        del self.epoch_avgs[name]

    def get_epoch_loss(self, name: str, data_len: int):
        """
            Returns the average loss of the epoch

            :param name: name of the loss
            :param data_len: length of the data
            :returns: average loss of the epoch
        """
        if name in self.epoch_avgs:
            prev_epoch, prev_step, epoch_value = self.epoch_avgs[name]
            epoch_value /= data_len
            return epoch_value

    def add_figure(self, tag, figure, close=True, walltime=None):
        self.writer.add_figure(tag, figure, global_step=self.global_step, close=close, walltime=walltime)
        return self

    def log_embedding(self, features, labels, labels_header=None, images=None, step=None, name='default'):
        if step is None:
            step = self.global_step

        if images is not None:
            images = images.clone().detach()

            for k, img in enumerate(images):
                img = (img - img.min()) / (img.max() - img.min())
                images[k] = img

        self.writer.add_embedding(torch.Tensor(features), labels, images, step, tag=name, metadata_header=labels_header)
        return self

    def accumulate_embedding_set_for_epoch(self, features, labels, labels_header=None, images=None, name='default', del_after_epoch=False):
        if name in self.embeddings:
            prev_features, prev_labels, prev_labels_header, prev_images = self.embeddings[name]
            prev_features = torch.cat([prev_features, features])
            if labels is not None:
                prev_labels = torch.cat([prev_labels, labels])
            if labels_header is not None:
                prev_labels_header = torch.cat([prev_labels_header, labels])
            if prev_images is not None:
                prev_images = torch.cat([prev_images, images])
            self.embeddings.update({name: [prev_features, prev_labels, prev_labels_header, prev_images]})
        else:
            self.embeddings.update({name: [features, labels, labels_header, images]})

    def get_embedding_set(self, name='default'):
        if name in self.embeddings:
            return self.embeddings[name]

    def log_embedding_set(self, name='default', step=None):
        if name in self.embeddings:
            prev_features, prev_labels, prev_labels_header, prev_images = self.embeddings[name]
            self.log_embedding(prev_features, prev_labels, labels_header=prev_labels_header, images=prev_images, name=name, step=step)
            del self.embeddings[name]

    def clear_embedding_set(self, name='default'):
        if name in self.embeddings:
            del self.embeddings[name]

    def step(self, step=1):
        self.global_step += step

    def log_options(self, options, changes=None):
        if type(options) != dict:
            options = options.__dict__

        options['hash'] = json_hash(options)

        with open(os.path.join(self.log_dir, 'options.json'), 'w') as fp:
            json.dump(options, fp)

        if changes:
            with open(os.path.join(self.log_dir, 'changes.json'), 'w') as fp:
                json.dump(changes, fp)

    def log_dict(self, dict_to_log, prefix=None, suffix=None, stdout=True, step=None):
        for k, v in dict_to_log.items():
            name = '-'.join(filter(None, [prefix, k, suffix]))
            self.log_value(name, v, step)
            if stdout:
                logging.info('{} {:5f}'.format(name, v))

    def add_pr_curve_from_dict_list(self, dict_list, step=None, name='ROC'):
        if not step and self.epoch:
            suffix = self.epoch
        elif not step:
            suffix = self.global_step
        else:
            suffix = step

        with open(os.path.join(self.log_dir, 'pr-curve-{}.pkl'.format(suffix)), 'wb') as fp:
            pickle.dump(dict_list, fp)

        true_positive_counts = [d['true_positives'] for d in dict_list]
        false_positive_counts = [d['false_positives'] for d in dict_list]
        true_negative_counts = [d['true_negatives'] for d in dict_list]
        false_negative_counts = [d['false_negatives'] for d in dict_list]
        precision = [d['precision'] for d in dict_list]
        recall = [d['recall'] for d in dict_list]
        thresh = [d['threshold'] for d in dict_list]

        fig, ax1 = plt.subplots()
        ax1.plot(recall, '-r', label='recall')
        ax1.plot(precision, '-b', label='precision')
        ax1.set_ylabel('precision', color='b')
        ax1.legend()

        ax2 = ax1.twinx()
        ax2.plot(thresh, '-g', label='threshold')
        ax2.set_ylabel('threshold')
        ax2.legend()
        fig.tight_layout()

        #    fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(18.5, 10.5)
        fig.savefig(os.path.join(self.log_dir, 'pr-curve-{}.png'.format(suffix)), dpi=100)
        plt.clf()
        plt.close(fig)

        # fig2, ax3 = plt.subplots()
        # pr = [(recall[i], precision[i]) for i in range(len(thresh))]
        # ax3.plot(pr)
        # ax3.set_ylabel('threshold')
        # ax3.set_yticks(thresh)
        # fig2.set_size_inches(18.5, 10.5)
        # fig2.savefig(os.jsonPath.join(self.log_dir, 'other-pr-curve-{}.png'.format(suffix)), dpi=100)

        recall = np.array(recall)
        recall, uniq_idx = np.unique(recall, return_index=True)
        true_positive_counts = np.array(true_positive_counts)[uniq_idx]
        false_positive_counts = np.array(false_positive_counts)[uniq_idx]
        true_negative_counts = np.array(true_negative_counts)[uniq_idx]
        false_negative_counts = np.array(false_negative_counts)[uniq_idx]
        precision = np.array(precision)[uniq_idx]

        idxs = np.argsort(recall)[::-1]
        true_positive_counts = true_positive_counts[idxs].tolist()
        false_positive_counts = false_positive_counts[idxs].tolist()
        true_negative_counts = true_negative_counts[idxs].tolist()
        false_negative_counts = false_negative_counts[idxs].tolist()
        precision = precision[idxs].tolist()
        recall = recall[idxs].tolist()

        self.add_pr_curve_raw(true_positive_counts, false_positive_counts, true_negative_counts, false_negative_counts,
                              precision, recall, step, name)

    def add_pr_curve_raw(self, true_positive_counts, false_positive_counts, true_negative_counts, false_negative_counts,
                         precision, recall, step=None, name='ROC'):
        if not step:
            step = self.global_step

        num_thresholds = len(true_positive_counts)
        self.writer.add_pr_curve_raw(name, true_positive_counts, false_positive_counts, true_negative_counts,
                                     false_negative_counts, precision, recall, step, num_thresholds,
                                     weights=None)

    def add_pr_curve(self, tag, labels, predictions):
        self.writer.add_pr_curve(tag, labels, predictions, global_step=self.global_step)

    @staticmethod
    def _prepare_log_dir(log_path, save_modules=None, images_dir=None):
        if save_modules is None:
            save_modules = ['__main__']
        else:
            save_modules = ['__main__'] + save_modules

        import datetime
        now = datetime.datetime.now()
        log_path = log_path + '-%d-%02d-%02d-%02d-%02d' % (now.year, now.month, now.day, now.hour, now.minute)
        if os.path.isdir(log_path):
            log_path += '-%02d' % now.second

        exp_log_path = os.path.expanduser(log_path)
        os.mkdir(exp_log_path)
        if images_dir:
            os.mkdir(os.path.join(exp_log_path, 'data'))

        file_handler = logging.FileHandler(os.path.join(log_path, 'log.txt'), mode='w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s '))
        logging.info("writing log file to: %s ", os.path.join(log_path, 'log.txt'))
        logging.getLogger().addHandler(file_handler)

        for module in save_modules:
            shutil.copy(os.path.abspath(sys.modules[module].__file__), log_path)
        return log_path


class TensorboardLoggerWithMLFlow(TensorboardLogger):
    def __init__(self, logdir, experiment_name=None):
        # check this before log directory gets created
        try:
            repo = git.Repo(search_parent_directories=True)
        except git.exc.InvalidGitRepositoryError:
            assert False, 'when using mlflow the project has to be run in a git repo'
        assert not repo.is_dirty(), 'when using mlflow the project has to be in a clean state\n' \
                                    'Directory is dirty - please commit all changes.'
        assert not repo.is_dirty(untracked_files=True), 'when using mlflow the project has to be in a clean state\n' \
                                                        'Directory is dirty - please add all files.'
        self.git_sha = repo.head.object.hexsha

        self._mlflow = None
        self.mlflow_writes = None
        self._experiment_name = experiment_name

        super().__init__(logdir)

    def log_value(self, name, value, step=None):
        super().log_value(name, value, step)

        if self.mlflow:
            last_write = self.mlflow_writes.get(name, None)
            if last_write is None:
                mlflow.log_metric(name, value)
                self.mlflow_writes[name] = datetime.datetime.now()
            else:
                if last_write + datetime.timedelta(minutes=5) < datetime.datetime.now():
                    self.mlflow_writes[name] = datetime.datetime.now()
                    mlflow.log_metric(name, value)

    def log_options(self, options, changes=None):
        super().log_options(options, changes)
        ignore_for_mlflow = ['gpuid', 'log_dir', 'no_cuda', 'cuda', 'test_trainset', 'test_writer',
                             'super_fancy_new_name', 'nfeat']
        if self.mlflow:
            self.log_artifcat(os.path.join(self.log_dir, 'options.json'))
            if changes:
                for k, v in changes:
                    if k not in ignore_for_mlflow:
                        mlflow.log_param(k, v)

    def log_artifcat(self, path):
        if self.mlflow:
            mlflow.log_artifact(path)

    def get_mlflow(self):
        if self._mlflow is None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('edna.cvl.tuwien.ac.at', 5000))

            if result == 0:
                self._mlflow = True
                mlflow.set_tracking_uri('http://edna.cvl.tuwien.ac.at:5000')
                logging.info('starting mlflow run with git commit: {}'.format(self.git_sha))
                exp_id = None
                if self._experiment_name:
                    exps = mlflow.tracking.list_experiments()
                    logging.info('mlflow: searching for existing experiment {}'.format(self._experiment_name))
                    for e in exps:
                        if self._experiment_name == e.name:
                            exp_id = e.experiment_id
                            logging.info('experiment found with id {}'.format(exp_id))
                            break

                    if not exp_id:
                        logging.info('mlflow: experiment {} does not exist ... creating'.format(self._experiment_name))
                        exp_id = mlflow.create_experiment(self._experiment_name)

                mlflow.start_run(source_version=self.git_sha, experiment_id=exp_id)

            else:
                self._mlflow = False
                logging.warning('unable to connect to mlflow server ... logging without it')
            self.mlflow_writes = {}
        return self._mlflow

    mlflow = property(get_mlflow, None)

    def __del__(self):
        if hasattr(self, 'mlflow') and self.mlflow:
            mlflow.end_run()


class GPU:
    device = torch.device('cpu')

    @staticmethod
    def get_free_gpus_linux():
        a = os.popen("/usr/bin/nvidia-smi | grep 'MiB /' | awk -e '{print $9}' | sed -e 's/MiB//'")

        free_memory = []
        while 1:
            line = a.readline()
            if not line:
                break
            free_memory.append(int(line))
        return free_memory

    @staticmethod
    def get_free_gpus_windows():
        free_memory = []

        for n in range(torch.cuda.device_count()):
            free_memory.append(int(torch.cuda.get_device_properties(n).total_memory / 1024 ** 2))

        return free_memory

    @staticmethod
    def get_free_gpu(memory=1000):
        skinner_map = {0: 2, 1: 0, 2: 1, 3: 3}

        free_memory = []
        if sys.platform == "posix":
            free_memory = GPU.get_free_gpus_linux()
        elif sys.platform == "win32":
            free_memory = GPU.get_free_gpus_windows()

        gpu = np.argmin(free_memory)
        if free_memory[gpu] > memory:
            if socket.gethostname() == "skinner":
                for k, v in skinner_map.items():
                    if v == gpu:
                        return k
            return gpu

        logging.error('No free GPU available.')
        exit(1)

    @classmethod
    def set(cls, gpuid, memory=1000):
        gpuid = int(gpuid)
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            logging.info("searching for free GPU")
            if gpuid == -1:
                gpuid = GPU.get_free_gpu(memory)
            os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpuid)
            if torch.cuda.device_count() == 1:  # sometimes this does not work
                torch.cuda.set_device(0)
            else:
                torch.cuda.set_device(int(gpuid))
        else:
            gpuid = os.environ['CUDA_VISIBLE_DEVICES']
            logging.info('taking GPU {} as specified in envorionment variable'.format(gpuid))
            torch.cuda.set_device(0)

        cls.device = torch.device('cuda:{}'.format(torch.cuda.current_device()))

        logging.info('Using GPU {}'.format(gpuid))
        return gpuid


class Timer:
    def __init__(self):
        self._start = time.time()

    def __str__(self):
        end = time.time()
        hours, rem = divmod(end - self._start, 3600)
        minutes, seconds = divmod(rem, 60)
        return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

    def __call__(self):
        return self.__str__()


def make_dict(args):
    try:
        args = args.__dict__
    except AttributeError:
        pass
    return args


def load_config(args):
    args = make_dict(args)

    c = args['config']

    _, _l = {}, {}
    with open(c[0], 'r') as f:
        exec(f.read(), globals(), _l)

    config = _l.get('config__', None)
    assert config, 'config__ must be set in config file'

    if len(c) > 1:
        d = getattr(config, c[1])()
    else:
        d = config()

    dict_merge(args, d)
    return args


def dict_merge(dct, merge_dct, verify=False):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :param verify: checks if no entry is added to the dictionary
    :return: None
    """
    #     dct = copy.copy(dct)
    changes_values = {}
    changes_lists = {}

    for k, _ in merge_dct.items():
        if verify:
            assert k in dct, 'key "{}" is not part of the default dict'.format(k)
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.abc.Mapping)):
            changes_lists[k] = dict_merge(dct[k], merge_dct[k], verify=verify)
        else:
            if k in dct and dct[k] != merge_dct[k]:
                changes_values[k] = merge_dct[k]

            dct[k] = merge_dct[k]

    _sorted = []
    for k, _ in dct.items():
        if k in changes_values:
            _sorted.append((k, changes_values[k]))
        elif k in changes_lists:
            _sorted.extend(changes_lists[k])

    return _sorted


def json_hash(d):
    from hashlib import sha1
    assert d is not None, "Cannot hash None!"

    def hashnumpy(a):
        if type(a) == dict:
            for k, v in a.items():
                a[k] = hashnumpy(v)

        if type(a) == list:
            for i, v in enumerate(a):
                a[i] = hashnumpy(v)

        if type(a) == np.ndarray:
            return sha1(a).hexdigest()

        if hasattr(a, '__dict__'):
            return hashnumpy(a.__dict__)

        return a

    return sha1(json.dumps(hashnumpy(d), sort_keys=True).encode()).hexdigest()


class ColorString:

    @classmethod
    def color(cls, string, color):
        return color + string + '\033[0m'

    @classmethod
    def magenta(cls, string):
        return cls.color(string, '\033[95m')

    @classmethod
    def blue(cls, string):
        return cls.color(string, '\033[94m')

    @classmethod
    def green(cls, string):
        return cls.color(string, '\033[92m')

    @classmethod
    def yellow(cls, string):
        return cls.color(string, '\033[93m')

    @classmethod
    def fail(cls, string):
        return cls.color(string, '\033[91m')

    @classmethod
    def bold(cls, string):
        return cls.color(string, '\033[1m')

    @classmethod
    def underline(cls, string):
        return cls.color(string, '\033[4m')


def read_line_seperated_file(filename) -> []:
    with open(filename) as file:
        return file.read().splitlines()


def dynamic_import(path, class_name, package='experiment'):
    module_object = import_module(path, package)
    target_class = getattr(module_object, class_name)
    return target_class


def dynamic_import_experiment(name, args):
    experiment = dynamic_import(name, 'Experiment')
    return experiment(args)
