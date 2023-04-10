import json
import os
import shutil
import stat
from json import JSONDecodeError

import git
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import Config
from .general_utils import get_random_time_stamp
from .logger import logging_info


class ResultRecorder:
    def __init__(self, path_record, initial_record=None, use_git=True):
        """
        Initialize the result recorder. The results will be saved in a temporary file defined by path_record.temp.
        To end recording and transfer the temporary files, self.end_recording() must be called.
        :param path_record: the saving path of the recorded results.
        :type path_record: str
        :param initial_record: a record to be initialize with, usually the config in practice
        :type initial_record: dict
        """
        self.__ending = False
        self.__record = dict()

        self.__path_temp_record = (
            "%s.result.temp" % path_record
            if not path_record.endswith(".result")
            else path_record + ".temp"
        )
        self.__path_record = (
            "%s.result" % path_record
            if not path_record.endswith(".result")
            else path_record
        )

        if os.path.exists(self.__path_temp_record):
            shutil.move(
                self.__path_temp_record,
                self.__path_temp_record + ".%s" % get_random_time_stamp(),
            )
        if os.path.exists(self.__path_record):
            shutil.move(
                self.__path_record, self.__path_record + ".%s" % get_random_time_stamp()
            )

        if initial_record is not None:
            self.update(initial_record)

        if use_git:
            repo = git.Repo(path=os.getcwd())
            assert not repo.is_dirty()
            self.__setitem__("git_commit", repo.head.object.hexsha)

    def write_record(self, line):
        """
        Add a line to the recorded result file.
        :param line: the content to be write
        :type line: str
        """
        with open(self.__path_temp_record, "a", encoding="utf-8") as fin:
            fin.write(line + "\n")

    def __getitem__(self, key):
        """
        Return the item based on the key.
        :param key:
        :type key:
        :return: results[key]
        """
        return self.__record[key]

    def __setitem__(self, key, value):
        """
        Set result[key] = value
        """
        assert not self.__ending
        assert key not in self.__record.keys()
        self.__record[key] = value
        self.write_record(json.dumps({key: value}))

    def update(self, new_record):
        """
        Update the results from new_record.
        :param new_record: the new results dict
        :type new_record: dict
        """
        for k in new_record.keys():
            self.__setitem__(k, new_record[k])

    def add_with_logging(self, key, value, msg=None):
        """
        Add an item to results and also print with logging. The format of logging can be defined.
        :param key: the key
        :type key: str
        :param value: the value to be added to the results
        :param msg: the message to the logger, format can be added. e.g. msg="Training set %s=%.4lf."
        :type msg: str
        :return:
        :rtype:
        """
        self.__setitem__(key, value)
        if msg is None:
            logging_info("%s: %s" % (key, str(value)))
        else:
            logging_info(msg % value)

    def end_recording(self):
        """
        End the recording. This function will remove the .temp suffix of the recording file and add an END signal.
        :return:
        :rtype:
        """
        self.__ending = True
        self.write_record("\n$END$\n")
        shutil.move(self.__path_temp_record, self.__path_record)
        os.chmod(self.__path_record, stat.S_IREAD)

    def dump(self, path_dump):
        """
        Dump the result record in the path_dump.
        :param path_dump: the path to dump the result record
        :type path_dump: str
        """
        assert self.__ending
        path_dump = (
            "%s.result" % path_dump if not path_dump.endswith(".result") else path_dump
        )
        assert not os.path.exists(path_dump)
        shutil.copy(self.__path_record, path_dump)

    def to_dict(self):
        """
        Return the results as a dict.
        :return: the results
        :rtype: dict
        """
        return self.__record

    def show(self):
        """
        To show the reuslts in logger.
        """
        logging_info(
            "\n%s"
            % json.dumps(
                self.__record, sort_keys=True, indent=4, separators=(",", ": ")
            )
        )


def load_result(path_record, return_type="dict"):
    """
    Load the result based on path_record.
    :param path_record: the path of the record
    :type path_record: str
    :param return_type: "dict" or "dataframe"
    :return: the result and whether the result record is ended
    :rtype: dict, bool
    """
    ret = dict()
    with open(path_record, "r", encoding="utf-8") as fin:
        ret["path"] = path_record
        for line in fin.readlines():
            if line.strip() == "$END$":
                return ret, True
            if len(line.strip().split()) == 0:
                continue
            ret.update(json.loads(line))
    if return_type == "dataframe":
        ret = pd.DataFrame(pd.Series(ret)).transpose()
    return ret, False


def collect_results(
    dir_results,
    collect_condition_func=None,
    pickled_filename=".pickled_results.jbl",
    load_temp=False,
    force_reload=False,
):
    """
    Collect all the ended results in dir_results.
    :param dir_results: the directory of the reuslts to be collected
    :type dir_results: str
    :param collect_condition_func: function to judge whether collect or not
    :param pickled_filename: filename of the pickled file
    :param load_temp: whether to include temp result
    :param force_reload: whether to reload all result
    :return: all ended result records
    :rtype: pd.DataFrame
    """
    assert os.path.exists(dir_results)
    path_pickled_results = os.path.join(dir_results, pickled_filename)
    if not force_reload and os.path.exists(path_pickled_results):
        data = joblib.load(path_pickled_results)
        already_collect_list = data["path"].values
    else:
        data = pd.DataFrame()
        already_collect_list = []

    updated = False
    to_be_read = []
    for path, dir_list, file_list in os.walk(dir_results):
        for file_name in file_list:
            file_path = os.path.join(path, file_name)
            if not os.path.isdir(file_path) and (
                file_path.endswith(".result")
                or (load_temp and file_path.endswith(".result.temp"))
            ):
                if file_path not in already_collect_list:
                    if collect_condition_func is None or collect_condition_func(
                        file_path
                    ):
                        to_be_read.append(file_path)
    print("Got %d to be read." % len(to_be_read))
    new_data = list()
    for file_path in tqdm(to_be_read):
        try:
            result, ended = load_result(file_path)
            if ended or load_temp:
                new_data.append(pd.DataFrame(pd.Series(result)).transpose())
            updated = True
        except JSONDecodeError:
            print("Collection Failed at %s" % file_path)
    print("Got %d new." % len(new_data))

    if updated:
        new_data = pd.concat(new_data, axis=0)
        data = pd.concat([data, new_data], axis=0)
        joblib.dump(data, path_pickled_results)
    return data.copy()


def collect_dead_results(dir_results):
    """
    Collect all un-ended results.
    :param dir_results: the directory of the reuslts to be collected
    :type dir_results: str
    :return: all un-ended result records.
    :rtype: pd.DataFrame
    """
    assert os.path.exists(dir_results)
    data = list()
    for path, dir_list, file_list in os.walk(dir_results):
        for file_name in file_list:
            path_file = os.path.join(path, file_name)
            if not os.path.isdir(path_file) and path_file.endswith(".result.temp"):
                result, ended = load_result(os.path.join(path_file))
                if not ended:
                    data.append(pd.DataFrame(pd.Series(result)).transpose())
    data = pd.concat(data, axis=0)
    return data


def get_max_epoch(data):
    max_epoch = max(
        [
            int(c.split("-")[0].split("_")[1])
            for c in data.columns
            if c.startswith("epoch_")
        ]
    )
    return max_epoch


def get_recorded_metrics(data):
    return set([c.split("-")[1] for c in data.columns if c.startswith("epoch_0-")])


def get_trajectory(data, metric, filters=None):
    data_filtered = data[filters] if filters is not None else data
    assert len(data_filtered) == 1, "%d Files Located" % len(data_filtered)
    max_epoch = get_max_epoch(data)

    x, y = [], []
    for epoch in range(max_epoch + 1):
        if "epoch_%d-%s" % (epoch, metric) in data_filtered.columns:
            v = data_filtered["epoch_%d-%s" % (epoch, metric)].values[0]
            if (type(v) in [str]) or (not np.isnan(v) and not np.isinf(v)):
                x.append(epoch)
                y.append(v)
            else:
                break
        elif epoch == 0:
            continue
        else:
            break

    return x, y


def fill_config_na(data, config_path, prefix="", suffix="", exclude_key=None):
    config = Config(default_config_file=config_path)
    for k in config.keys():
        if k not in exclude_key:
            data[prefix + k + suffix] = data[prefix + k + suffix].fillna(config[k])
    return data


def get_columns_group_by(data, config_path, exclude_key=("exp_name", "random_seed")):
    ret = []
    config = Config(default_config_file=config_path)
    for k in config.keys():
        if (
            k not in exclude_key
            and len(set([(tuple(i) if type(i) == list else i) for i in data[k].values]))
            != 1
        ):
            ret.append(k)
    return ret


def remove_duplicate(data, keys=("phase", "exp_name")):
    data = data.drop_duplicates(subset=keys, keep="last")
    return data


def merge_phase(
    data, data_to_merge, merge_on_keys=("exp_name",), suffixes=("", "_eval")
):
    return data.merge(data_to_merge, how="inner", on=merge_on_keys, suffixes=suffixes)
