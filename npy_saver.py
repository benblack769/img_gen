import numpy as np
import os
import collections
import tensorflow as tf


class NpySaver:
    def __init__(self,folder):
        self.folder = folder
        self.tf_obj_dict = {}

    def add(self,tf_obj,name=None):
        if name is None:
            name = tf_obj.name

        if name in self.tf_obj_dict:
            raise RuntimeError("name already in saver, no duplicates allowed")

        self.tf_obj_dict[name] = tf_obj

    def add_list(self,objs):
        for obj in name_obj_pairs:
            self.add(obj)

    def load_all(self,sess):
        for name,obj in self.tf_obj_dict.items():
            path = os.path.join(self.folder,name)
            value = np.load(path)
            assign = tf.assign(obj,value)
            sess.run(assign)

    def save_all(self,sess):
        for name,obj in self.tf_obj_dict.items():
            path = os.path.join(self.folder,name)
            value = sess.run(obj)
            np.save(path,value)
