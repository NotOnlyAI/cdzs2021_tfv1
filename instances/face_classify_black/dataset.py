import random
from imutils import paths
import cv2
import numpy as np
import tensorflow as tf


def generator_for_dataset(is_training):
    print("is training : loading training dataset")
    if is_training:
        train_img_dirs = [
            "E:/datasets/face_classify_black/black",
            "E:/datasets/face_classify_black/noblack",
        ]
    else:
        train_img_dirs = ["D:/YAOCHAO/datasets/facemask/cdzs/20200316/mask",
                          "D:/YAOCHAO/datasets/facemask/cdzs/20200316/nomask",
                          ]

    img_paths1 = []
    for i in range(len(train_img_dirs)):
        img_paths1 += [el for el in paths.list_images(train_img_dirs[i])]

    img_paths = img_paths1
    print(len(img_paths))
    random.shuffle(img_paths)
    while True:
        random.shuffle(img_paths)
        for image_path in img_paths:
            image = cv2.imread(image_path)
            image = cv2.resize(image, (112, 112))
            image = np.array(image, np.float32)
            image = image / 255.0
            label = image_path.split("/")[-1].split("\\")[0]
            print(label)
            if label == "mask":
                id = 1
            else:
                id = 0
            yield image, id


def load_dataset(is_training=True):
    if is_training:
        dataset = tf.data.Dataset.from_generator(generator_for_dataset, (tf.float32, tf.int32),
                                                 args=[True])
    else:
        dataset = tf.data.Dataset.from_generator(generator_for_dataset, (tf.float32, tf.int32),
                                                 args=[False])
    return dataset


def check_dataset():
    dataset = load_dataset()
    batch_size = 8
    dataset = dataset.batch(batch_size, drop_remainder=True)
    iterator = dataset.make_initializable_iterator()
    image, id = iterator.get_next()

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(10):
            image0, id0 = sess.run([image, id])
            for j in range(batch_size):
                show_image = image0[j]
                show_image = np.array(show_image * 255, np.uint8)
                cls = id0[j]
                if cls == 0:
                    show_path = "show/no_mask_%s_%s.jpg" % (i, j)
                else:
                    show_path = "show/mask_%s_%s.jpg" % (i, j)
                cv2.imwrite(show_path, show_image)


if __name__ == '__main__':
    check_dataset()
