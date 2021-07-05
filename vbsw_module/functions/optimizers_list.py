import tensorflow as tf


def opt_list(name, lr = None):
    if name == "adamlr":
        return tf.keras.optimizers.Adam(lr)
    elif name == "adam":
        return tf.keras.optimizers.Adam()
    elif name == "sgd":
        return tf.keras.optimizers.SGD()
    elif name == "sgdlr":
        return tf.keras.optimizers.SGD(lr)
    elif name == "rmsprop":
        return tf.keras.optimizers.RMSprop()
    elif name == "rmsproplr":
        return tf.keras.optimizers.RMSprop(lr)
    elif name == "adagrad":
        return tf.keras.optimizers.Adagrad()
    elif name == "adagradlr":
        return tf.keras.optimizers.Adagrad(lr)
    else:
        pass
