from config import Config

session_config = tf.ConfigProto()
session_config.gpu_options.visible_device_list = "0"
session_config.gpu_options.allow_growth = True
set_session(tf.Session(config=session_config))

config = Config()
model.train_predict(config)
