import os
import numpy as np
import tensorflow as tf
import scipy.io as sio
from models import HeteGAT_multi
from utils import process

# Configure GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Configuration
config = {
    "dataset": "monusac",
    "featype": "hovernet",
    "train_dir": "/data2/zlb/data/MoNuSAC/split231/newsplit/train/",
    "valid_dir": "/data2/zlb/data/MoNuSAC/split231/newsplit/valid/",
    "save_path": "pre_trained/hovernet_monusac/multi_monusac",
    "best_model_path": "pre_trained/hovernet_monusac/best_model",
    "my_best_model_path": "pre_trained/hovernet_monusac/my_best_model",
    "log_file": "training_hovernet_monusac.txt",
    "batch_size": 1,
    "epochs": 100,
    "patience": 100,
    "learning_rate": 0.0001,
    "weight": [0, 1, 1, 5, 20],
    "l2_coef": 0.001,
    "hid_units": [8],
    "n_heads": [8, 1],
    "residual": False,
    "nonlinearity": tf.nn.elu,
    "nb_classes": 5,
    "ft_size": 1024,
    "dropout": 0.6
}

# Data Loading
def load_data(data_dir, name, num_classes=5):
    try:
        label_path = os.path.join(data_dir, "cell_label", f"{name}.npy")
        feature_path = os.path.join(data_dir, "cell", f"{name}.npy")
        nn_path = os.path.join(data_dir, "N*N_adj", f"{name}.mat")
        nt_path = os.path.join(data_dir, "N*T_adj", f"{name}.mat")
        nnt_path = os.path.join(data_dir, "N*N+T_adj", f"{name}.mat")

        labels = np.load(label_path)
        features = np.load(feature_path)
        NN = sio.loadmat(nn_path)["N*N_adj"]
        NT_data = sio.loadmat(nt_path)
        NNT = sio.loadmat(nnt_path)["N*N+T_adj"]

        truelabels = np.eye(num_classes)[labels].reshape(1, labels.shape[0], num_classes)
        tissue_features = NT_data["tissue_feat"][np.newaxis]
        cell_features = features[np.newaxis]
        row_networks = [NN, NT_data["N*T_adj"], NNT]
        feature_list = [features] * 3

        return row_networks, feature_list, truelabels, tissue_features, cell_features
    except Exception as e:
        raise ValueError(f"Error loading data for {name}: {e}")

# Training Loop
def train_model(config):
    print(f"Initializing model: {config['save_path']}")

    with tf.Graph().as_default():
        inputs = initialize_placeholders(config)
        logits, train_op, loss, accuracy = build_model(inputs, config)
        saver = tf.train.Saver(max_to_keep=None)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            best_val_acc = 0
            for epoch in range(config["epochs"]):
                train_loss, train_acc = run_epoch(sess, config["train_dir"], inputs, logits, train_op, loss, accuracy, is_training=True)
                val_loss, val_acc = run_epoch(sess, config["valid_dir"], inputs, logits, None, loss, accuracy, is_training=False)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    saver.save(sess, config["best_model_path"])
                print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")

# Helper Functions
def initialize_placeholders(config):
    ft_size = config["ft_size"]
    nb_classes = config["nb_classes"]
    placeholders = {
        "ftr_in": [tf.placeholder(tf.float32, shape=(config["batch_size"], None, ft_size)) for _ in range(3)],
        "bias_in": [tf.placeholder(tf.float32, shape=(config["batch_size"], None, None)) for _ in range(3)],
        "lbl_in": tf.placeholder(tf.int32, shape=(config["batch_size"], None, nb_classes)),
        "msk_in": tf.placeholder(tf.int32, shape=(config["batch_size"], None)),
        "attn_drop": tf.placeholder(tf.float32),
        "ffd_drop": tf.placeholder(tf.float32),
        "is_train": tf.placeholder(tf.bool),
        "tissuefea": tf.placeholder(tf.float32, shape=(config["batch_size"], None, ft_size)),
        "cellfea": tf.placeholder(tf.float32, shape=(config["batch_size"], None, ft_size))
    }
    return placeholders

def build_model(inputs, config):
    logits, _, _, _ = HeteGAT_multi.inference(
        inputs["tissuefea"], inputs["cellfea"], config["nb_classes"],
        inputs["is_train"], inputs["attn_drop"], inputs["ffd_drop"],
        inputs["bias_in"], config["hid_units"], config["n_heads"],
        residual=config["residual"], activation=config["nonlinearity"]
    )
    loss = HeteGAT_multi.masked_softmax_cross_entropy(logits, inputs["lbl_in"], inputs["msk_in"], config["weight"])
    accuracy = HeteGAT_multi.masked_accuracy(logits, inputs["lbl_in"], inputs["msk_in"])
    train_op = HeteGAT_multi.training(loss, config["learning_rate"], config["l2_coef"])
    return logits, train_op, loss, accuracy

def run_epoch(sess, data_dir, inputs, logits, train_op, loss, accuracy, is_training):
    losses, accuracies = [], []
    for filename in os.listdir(data_dir):
        iname = filename.split(".")[0]
        data = load_data(data_dir, iname)
        feed_dict = prepare_feed_dict(inputs, data, is_training)
        if is_training:
            _, loss_val, acc_val = sess.run([train_op, loss, accuracy], feed_dict=feed_dict)
        else:
            loss_val, acc_val = sess.run([loss, accuracy], feed_dict=feed_dict)
        losses.append(loss_val)
        accuracies.append(acc_val)
    return np.mean(losses), np.mean(accuracies)

def prepare_feed_dict(inputs, data, is_training):
    rownetworks, feature_list, labels, tissuefea, cellfea = data
    feed_dict = {inputs["tissuefea"]: tissuefea, inputs["cellfea"]: cellfea, inputs["lbl_in"]: labels, inputs["msk_in"]: np.ones((1, labels.shape[1])), inputs["is_train"]: is_training}
    for i, network in enumerate(rownetworks):
        feed_dict[inputs["ftr_in"][i]] = feature_list[i][np.newaxis]
        feed_dict[inputs["bias_in"][i]] = network[np.newaxis]
    return feed_dict

# Run Training
if __name__ == "__main__":
    train_model(config)
