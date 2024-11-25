import tensorflow as tf
import numpy as np

class BaseGAttN:
    def loss(logits, labels, nb_classes, class_weights):
        sample_wts = tf.reduce_sum(tf.multiply(
            tf.one_hot(labels, nb_classes), class_weights), axis=-1)
        xentropy = tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits), sample_wts)
        return tf.reduce_mean(xentropy, name='xentropy_mean')

    def training(loss, lr, l2_coef):
        # weight decay
        vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef

        # optimizer
        #opt = tf.train.AdamOptimizer(learning_rate=lr)
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        # training op
        train_op = opt.minimize(loss+lossL2)

        return train_op

    def preshape(logits, labels, nb_classes):
        new_sh_lab = [-1]
        new_sh_log = [-1, nb_classes]
        log_resh = tf.reshape(logits, new_sh_log)
        lab_resh = tf.reshape(labels, new_sh_lab)
        return log_resh, lab_resh

    def confmat(logits, labels):
        preds = tf.argmax(logits, axis=1)
        return tf.confusion_matrix(labels, preds)


    def masked_softmax_cross_entropy(logits, labels, mask,class_weights):

        """Softmax cross-entropy loss with masking."""
      

        loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
        
        class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)
        class_weights_per_sample = tf.reduce_sum(class_weights_tensor * tf.cast(labels, dtype=tf.float32), axis=-1)
        
        weighted_loss = loss * class_weights_per_sample
        
        return tf.reduce_mean(weighted_loss)

    def masked_sigmoid_cross_entropy(logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        labels = tf.cast(labels, dtype=tf.float32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labels)
        loss = tf.reduce_mean(loss, axis=1)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_accuracy(logits, labels, mask):
        """Accuracy with masking."""
        correct_prediction = tf.equal(
            tf.argmax(logits, 1), tf.argmax(labels, 1))
        
        
        accuracy_all = tf.cast(correct_prediction, tf.float32)

        return tf.reduce_mean(accuracy_all),tf.argmax(logits, 1), tf.argmax(labels, 1)

    def micro_f1(logits, labels, mask):
        """Accuracy with masking."""
        predicted = tf.round(tf.nn.sigmoid(logits))

        predicted = tf.cast(predicted, dtype=tf.int32)
        labels = tf.cast(labels, dtype=tf.int32)
        mask = tf.cast(mask, dtype=tf.int32)

        mask = tf.expand_dims(mask, -1)

        tp = tf.count_nonzero(predicted * labels * mask)
        tn = tf.count_nonzero((predicted - 1) * (labels - 1) * mask)
        fp = tf.count_nonzero(predicted * (labels - 1) * mask)
        fn = tf.count_nonzero((predicted - 1) * labels * mask)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fmeasure = (2 * precision * recall) / (precision + recall)

        fmeasure = tf.cast(fmeasure, tf.float32)

        return fmeasure
