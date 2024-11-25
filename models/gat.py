import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN


class GAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        
        attns = []
        for _ in range(n_heads[0]):
            attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                                          out_sz=hid_units[0], activation=activation,
                                          in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
                                              out_sz=hid_units[i], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-1)
        out = []
        for i in range(n_heads[-1]):
            out.append(layers.attn_head(h_1, bias_mat=bias_mat,
                                        out_sz=nb_classes, activation=lambda x: x,
                                        in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]

        return logits

class HeteGAT_multi(BaseGAttN):
    def inference(tissuefea, cellfea, nb_classes,  training, attn_drop, ffd_drop,
                  bias_mat_list, hid_units, n_heads, activation=tf.nn.elu, residual=False,   #hid_units = [8]
                  mp_att_size=128):


        embed_list = []

        attns = []
        coefs=[]  
        for _ in range(n_heads[0]):
            
            attn,coef=layers.attn_head(cellfea,cellfea, bias_mat=bias_mat_list[0],
                                            out_sz=hid_units[0], activation=activation,
                                            in_drop=ffd_drop, coef_drop=attn_drop, residual=False,return_coef=True)
            attns.append(attn)
            coefs.append(coef)
            
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):

            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.attn_head(h_1, bias_mat=bias_mat_list[0],
                                                out_sz=hid_units[i],
                                                activation=activation,
                                                in_drop=ffd_drop,
                                                coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-1)
        
        embed_list.append(tf.expand_dims(tf.squeeze(h_1,[0]), axis=1))
        
        attns = []
        for _ in range(n_heads[0]):

            attns.append(layers.attn_head(cellfea,tissuefea, bias_mat=bias_mat_list[1],
                                            out_sz=hid_units[0], activation=activation,
                                            in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-1)

        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
                                                out_sz=hid_units[i],
                                                activation=activation,
                                                in_drop=ffd_drop,
                                                coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-1)
        
        embed_list.append(tf.expand_dims(tf.squeeze(h_1,[0]), axis=1))

        multi_embed = tf.concat(embed_list, axis=1)

        final_embed, att_val = layers.SimpleAttLayer(multi_embed, mp_att_size,
                                                     time_major=False,
                                                     return_alphas=True)
        out = []
        for i in range(n_heads[-1]):
      
            out.append(tf.layers.dense(final_embed, nb_classes, activation=None))

        logits = tf.add_n(out) / n_heads[-1]


        logits = tf.expand_dims(logits, axis=0)
        return logits, final_embed, att_val,coefs 
    
class HeteGAT_multi_attn(BaseGAttN):
    def inference(tissuefea, cellfea, nb_classes,  training, attn_drop, ffd_drop,
                  bias_mat_list, hid_units, n_heads, activation=tf.nn.elu, residual=False,  
                  mp_att_size=128):

        embed_list = []

        attns = []
        coefs=[]  
        for _ in range(n_heads[0]):
            
            attn,coef=layers.attn_head(cellfea,cellfea, bias_mat=bias_mat_list[0],
                                            out_sz=hid_units[0], activation=activation,
                                            in_drop=ffd_drop, coef_drop=attn_drop, residual=False,return_coef=True)
            attns.append(attn)
        coefs.append(coef)
        
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attn,coef=layers.attn_head(h_1, bias_mat=bias_mat_list[0],
                                                out_sz=hid_units[i],
                                                activation=activation,
                                                in_drop=ffd_drop,
                                                coef_drop=attn_drop, residual=residual,return_coef=True)
                attns.append(attn)
        
            h_1 = tf.concat(attns, axis=-1)
        
        embed_list.append(tf.expand_dims(tf.squeeze(h_1,[0]), axis=1))

        
        attns = []
        for _ in range(n_heads[0]):

            attn,coef=layers.attn_head(cellfea,tissuefea, bias_mat=bias_mat_list[1],
                                            out_sz=hid_units[0], activation=activation,
                                            in_drop=ffd_drop, coef_drop=attn_drop, residual=False,return_coef=True)
            attns.append(attn)
        coefs.append(coef)
        
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
                                                out_sz=hid_units[i],
                                                activation=activation,
                                                in_drop=ffd_drop,
                                                coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-1)
        
        embed_list.append(tf.expand_dims(tf.squeeze(h_1,[0]), axis=1))

        multi_embed = tf.concat(embed_list, axis=1)
        final_embed, att_val = layers.SimpleAttLayer(multi_embed, mp_att_size,
                                                     time_major=False,
                                                     return_alphas=True)
        out = []
        for i in range(n_heads[-1]):
      
            out.append(tf.layers.dense(final_embed, nb_classes, activation=None))
      
        logits = tf.add_n(out) / n_heads[-1]


        logits = tf.expand_dims(logits, axis=0)
        return logits, multi_embed,final_embed, att_val,coefs


