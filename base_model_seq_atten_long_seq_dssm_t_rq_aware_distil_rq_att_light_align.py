# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from model_ops import utils
from model_ops import ops as base_ops, metrics
from model_ops.tflog import tflogger as logging
from model_ops.attention import multihead_attention, feedforward, multihead_target_attention
from model_zoo.model import Model
from model_zoo.base_model_seq_atten_long_seq_dssm_t_rq_aware_distil import BaseModelLongSeqDssmTRqAwareDistil
from model_zoo.base_model_seq_atten_long_seq_dssm_t_rq_aware_distil_rq_att_light import BaseModelLongSeqDssmTRqAwareDistilRqAttLight


class BaseModelLongSeqDssmTRqAwareDistilRqAttLightAlign(BaseModelLongSeqDssmTRqAwareDistilRqAttLight):
    def __init__(self,
                 model_config,
                 training_config,
                 mc,
                 fg,
                 context,
                 name="CTR"):
        super(BaseModelLongSeqDssmTRqAwareDistilRqAttLightAlign, self).__init__(model_config,
                                                    training_config,
                                                    mc,
                                                    fg,
                                                    context,
                                                    name)

        self.distil_kl_loss_weight=1.0
        if hasattr(self.training_config, 'distil_kl_loss_weight'):
            self.distil_kl_loss_weight = self.training_config.distil_kl_loss_weight

   

    def ta_loss(self):
        with tf.name_scope("{}_Ta_Loss_Op".format(self.name)):
            
            if self.config.loss_type == 'sample_correct':
                logging.info("Loss Sample Correct.")
                self.ta_loss = tf.reduce_mean(
                    sigmoid_cross_entropy_with_logits_sample_correct(
                        logits=self.ta_logits,
                        labels=self.label))
            elif self.config.loss_type == 'pu_loss':
                logging.info("Loss PU Loss.")
                self.ta_loss = tf.reduce_mean(
                    sigmoid_cross_entropy_with_logits_pu_loss(
                        logits=self.ta_logits,
                        labels=self.label))
            else:
                self.ta_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.ta_logits,
                        labels=self.label))
            # 新增KL散度蒸馏损失
            batch_size = tf.shape(self.ta_main_net)[0]
            total_kl_loss = 0.0
            
            # 获取原始target embedding (v) 和注意力后的embedding (v_bar)
            #v = self.block_layer_dict[self.target_atten_query]  # [N, D]
            v= self.cross_item_net   #[N, K_dim1]
            v_bar = tf.stop_gradient(self.ta_main_net)  # [N, D]
            
            
            for l in range(self.kernel_num):  
                # 原始码本 (用于计算原始分布 P_v^l)
                x_l = self.kernel_list[l]  # [K_dim0, K_dim1] (原始码本向量)
                
                # 注意力后的码本 (来自user_kernel_cross_net的第l个kernel结果)
                x_l_bar = self.user_kernel_cross_net[:, l, :, :]  # [N, K_dim0, D_out]
                
                # 计算原始分布 P_v^l: 基于原始码本
                # v: [N, D], x_l: [K_dim0, D] => 扩展为[N, K_dim0, D]后计算距离
                distance=tf.reduce_sum(
                    tf.square(tf.expand_dims(v, 1) - x_l),  # [N, K_dim0, D]
                    axis=2)
                #z_idxs = tf.argmin(distance, axis=1)# [N]

                z_idxs = tf.cast(tf.argmin(distance, axis=1), tf.int32)# [N]
                select_x = tf.gather(x_l, z_idxs)  # [N, K_dim1]
                batch_indices = tf.stack([
                    tf.range(tf.shape(z_idxs)[0], dtype=tf.int32),
                    z_idxs
                ], axis=1)#[N,2]
                select_x = tf.gather(x_l, z_idxs)  # [N, K_dim1]

                select_x_bar = tf.gather_nd(x_l_bar, batch_indices)  # [N, D_out]

                #select_x_bar = tf.batch_gather(x_l_bar, tf.expand_dims(z_idxs, axis=1))  # [N, 1, D_out]
                #select_x_bar = tf.squeeze(select_x_bar, axis=1)  # [N, D_out]


                
                logits_p = -distance
                p = tf.nn.softmax(logits_p, axis=1)  # [N, K_dim0]
                
                # 计算注意力后的分布 P_v_bar^l: 基于注意力后的码本
                # v_bar: [N, D], x_l_bar: [N, K_dim0, D]
                logits_p_bar = -tf.reduce_sum(
                    tf.square(tf.expand_dims(v_bar, 1) - x_l_bar),  # [N, K_dim0, D]
                    axis=2)  # [N, K_dim0]
                p_bar = tf.nn.softmax(logits_p_bar, axis=1)  # [N, K_dim0]
                
                
                kl_forward = tf.reduce_sum(
                    p * tf.math.log(p / (p_bar + 1e-8) + 1e-8), 
                    axis=1)  # [N]
                kl_backward = tf.reduce_sum(
                    p_bar * tf.math.log(p_bar / (p + 1e-8) + 1e-8), 
                    axis=1)  # [N]
                total_kl_loss += tf.reduce_mean(kl_forward + kl_backward)

                #v更新
                v-=select_x
                v_bar-=select_x_bar
            
            reconstruction_loss=tf.nn.l2_loss(self.final_user_kernel_cross_net - tf.stop_gradient(self.ta_main_net)) / tf.cast(batch_size, tf.float32)
            
            self.ta_distil_loss = self.distil_kl_loss_weight *total_kl_loss + reconstruction_loss

    


    def metrics_op(self):
        super(BaseModelLongSeqDssmTRqAwareDistilRqAttLightAlign, self).metrics_op()
        with tf.name_scope("{}_Metrics".format(self.name)):
            
            for i in range(self.kernel_num):
                layer_i_selected_ids = tf.squeeze(self.z_idx_list[i])
                unique_ids, _, counts = tf.unique_with_counts(layer_i_selected_ids)
                most_frequent_index = tf.argmax(counts)
                most_frequent_id_for_layer_i = unique_ids[most_frequent_index]
                self.metrics['scalar/rq_{}_most_frequent_id'.format(i)] = tf.cast(most_frequent_id_for_layer_i,tf.float32)
                    
            all_codebook_utilizations = []
            all_token_entropies = []

            for i in range(self.kernel_num):
                    # self.z_idx_list[i] 的形状是 [batch_size, 1]，需要降维
                    # selected_indices 的形状是 [batch_size]
                selected_indices = tf.squeeze(self.z_idx_list[i])

                    # --- 指标1: 码本利用率 (Codebook Utilization) ---
                    # 计算在一个批次中，当前码本被使用了的向量（token）的比例
                    
                    # 1. 获取批次内使用的唯一索引
                unique_used_indices, _ = tf.unique(selected_indices)
                    
                    # 2. 计算唯一索引的数量
                num_unique_used = tf.cast(tf.size(unique_used_indices), dtype=tf.float32)
                    
                    # 3. 获取码本的总大小
                codebook_size = tf.cast(self.kernel_shape_0, dtype=tf.float32)
                    
                    # 4. 计算利用率
                utilization = num_unique_used / codebook_size
                    
                    # 将单个码本的利用率添加到 metrics 字典中
                self.metrics['scalar/rq_{}_codebook_utilization'.format(i)] = utilization
                all_codebook_utilizations.append(utilization)

                    # --- 指标2: 令牌分布熵 (Token Distribution Entropy) ---
                    # 使用香农熵衡量码本中令牌选择的均匀性。熵越高，表示选择越均匀、多样化。
                    
                    # 1. 获取每个索引在批次中出现的次数
                _, _, counts = tf.unique_with_counts(selected_indices)
                    
                    # 2. 将次数转换为概率分布
                batch_size_float = tf.cast(tf.shape(selected_indices)[0], dtype=tf.float32)
                probabilities = tf.cast(counts, dtype=tf.float32) / batch_size_float
                    
                    # 3. 计算香农熵: H(X) = -sum(p(x) * log(p(x)))
                    #    为防止 log(0) 导致 NaN，给概率加上一个极小值 epsilon
                epsilon = 1e-9
                token_entropy = -tf.reduce_sum(probabilities * tf.math.log(probabilities + epsilon))
                    
                    # 将单个码本的令牌熵添加到 metrics 字典中
                self.metrics['scalar/rq_{}_token_entropy'.format(i)] = token_entropy
                all_token_entropies.append(token_entropy)
                
                # 计算所有码本的平均利用率和平均熵，作为模型的总体指标
                
            self.metrics['scalar/rq_avg_codebook_utilization'] = tf.reduce_mean(tf.stack(all_codebook_utilizations))
                
            self.metrics['scalar/rq_avg_token_entropy'] = tf.reduce_mean(tf.stack(all_token_entropies))
                

