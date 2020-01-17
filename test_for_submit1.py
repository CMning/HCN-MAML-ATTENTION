# import os
# import json
# import numpy as np
# import tensorflow as tf
#
#
# class Model(object):
#     def __init__(self, config, cate_list):
#         self.config = config
#
#         # Summary Writer
#         self.train_writer = tf.summary.FileWriter(config['model_dir'] + '/train')
#         self.eval_writer = tf.summary.FileWriter(config['model_dir'] + '/eval')
#
#         # Building network
#         self.init_placeholders()
#         self.build_model(cate_list)
#         self.init_optimizer()
#
#     def init_placeholders(self):
#         # [B] user id
#         self.u = tf.placeholder(tf.int32, [None, ])
#
#         # [B] item id
#         self.i = tf.placeholder(tf.int32, [None, ])
#
#         # [B] item label
#         self.y = tf.placeholder(tf.float32, [None, ])
#
#         # [B, T] user's history item id
#         self.hist_i = tf.placeholder(tf.int32, [None, None])
#
#         # [B, T] user's history item purchase time
#         self.hist_t = tf.placeholder(tf.int32, [None, None])
#
#         # [B] valid length of `hist_i`
#         self.sl = tf.placeholder(tf.int32, [None, ])
#
#         # learning rate
#         self.lr = tf.placeholder(tf.float64, [])
#
#         # whether it's training or not
#         self.is_training = tf.placeholder(tf.bool, [])
#
#     def build_model(self, cate_list):
#         item_emb_w = tf.get_variable(
#             "item_emb_w",
#             [self.config['item_count'], self.config['itemid_embedding_size']])
#
#         item_b = tf.get_variable(
#             "item_b",
#             [self.config['item_count'], ],
#             initializer=tf.constant_initializer(0.0))
#
#         cate_emb_w = tf.get_variable(
#             "cate_emb_w",
#             [self.config['cate_count'], self.config['cateid_embedding_size']])
#
#
#         cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)
#
#         #i_emb=(?,128),item_emb_w=(item_count,64),cate_emb_w=(cate_count,64)
#         i_emb = tf.concat([
#             tf.nn.embedding_lookup(item_emb_w, self.i),
#             tf.nn.embedding_lookup(cate_emb_w, tf.gather(cate_list, self.i)),
#         ], 1)
#
#         i_b = tf.gather(item_b, self.i)
#         #(?,?,128)
#         h_emb = tf.concat([
#             tf.nn.embedding_lookup(item_emb_w, self.hist_i),
#             tf.nn.embedding_lookup(cate_emb_w, tf.gather(cate_list, self.hist_i)),
#         ], 2)
#
#         if self.config['time_emb'] == True:
#             if self.config['concat_time_emb'] == True:
#                 t_emb = tf.one_hot(self.hist_t, 12, dtype=tf.float32)
#                 h_emb = tf.concat([h_emb, t_emb], -1)
#                 h_emb = tf.layers.dense(h_emb, self.config['hidden_units'])
#             else:
#                 t_emb = tf.layers.dense(tf.expand_dims(self.hist_t, -1),
#                                         self.config['hidden_units'],
#                                         activation=tf.nn.tanh)
#                 h_emb += t_emb
#
#
#         num_blocks = self.config['num_blocks']
#         num_heads = self.config['num_heads']
#         dropout_rate = self.config['dropout']
#         num_units = h_emb.get_shape().as_list()[-1]
#
#         u_emb, self.att, self.stt = attention_net(
#             h_emb,
#             self.sl,
#             i_emb,
#             num_units,
#             num_heads,
#             num_blocks,
#             dropout_rate,
#             self.is_training,
#             False)
#
#         self.logits = i_b + tf.reduce_sum(tf.multiply(u_emb, i_emb), 1)
#
#         # ============== Eval ===============
#         self.eval_logits = self.logits
#
#         # Step variable
#         TODO: 这里原来是0
#         self.global_step = tf.Variable(0, trainable=False, name='global_step')
#         self.global_epoch_step = \
#             tf.Variable(0, trainable=False, name='global_epoch_step')
#         self.global_epoch_step_op = \
#             tf.assign(self.global_epoch_step, self.global_epoch_step + 1)
#
#         # Loss
#         l2_norm = tf.add_n([
#             tf.nn.l2_loss(u_emb),
#             tf.nn.l2_loss(i_emb),
#         ])
#
#         self.loss = tf.reduce_mean(
#             tf.nn.sigmoid_cross_entropy_with_logits(
#                 logits=self.logits,
#                 labels=self.y)
#         ) + self.config['regulation_rate'] * l2_norm
#
#         self.train_summary = tf.summary.merge([
#             tf.summary.histogram('embedding/1_item_emb', item_emb_w),
#             tf.summary.histogram('embedding/2_cate_emb', cate_emb_w),
#             tf.summary.histogram('embedding/3_time_raw', self.hist_t),
#             # tf.summary.histogram('embedding/3_time_dense', t_emb),
#             tf.summary.histogram('embedding/4_final', h_emb),
#             tf.summary.histogram('attention_output', u_emb),
#             tf.summary.scalar('L2_norm Loss', l2_norm),
#             tf.summary.scalar('Training Loss', self.loss),
#         ])
#
#     def init_optimizer(self):
#         # Gradients and SGD update operation for training the model
#         trainable_params = tf.trainable_variables()
#         if self.config['optimizer'] == 'adadelta':
#             self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
#         elif self.config['optimizer'] == 'adam':
#             self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
#         elif self.config['optimizer'] == 'rmsprop':
#             self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)
#         else:
#             self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
#
#         # Compute gradients of loss w.r.t. all trainable variables
#         gradients = tf.gradients(self.loss, trainable_params)
#
#         # Clip gradients by a given maximum_gradient_norm
#         clip_gradients, _ = tf.clip_by_global_norm(
#             gradients, self.config['max_gradient_norm'])
#
#         # Update the model
#         self.train_op = self.opt.apply_gradients(
#             zip(clip_gradients, trainable_params), global_step=self.global_step)
#     TODO:add_summary=False
#     def train(self, sess, uij, l, add_summary=False):
#
#         input_feed = {
#             self.u: uij[0],
#             self.i: uij[1],
#             self.y: uij[2],
#             self.hist_i: uij[3],
#             self.hist_t: uij[4],
#             self.sl: uij[5],
#             self.lr: l,
#             self.is_training: True,
#         }
#
#         output_feed = [self.loss, self.train_op]
#
#         if add_summary:
#             output_feed.append(self.train_summary)
#
#         outputs = sess.run(output_feed, input_feed)
#
#         if add_summary:
#             self.train_writer.add_summary(
#                 outputs[2], global_step=self.global_step.eval())
#
#         return outputs[0]
#
#     def eval(self, sess, uij):
#         res1 = sess.run(self.eval_logits, feed_dict={
#             self.u: uij[0],
#             self.i: uij[1],
#             self.hist_i: uij[3],
#             self.hist_t: uij[4],
#             self.sl: uij[5],
#             self.is_training: False,
#         })
#         res2 = sess.run(self.eval_logits, feed_dict={
#             self.u: uij[0],
#             self.i: uij[2],
#             self.hist_i: uij[3],
#             self.hist_t: uij[4],
#             self.sl: uij[5],
#             self.is_training: False,
#         })
#         return np.mean(res1 - res2 > 0)
#     def eval_test(self,sess,uij):
#         res1 = sess.run(self.eval_logits, feed_dict={
#             self.u: uij[0],
#             self.i: uij[1],
#             self.hist_i: uij[3],
#             self.hist_t: uij[4],
#             self.sl: uij[5],
#             self.is_training: False,
#         })
#         res1=np.reshape(res1, (res1.size, -1))
#         pos_label = np.ones(res1.size)
#         res1 = np.insert(res1, 1, pos_label, axis=1)
#         res2 = sess.run(self.eval_logits, feed_dict={
#             self.u: uij[0],
#             self.i: uij[2],
#             self.hist_i: uij[3],
#             self.hist_t: uij[4],
#             self.sl: uij[5],
#             self.is_training: False,
#         })
#         res2 = np.reshape(res2, (res2.size, -1))
#         neg_label = np.zeros(res2.size)
#         res2 = np.insert(res2, 1, neg_label, axis=1)
#         return np.concatenate((res1,res2),axis=0)
#
#     def test(self, sess, uij):
#         res1, att_1, stt_1 = sess.run([self.eval_logits, self.att, self.stt], feed_dict={
#             self.u: uij[0],
#             self.i: uij[1],
#             self.hist_i: uij[3],
#             self.hist_t: uij[4],
#             self.sl: uij[5],
#             self.is_training: False,
#         })
#         res2, att_2, stt_2 = sess.run([self.eval_logits, self.att, self.stt], feed_dict={
#             self.u: uij[0],
#             self.i: uij[2],
#             self.hist_i: uij[3],
#             self.hist_t: uij[4],
#             self.sl: uij[5],
#             self.is_training: False,
#         })
#         return res1, res2, att_1, stt_1, att_2, stt_1
#
#     def save(self, sess):
#         checkpoint_path = os.path.join(self.config['model_dir'], 'atrank')
#         #print(self.global_step)
#         saver = tf.train.Saver()
#         save_path = saver.save(
#             sess, save_path=checkpoint_path, global_step=self.global_step.eval(session=sess))
#         json.dump(self.config,
#                   open('%s-%d.json' % (checkpoint_path, self.global_step.eval(session=sess)), 'w'),
#                   indent=2)
#         print('model saved at %s' % save_path, flush=True)
#
#     def restore(self, sess, path):
#         saver = tf.train.Saver()
#         saver.restore(sess, save_path=path)
#         print('model restored from %s' % path, flush=True)
#
#
# def attention_net(enc, sl, dec, num_units, num_heads, num_blocks, dropout_rate, is_training, reuse):
#     with tf.variable_scope("all", reuse=reuse):
#         with tf.variable_scope("user_hist_group"):
#             for i in range(num_blocks):
#                 with tf.variable_scope("num_blocks_{}".format(i)):
#                     ### Multihead Attention
#                     enc, stt_vec = multihead_attention(queries=enc,
#                                                        queries_length=sl,
#                                                        keys=enc,
#                                                        keys_length=sl,
#                                                        num_units=num_units,
#                                                        num_heads=num_heads,
#                                                        dropout_rate=dropout_rate,
#                                                        is_training=is_training,
#                                                        scope="self_attention"
#                                                        )
#
#                     ### Feed Forward
#                     enc = feedforward(enc,
#                                       num_units=[num_units // 4, num_units],
#                                       scope="feed_forward", reuse=reuse)
#
#         dec = tf.expand_dims(dec, 1)#在1的位置上增加1维
#         with tf.variable_scope("item_feature_group"):
#             for i in range(num_blocks):
#                 with tf.variable_scope("num_blocks_{}".format(i)):
#                     ## Multihead Attention ( vanilla attention)
#                     dec, att_vec = multihead_attention(queries=dec,
#                                                        queries_length=tf.ones_like(dec[:, 0, 0], dtype=tf.int32),
#                                                        keys=enc,
#                                                        keys_length=sl,
#                                                        num_units=num_units,
#                                                        num_heads=num_heads,
#                                                        dropout_rate=dropout_rate,
#                                                        is_training=is_training,
#                                                        scope="vanilla_attention")
#
#                     ## Feed Forward
#                     dec = feedforward(dec,
#                                       num_units=[num_units // 4, num_units],
#                                       scope="feed_forward", reuse=reuse)
#
#         dec = tf.reshape(dec, [-1, num_units])
#         return dec, att_vec, stt_vec

while True:
    print('孙博小肥猪')