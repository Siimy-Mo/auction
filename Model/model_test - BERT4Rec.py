import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
import numpy as np
from tensorflow.contrib import rnn
import math
import six
seed = 42
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

def gradients(opt, loss, vars, step, max_gradient_norm=None, dont_clip=[]):
    # print(loss, loss.shape)
    # print(vars)
    # for i in vars:
    #     print(i.name,'with shape:',i.shape)
    gradients = opt.compute_gradients(loss, vars)
    if max_gradient_norm is not None:
        to_clip = [(g, v) for g, v in gradients if v.name not in dont_clip]
        not_clipped = [(g, v) for g, v in gradients if v.name in dont_clip]
        gradients, variables = zip(*to_clip)
        clipped_gradients, _ = clip_ops.clip_by_global_norm(
            gradients,
            max_gradient_norm
        )
        gradients = list(zip(clipped_gradients, variables)) + not_clipped

    # Add histograms for variables, gradients and gradient norms
    for gradient, variable in gradients:
        if isinstance(gradient, ops.IndexedSlices):
            grad_values = gradient.values
        else:
            grad_values = gradient
        if grad_values is None:
            print('warning: missing gradient: {}'.format(variable.name))
        if grad_values is not None:
            tf.compat.v1.summary.histogram(variable.name, variable)
            tf.compat.v1.summary.histogram(variable.name + '/gradients', grad_values)
            tf.compat.v1.summary.histogram(
                variable.name + '/gradient_norm',
                clip_ops.global_norm([grad_values])
            )

    return opt.apply_gradients(gradients, global_step=step) #更新variable

def Loss_F(useremb, queryemb, product_pos, product_neg, k):
    # useremb -> [batch_size, emb_size]
    # product_neg -> [batch_size, k, emb_size] 
    u_plus_q = useremb + queryemb
    dis_pos = tf.multiply(tf.norm(u_plus_q - product_pos, ord=2, axis = 1), -1.0)
    log_u_plus_q_minus_i_pos = tf.math.log(tf.sigmoid(dis_pos))
    
    expand_u_plus_q = tf.tile(tf.expand_dims(u_plus_q, axis=1),[1,k,1])
    dis_neg = tf.norm(expand_u_plus_q - product_neg, ord=2, axis = 2)
    log_u_plus_q_minus_i_neg = tf.reduce_sum(tf.math.log(tf.sigmoid(dis_neg)),axis=1)
    batch_loss = -1 * (log_u_plus_q_minus_i_neg + log_u_plus_q_minus_i_pos)
    batch_loss = tf.reduce_mean(batch_loss)
    return batch_loss, tf.reduce_mean(-dis_pos), tf.reduce_mean(dis_neg)

def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=128,
                      num_hidden_layers=12,
                      num_attention_heads=16,
                      intermediate_size=3072,
                    #   intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads))
    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    if input_width != hidden_size:
        raise ValueError(
            "The width of the input tensor (%d) != hidden size (%d)" %
            (input_width, hidden_size))
    prev_output = reshape_to_matrix(input_tensor)

    all_layer_outputs = []
    for layer_idx in range(num_hidden_layers):
        with tf.variable_scope("layer_%d" % layer_idx):
            layer_input = prev_output

            with tf.variable_scope("attention"):
                attention_heads = []
                with tf.variable_scope("self"):
                    attention_head = attention_layer(
                        from_tensor=input_tensor,
                        to_tensor=input_tensor,
                        attention_mask=attention_mask,
                        num_attention_heads=num_attention_heads,
                        size_per_head=attention_head_size,
                        attention_probs_dropout_prob=
                        attention_probs_dropout_prob,
                        initializer_range=initializer_range,
                        do_return_2d_tensor=True,
                        batch_size=batch_size,
                        from_seq_length=seq_length,
                        to_seq_length=seq_length)
                    attention_heads.append(attention_head)

                attention_output = None
                if len(attention_heads) == 1:
                    attention_output = attention_heads[0]
                else:
                    # In the case where we have other sequences, we just concatenate
                    # them to the self-attention head before the projection.
                    attention_output = tf.concat(attention_heads, axis=-1)

                with tf.variable_scope("output"):
                    attention_output = tf.layers.dense(
                        attention_output,
                        hidden_size,
                        kernel_initializer=create_initializer(
                            initializer_range))
                    attention_output = dropout(attention_output,
                                                hidden_dropout_prob)
                    attention_output = layer_norm(attention_output +
                                                    layer_input)
            # The activation is only applied to the "intermediate" hidden layer.
            with tf.variable_scope("intermediate"):
                intermediate_output = tf.layers.dense(
                    attention_output,
                    intermediate_size,
                    activation=gelu,
                    kernel_initializer=create_initializer(initializer_range))
            
            # Down-project back to `hidden_size` then add the residual.
            with tf.variable_scope("output"):
                layer_output = tf.layers.dense(
                    intermediate_output,
                    hidden_size,
                    kernel_initializer=create_initializer(initializer_range))
                layer_output = dropout(layer_output, hidden_dropout_prob)
                layer_output = layer_norm(layer_output + attention_output)
                prev_output = layer_output
                all_layer_outputs.append(layer_output)
    # 转换一下尺寸
    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
        return final_outputs
    else:
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output

def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
    """Performs multi-headed attention from `from_tensor` to `to_tensor`.

    This is an implementation of multi-headed attention based on "Attention
    is all you Need". If `from_tensor` and `to_tensor` are the same, then
    this is self-attention. Each timestep in `from_tensor` attends to the
    corresponding sequence in `to_tensor`, and returns a fixed-with vector.

    This function first projects `from_tensor` into a "query" tensor and
    `to_tensor` into "key" and "value" tensors. These are (effectively) a list
    of tensors of length `num_attention_heads`, where each tensor is of shape
    [batch_size, seq_length, size_per_head].

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor and returned.

    In practice, the multi-headed attention are done with transposes and
    reshapes rather than actual separate tensors.

    Args:
        from_tensor: float Tensor of shape [batch_size, from_seq_length,
        from_width].
        to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
        attention_mask: (optional) int32 Tensor of shape [batch_size,
        from_seq_length, to_seq_length]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions in
        the mask that are 0, and will be unchanged for positions that are 1.
        num_attention_heads: int. Number of attention heads.
        size_per_head: int. Size of each attention head.
        query_act: (optional) Activation function for the query transform.
        key_act: (optional) Activation function for the key transform.
        value_act: (optional) Activation function for the value transform.
        attention_probs_dropout_prob: (optional) float. Dropout probability of the
        attention probabilities.
        initializer_range: float. Range of the weight initializer.
        do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
        * from_seq_length, num_attention_heads * size_per_head]. If False, the
        output will be of shape [batch_size, from_seq_length, num_attention_heads
        * size_per_head].
        batch_size: (Optional) int. If the input is 2D, this might be the batch size
        of the 3D version of the `from_tensor` and `to_tensor`.
        from_seq_length: (Optional) If the input is 2D, this might be the seq length
        of the 3D version of the `from_tensor`.
        to_seq_length: (Optional) If the input is 2D, this might be the seq length
        of the 3D version of the `to_tensor`.

    Returns:
        float Tensor of shape [batch_size, from_seq_length,
        num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
        true, this will be of shape [batch_size * from_seq_length,
        num_attention_heads * size_per_head]).

    Raises:
        ValueError: Any of the arguments or tensor shapes are invalid.
    """

    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                             seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None
                or to_seq_length is None):
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # `query_layer` = [B*F, N*H]
    query_layer = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        kernel_initializer=create_initializer(initializer_range))

    # `key_layer` = [B*T, N*H]
    key_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=key_act,
        name="key",
        kernel_initializer=create_initializer(initializer_range))

    # `value_layer` = [B*T, N*H]
    value_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="value",
        kernel_initializer=create_initializer(initializer_range))

    # `query_layer` = [B, N, F, H]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                       num_attention_heads, from_seq_length,
                                       size_per_head)

    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size,
                                     num_attention_heads, to_seq_length,
                                     size_per_head)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape(
        value_layer,
        [batch_size, to_seq_length, num_attention_heads, size_per_head])

    # `value_layer` = [B, N, T, H]
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    if do_return_2d_tensor:
        # `context_layer` = [B*F, N*V]
        context_layer = tf.reshape(context_layer, [
            batch_size * from_seq_length, num_attention_heads * size_per_head
        ])
    else:
        # `context_layer` = [B, F, N*V]
        context_layer = tf.reshape(
            context_layer,
            [batch_size, from_seq_length, num_attention_heads * size_per_head])

    return context_layer

def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
        input_tensor: float Tensor.
        dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
        A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output

def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor,
        begin_norm_axis=-1,
        begin_params_axis=-1,
        scope=name)

def gelu(input_tensor):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
        input_tensor: float Tensor to perform activation.

    Returns:
        `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf

def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)

def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
        tensor: A tf.Tensor object to find the shape of.
        expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
        name: Optional name of the tensor for the error message.

    Returns:
        A list of dimensions of the shape of tensor. All static dimensions will
        be returned as python integers, and dynamic dimensions will be returned
        as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape

def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor

def reshape_from_matrix(output_tensor, orig_shape_list):
    """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]
    print(output_tensor.shape,orig_dims,width)
    return tf.reshape(output_tensor, orig_dims + [width])

def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
        tensor: A tf.Tensor to check the rank of.
        expected_rank: Python integer or list of integers, expected rank.
        name: Optional name of the tensor for the error message.

    Raises:
        ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape),
             str(expected_rank)))

class Seq(object):
    def __init__(self, Embed, params):
        self.UserID = tf.compat.v1.placeholder(tf.int32, shape=(None), name = 'uid')
        self.current_product_id = tf.compat.v1.placeholder(tf.int32, shape=(None), name='current_product_id')
        self.current_product_neg_id = tf.compat.v1.placeholder(tf.int32, shape=[None,None], name='current_product_neg_id')
        self.short_term_query_id = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name='short_term_query_id')
        self.short_term_query_len = tf.compat.v1.placeholder(tf.int32, shape=(None), name='short_term_query_len')
        self.short_term_query_len_mask = tf.compat.v1.placeholder(tf.float32, shape=[None,None], name='short_term_query_len_mask')
       
        self.short_term_before_product_id = tf.compat.v1.placeholder(tf.int32, shape=[None,None], name='short_term_before_product_id')
        self.short_term_before_product_len = tf.compat.v1.placeholder(tf.int32, shape=(None), name='short_term_before_product_len')
        self.short_term_before_product_flag = tf.compat.v1.placeholder(tf.int32, shape=(None), name='short_term_before_product_flag')

        self.current_duration = tf.compat.v1.placeholder(tf.int32, shape=(None), name='current_auction_duration')
        self.current_openbid = tf.compat.v1.placeholder(tf.int32, shape=(None), name='current_auction_openbid')
        self.current_type = tf.compat.v1.placeholder(tf.int32, shape=(None), name='current_auction_type')
        self.before_durations = tf.compat.v1.placeholder(tf.int32, shape=[None,None], name='short_term_before_product_flag1')
        self.before_openbids = tf.compat.v1.placeholder(tf.int32, shape=[None,None], name='short_term_before_product_flag2')
        self.before_types = tf.compat.v1.placeholder(tf.int32, shape=[None,None], name='short_term_before_product_type')
        self.current_pid_time = tf.compat.v1.placeholder(tf.int32, shape=(None), name='current_product_time')
        self.time_diff = tf.compat.v1.placeholder(tf.float32, shape=[None,None], name='before_pids_time_diff')

        self.num_units = params.num_units

        # batch_size = tf.shape(self.UserID)[0]
        self.productemb = Embed.GetAllTestProductEmbedding()
        self.test_productemb = Embed.GetAllTestProductEmbedding()

        # emb
        self.user_ID_emb = Embed.GetUserEmbedding(self.UserID)

        # current query emb
        self.type_emb = Embed.GetAuctionAttrEmbedding(self.current_type)
        self.before_type_emb = Embed.GetAuctionAttrEmbedding(self.before_types) #64,5,128
        # self.duration_emb, self.openbid_emb = Embed.GetAuctionAttrEmbedding(self.current_duration, self.current_openbid)

        # only have pid
        self.beforeproductemb = Embed.GetProductEmbedding(self.short_term_before_product_id)   #原来的productemb，未添加attr
        self.beforeproductemb_with_attr = tf.concat((self.before_type_emb, self.beforeproductemb),axis=2)

        # 追加type的attr特征
        auctionemb_weights=tf.Variable(tf.random_normal([2 * params.embed_size, params.embed_size]))
        auctionemb_bias=tf.Variable(tf.random_normal([params.embed_size]))
        self.beforeauctionemb = tf.tanh(tf.matmul(self.beforeproductemb_with_attr, auctionemb_weights) + auctionemb_bias)

        # positive emb
        # pos_emb = Embed.GetProductEmbedding(self.current_product_id)
        # pos_emb_with_attr = tf.concat((self.type_emb, pos_emb),axis=1)
        # pos_weights=tf.Variable(tf.random_normal([2 * params.embed_size, params.embed_size]))
        # pos_bias=tf.Variable(tf.random_normal([params.embed_size]))
        # self.product_pos_emb = tf.tanh(tf.matmul(pos_emb_with_attr, pos_weights) + pos_bias)
    
        self.product_pos_emb = Embed.GetProductEmbedding(self.current_product_id)
        self.product_neg_emb = Embed.GetProductEmbedding(self.current_product_neg_id)
        # (64,5) -> (64,5,100)

        # 处理时间衰减比重 (64,5)-> (64,5,128) 但是没有半衰系数
        self.timeDecay = tf.tile(tf.expand_dims(tf.exp(self.time_diff, name=None), axis=2),[1,1,params.embed_size])

        # Define Query-Based Attention LSTM for User's short-term inference
        # 設置LSTM層數，並輸入k个之前的emb然後得到hidden embedding
        self.short_term_lstm_layer = rnn.BasicLSTMCell(self.num_units, forget_bias=1)

        # self.short_term_lstm_outputs,_ = tf.nn.dynamic_rnn(self.short_term_lstm_layer, self.short_term_td_outputs, dtype="float32")  # 加type属性
        self.short_term_td_outputs = tf.nn.softmax(tf.multiply(self.beforeproductemb, self.timeDecay)) #考虑时间差的影响
        self.short_term_lstm_outputs,_ = tf.nn.dynamic_rnn(self.short_term_lstm_layer, self.short_term_td_outputs, dtype="float32")  # hidden: latent representation [64,5,100]

        # transformer:
        with tf.variable_scope("transform"):
            transformer_input = self.beforeproductemb   # [64,5,100]
            # 一些默认设定：
            self.all_encoder_layers = transformer_model(transformer_input)  # [64,5,100]
            self.transformer_output = self.all_encoder_layers[-1]      #[5,100]
            
            self.transformer_layer = tf.layers.dense(
                self.all_encoder_layers,
                units=params.embed_size,
                activation=gelu,
                kernel_initializer=create_initializer(0.02))
            self.transformer_input_tensor = layer_norm(self.transformer_layer)
            # 以上是transformer

        self.user_short_term_emb = tf.reduce_sum(self.transformer_input_tensor,axis=1) # 去掉queryemb，直接降维
        short_term_combine_user_item_emb =  tf.concat([self.user_ID_emb, self.user_short_term_emb], 1)  # short-term preference公式所需

        self.short_term_combine_weights=tf.Variable(tf.random_normal([2* params.embed_size, params.embed_size]))
        self.short_term_combine_bias=tf.Variable(tf.random_normal([params.embed_size]))

        self.short_term_useremb = tf.tanh(tf.matmul(short_term_combine_user_item_emb, self.short_term_combine_weights) + self.short_term_combine_bias)  # short-term preference
        # self.short_term_useremb = self.user_short_term_emb

        if params.user_emb == "Complete":
        #     self.useremb = self.long_term_useremb * self.long_short_rate + (1 - self.long_short_rate) * self.short_term_useremb
        # elif params.user_emb == "Complete":
            self.useremb = self.short_term_useremb
        # elif params.user_emb == "Long_term":
        #     self.useremb = self.long_term_useremb
        else:
            self.useremb = self.user_ID_emb

        # self.long_short_rate = tf.sigmoid(self.long_term_before_product_len - params.short_term_size)
            # (?, 100) <unknown>
        self.opt_loss, self.pos_loss, self.neg_loss = Loss_F(self.useremb, self.user_ID_emb, self.product_pos_emb, self.product_neg_emb, params.neg_sample_num)
        reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        self.opt_loss = self.opt_loss + sum(reg_losses)
        
        
        
        # Optimiser
        step = tf.Variable(0, trainable=False)
        
        self.opt = gradients(
            opt=tf.compat.v1.train.AdamOptimizer(learning_rate=params.learning_rate),
            loss=self.opt_loss,
            vars=tf.compat.v1.trainable_variables(),
            step=step
        )

    # ((uid, cur_before_pids_pos,cur_before_pids_pos_len, cur_before_pids_flag, cur_before_pids_attr,current_pid_pos,cur_pids_attr))
    def step(self, session, uid,sbpid,sbpidlen,time_diff,btypes,cpid,cpnid,cpid_time,cpid_type, testmode = False):
        input_feed = {}
        input_feed[self.UserID.name] = uid
        input_feed[self.short_term_before_product_id.name] = sbpid
        input_feed[self.short_term_before_product_len.name] = sbpidlen
        input_feed[self.time_diff.name] = time_diff
        input_feed[self.before_types.name] = btypes

        input_feed[self.current_product_id.name] = cpid
        input_feed[self.current_product_neg_id.name] = cpnid

        input_feed[self.current_pid_time.name] = cpid_time
        input_feed[self.current_type.name] = cpid_type

        if testmode == False:
            output_feed = [self.opt, self.opt_loss, self.pos_loss, self.neg_loss]
        else:
            #u_plus_q = self.useremb + self.queryemb
            u_plus_q = self.useremb
            output_feed = [tf.shape(self.UserID)[0], u_plus_q, self.test_productemb]

        outputs = session.run(output_feed, input_feed)

        # short_feed = [self.timeDecay,self.beforeproductemb, self.user_ID_emb , self.user_short_term_emb]
        # short = session.run(short_feed, input_feed)
        # for i in range(len(short)):
        #     print(short[i].shape)
        return outputs
        # return short

    def summary(self, session, summarys):
        # input_feed = {}
        # input_feed[self.UserID.name] = uid
        # input_feed[self.current_product_id.name] = cpid
        # input_feed[self.current_product_neg_id.name] = cpnid
        # input_feed[HR.name] = hr
        # input_feed[MRR.name] = mrr
        # input_feed[NDCG.name] = ndcg
        # input_feed[AVG_LOSS.name] = avg_loss
        output_feed = [summarys]
        # outputs = session.run(output_feed, input_feed)
        outputs = session.run(output_feed)
        return outputs