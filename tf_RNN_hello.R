rm(list=ls())
# Lab 12 RNN
library(tensorflow)
library(dplyr)
tf$set_random_seed(777)  # reproducibility

idx2char <- data.frame(cha=c('h', 'i', 'e', 'l', 'o'))

# Teach hello: hihell -> ihello
x_data <- matrix(c(0, 1, 0, 2, 3, 3), nrow=1) #hihell
one_hot <- with(idx2char, data.frame(model.matrix(~cha-1,idx2char),cha))
one_hot <- one_hot[,c("chah","chai", "chae", "chal", "chao", "cha")]
x_one_hot <- apply(x_data, 2, function(x) one_hot[x+1,]) %>%
  bind_rows() %>%
  select(-cha) %>%
  as.matrix()

y_data <- matrix(c(1, 0, 2, 3, 3, 4), nrow=1) #ihello


num_classes <- 5L
input_dim <- 5L  # one-hot size
hidden_size <- 5L  # output from the LSTM. 5 to directly predict one-hot
batch_size <- 1L   # one sentence
sequence_length <- 6L  # |ihello| == 6

x_one_hot_var = array(x_one_hot, dim=c(1, sequence_length, hidden_size))

X = tf$placeholder(
  tf$float32, shape(NULL, sequence_length, hidden_size))  # X one-hot
Y = tf$placeholder(tf$int32, shape(NULL, sequence_length))  # Y label

cell = tf$contrib$rnn$BasicLSTMCell(num_units=hidden_size, state_is_tuple=TRUE)
initial_state = cell$zero_state(batch_size, tf$float32)

out = tf$nn$dynamic_rnn(cell, X, initial_state = initial_state, dtype=tf$float32)
outputs = out[[1]]
states = out[[2]]

# FC layer
X_for_fc = tf$reshape(outputs, shape(-1L, hidden_size))
# fc_w = tf.get_variable("fc_w", [hidden_size, num_classes])
# fc_b = tf.get_variable("fc_b", [num_classes])
# outputs = tf.matmul(X_for_fc, fc_w) + fc_b
outputs = tf$contrib$layers$fully_connected(
  inputs=X_for_fc, num_outputs=num_classes, activation_fn=NULL)

# reshape out for sequence_loss
outputs = tf$reshape(outputs, shape(batch_size, sequence_length, num_classes))

weights = tf$ones(shape(batch_size, sequence_length))
sequence_loss = tf$contrib$seq2seq$sequence_loss(
  logits=outputs, targets=Y, weights=weights)
loss = tf$reduce_mean(sequence_loss)
train = tf$train$AdamOptimizer(learning_rate=0.1)$minimize(loss)

prediction = tf$argmax(outputs, axis=2L)

with(tf$Session() %as% sess, {
  sess$run(tf$global_variables_initializer())

  for(i in seq(50)){
    semi_result = sess$run(list(loss, train), feed_dict=dict(X=x_one_hot_var, Y=y_data))
    l = semi_result[[1]]
    result = sess$run(prediction, feed_dict=dict(X= x_one_hot_var))
    cat(paste(i, "loss:", l, "prediction: ", result, "true Y: ", y_data, "\n"))
    
    # print char using dic
    result_str = apply(result, 2, function(x) idx2char[x+1,])
    cat(paste("\tPrediction str: ", paste(result_str, collapse=""), "\n"))
  }
})
