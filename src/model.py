from keras import Model
from keras.layers import Dense, Conv1D, GlobalAveragePooling1D, Dropout, LayerNormalization, MultiHeadAttention


class TransformerModel(Model):
    def __init__(self, input_shape, output_node_activation, n_output_nodes, n_head_nodes, n_heads, n_filters, n_transformer_blocks, n_mlp_units, mlp_dropout_rate, base_dropout_rate):
        super(TransformerModel, self).__init__()
        self.n_transformer_blocks = n_transformer_blocks
        self.mlp_dropout_rate = mlp_dropout_rate
        self.layer_norm1 = [LayerNormalization(epsilon=1e-6) for _ in range(n_transformer_blocks)]
        self.multi_head_attention = [MultiHeadAttention(key_dim=n_head_nodes, num_heads=n_heads, dropout=base_dropout_rate) for _ in range(n_transformer_blocks)]
        self.dropout1 = [Dropout(base_dropout_rate) for _ in range(n_transformer_blocks)]
        self.layer_norm2 = [LayerNormalization(epsilon=1e-6) for _ in range(n_transformer_blocks)]
        self.conv1 = [Conv1D(filters=n_filters, kernel_size=1, activation='relu') for _ in range(n_transformer_blocks)]
        self.dropout2 = [Dropout(base_dropout_rate) for _ in range(n_transformer_blocks)]
        self.conv2 = [Conv1D(filters=input_shape[-1], kernel_size=1) for _ in range(n_transformer_blocks)]
        self.mlp_layers = [Dense(dim, activation='relu') for dim in n_mlp_units]
        self.mlp_dropout = Dropout(mlp_dropout_rate)
        self.output_layer = Dense(n_output_nodes, activation=output_node_activation)
        self.global_avg_pool = GlobalAveragePooling1D(data_format='channels_first')

    def call(self, inputs):
        x = inputs
        for i in range(self.n_transformer_blocks):
            res = x
            x = self.layer_norm1[i](x)
            x = self.multi_head_attention[i](x, x)
            x = self.dropout1[i](x)
            x = x + res
            res = x
            x = self.layer_norm2[i](x)
            x = self.conv1[i](x)
            x = self.dropout2[i](x)
            x = self.conv2[i](x)
            x = x + res
        x = self.global_avg_pool(x)
        for dense_layer in self.mlp_layers:
            x = dense_layer(x)
            x = self.mlp_dropout(x)
        outputs = self.output_layer(x)
        return outputs
    