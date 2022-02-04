class ConfigBiaffine:

    def __init__(self, **kwargs):
        # self.word_embedding_dim = kwargs.pop('word_embedding_dim', 300)
        # self.pos_embedding_dim = kwargs.pop('pos_embedding_dim', 300)
        # self.lstm_hidden_size_in = self.word_embedding_dim + self.pos_embedding_dim
        # self.lstm_hidden_size_out = kwargs.pop('lstm_hidden_size_out', 500)
        # self.lstm_num_layers = 3
        #
        # self.mlp_arc_head_dim = kwargs.pop('mlp_arc_head_dim', 500)
        # self.mlp_arc_dep_dim = kwargs.pop('mlp_arc_dep_dim', 500)
        # self.mlp_lbl_dim = kwargs.pop('mlp_lbl_dim', 500)
        #
        # self.dropout = kwargs.pop('dropout', 0.3)
        self.word_embedding_dim = kwargs.pop('word_embedding_dim', 300)
        self.pos_embedding_dim = kwargs.pop('pos_embedding_dim', 50)
        self.lstm_hidden_size_in = self.word_embedding_dim + self.pos_embedding_dim
        self.lstm_hidden_size_out = kwargs.pop('lstm_hidden_size_out', 300)
        self.lstm_num_layers = 3

        self.mlp_arc_head_dim = kwargs.pop('mlp_arc_head_dim', 512)
        self.mlp_arc_dep_dim = kwargs.pop('mlp_arc_dep_dim', 512)
        self.mlp_lbl_dim = kwargs.pop('mlp_lbl_dim', 128)

        self.dropout = kwargs.pop('dropout', 0.3)


class DependencyEvaluator:

    def eval(self, attention_mask, head_pred, label_pred, head_gold, label_gold):
        lengths = attention_mask.sum(-1)
        mask = (head_pred == head_gold) * attention_mask
        uas = (mask.sum(-1)/lengths).mean()
        las = (((label_pred == label_gold)*mask).sum(-1)/lengths).mean()
        summary = {
            'uas': uas.item(),
            'las': las.item()
        }
        return summary
