# Make patch size 14
# Make max height/width 392
# Limit time/slice if needed
# Make output representation 768

class Config(object):
    def __init__(
            self,
            max_height=384,
            max_width=384,
            max_time=5,
            max_slice=5,
            num_channels=3,
            
            patch_size=24,
            representation_size=768,
            
            num_layers=12,
            num_secondary_layers=4,
            num_heads=12,
            hidden_dim=576,
            mlp_dim=1024,
            dropout=0.0,
            attention_dropout=0.0,
            mask_prob=0.15,
            
            batch_size=8,
            effective_batch_size=64,
            lr=5e-4,
            lr_rampup=1000,
            tot_steps=100000,
            num_workers=8,
            
            downstream_epochs=5,
            downstream_batch_size=4,
            downstream_effective_batch_size=64,
            downstream_lr=1e-3,
    ):
        self.max_height = max_height
        self.max_width = max_width
        self.max_time = max_time
        self.max_slice = max_slice
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.representation_size = representation_size
        self.num_layers = num_layers
        self.num_secondary_layers = num_secondary_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.mask_prob = mask_prob
        self.batch_size = batch_size
        self.effective_batch_size = effective_batch_size
        self.lr = lr
        self.lr_rampup = lr_rampup
        self.tot_steps = tot_steps
        self.num_workers = num_workers
        self.downstream_epochs = downstream_epochs
        self.downstream_batch_size = downstream_batch_size
        self.downstream_effective_batch_size = downstream_effective_batch_size
        self.downstream_lr = downstream_lr
