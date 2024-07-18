# Original patch size 16
# Adjust Max height/width
# Increase Max slice if possible
# Note original 400 epochs of 1.2 million images, ~500 million images with batch size 256 so 2 million steps

class Config(object):
    def __init__(
            self,
            max_height=224,
            max_width=224,
            max_time=5,
            max_slice=5,
            num_channels=3,
            
            patch_size=14,
            representation_size=768,
            
            num_layers=12,
            num_secondary_layers=4,
            num_heads=12,
            projection_size=8192,
            mlp_dim=1024,
            dropout=0.0,
            attention_dropout=0.0,
            mask_prob=0.3,
            student_temp=0.1,
            teacher_cls_temp=0.04,
            teacher_patch_temp=0.07,
            momentum=0.999,
            center_momentum=0.9,
            
            batch_size=8,
            effective_batch_size=64,
            lr=1e-5,
            lr_rampup=10000,
            tot_steps=500000,
            num_workers=8,
            
            downstream_epochs=5, # 10 TODO: Linear Probing
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
        self.projection_size = projection_size
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.mask_prob = mask_prob
        self.student_temp = student_temp
        self.teacher_cls_temp = teacher_cls_temp
        self.teacher_patch_temp = teacher_patch_temp
        self.momentum = momentum
        self.center_momentum = center_momentum
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
