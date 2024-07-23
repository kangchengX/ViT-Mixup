import tensorflow as tf
from typing import Literal


############ vision transformer implementation #############

class MlpBlock(tf.keras.layers.Layer):
    """2-layer mlp block implementaion"""

    def __init__(
            self, 
            dim: int, 
            hidden_dim: int, 
            dropout: float | None = 0.5
    ):
        """
        Initialize the model
        
        Args:
            dim (int): dimension of the inputs and outputs, i.e.  dimension of the words vector.
            hidden_dim (int): hidden dimension.
            dropout (float): dropout percentage. Default to 0.5.
        """
        
        super().__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation=tf.nn.gelu),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(dim),
            tf.keras.layers.Dropout(dropout),
        ])

    def call(self, inputs):
        return self.net(inputs)


class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block implementation"""

    def __init__(
            self, 
            dim: int, 
            num_heads: int, 
            mlp_dim: int, 
            dropout: float | None = 0.5
    ):
        """
        Initialize the model
          
        Args:
            dim (int): dimension of the words vector.
            num_heads (int): number of heads.
            mlp_dim (int): hidden dimension of mlp blocks.
            dropout (float): dropout percentage. Default to 0.5.
        """
        
        super().__init__()
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mlp = MlpBlock(dim, hidden_dim=mlp_dim, dropout=dropout)

    def call(self, inputs):
        # first residual connection flow
        outputs1 = self.norm1(inputs)
        outputs1 = self.attention(query=outputs1, key=outputs1, value=outputs1)
        outputs1 = inputs + outputs1
        
        #second residual connection flow
        outputs2 = self.norm2(outputs1)
        outputs2 = self.mlp(outputs2)
        outputs = outputs1 + outputs2

        return outputs


class ViT(tf.keras.Model):
    """Visiton trandormer implementation"""

    def __init__(
            self, 
            image_size: int, 
            patch_size: int, 
            num_classes: int, 
            dim: int, 
            depth: int, 
            num_heads: int, 
            mlp_dim: int, 
            dropout: float | None = 0.5
    ):
        """
        Initialize the model
        
        Args:
            image_size (int): width or height of input images
            patch_size (int): width or height of patchs
            num_classes (int): number of the classes
            dim (int): dimension of the words vector
            depth (int): number of transformer blocks
            heads (int): number of heads
            mlp_dim (int): dimension of feddforward blocks
            dropout (float): dropout percentage. Default to 0.5
        """
        
        super().__init__()
        # sizes and shapes
        self.patch_size = patch_size
        self.dim = dim
        self.patch_dim = patch_size * patch_size * 3

        num_patches = (image_size // patch_size) ** 2

        # embedding
        self.pos_embedding = self.add_weight(name="position_embeddings", 
                                             shape=(1, num_patches + 1, dim), 
                                             initializer=tf.random_normal_initializer())
        self.cls_token = self.add_weight(name="cls_token", 
                                         shape=(1, 1, dim), 
                                         initializer=tf.random_normal_initializer())
        
        # initial layers/blocks
        self.patch_proj = tf.keras.layers.Dense(dim)
        self.transformer_blocks = tf.keras.Sequential(
            [TransformerBlock(dim, num_heads, mlp_dim, dropout) for _ in range(depth)])
        self.to_cls_token = tf.identity
        self.mlp_head = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(epsilon=1e-6),
            tf.keras.layers.Dense(mlp_dim, activation=tf.nn.gelu),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(num_classes),
        ])

    def call(self, images, training = False):
        shapes = tf.shape(images)
        batch_size, _, _, _ = tf.unstack(shapes)

        #image to flattened patches
        outputs = tf.image.extract_patches(images=images,
                                     sizes=[1, self.patch_size, self.patch_size, 1],
                                     strides=[1, self.patch_size, self.patch_size, 1],
                                     rates=[1, 1, 1, 1],
                                     padding="VALID")
        outputs = tf.reshape(outputs, [batch_size, -1, self.patch_dim])

        # flattened pathes to word vectors
        outputs = self.patch_proj(outputs)

        # cls token and position embedding
        cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, self.dim])
        outputs = tf.concat([cls_tokens, outputs], axis=1)
        outputs += self.pos_embedding
        
        # feed word vectors to the network
        outputs = self.transformer_blocks(outputs,training=training)
        outputs = self.to_cls_token(outputs[:, 0])
        outputs =  self.mlp_head(outputs, training=training)

        return outputs
    

######## vision transformer with augmentation implementation ########

def random_beta(alpha: float, beta: float):
    """
    Generate varible from Beta distribution
    
    Args: 
        alpha (float): parameter for beta distribution, i.e. Beta(alpha, beta)
        beta (float): parameter for beta distribution, i.e. Beta(alpha, beta)

    Returns:
        an observation from Beta distribution
    """

    gamma_alpha = tf.random.gamma(shape=[], alpha=alpha)
    gamma_beta = tf.random.gamma(shape=[], alpha=beta)

    return gamma_alpha / (gamma_alpha + gamma_beta)


class MixUp(tf.keras.layers.Layer):
    """Augmentation method mixup implementation"""

    def __init__(self, sampling_method: Literal['beta','uniform'], **kwargs):
        """
        Initialize the model
        
        Args:
            sampling_method (str): method to generate lambda. 'beta' indicates beta, 'uniform' indicate uniform
            alpha (float): parameter for beta distribution
            uniform_range (tuple): predefined range to generate lambda uniformly
        """
        super().__init__()
        self.index = None
        self.lam = None
        self.sampling_method = sampling_method

        # get method type and check necessary args
        if sampling_method == 'beta':
            alpha = kwargs.get('alpha', None)
            if alpha is None:
                raise ValueError('missing argument alpha for sampling_method = beta')
            self.lam_func = lambda : random_beta(alpha,alpha)

        elif sampling_method == 'uniform':
            uniform_range = kwargs.get('uniform_range', None)
            if uniform_range is None:
                raise ValueError('missing argument uniform_range for sampling_method = uniform')
            self.lam_func = lambda : tf.random.uniform(shape=[], minval=uniform_range[0], maxval=uniform_range[1])
        
        else:
            raise ValueError(f"sampling_method is required to be 'beta' or 'uniform', while the input is {sampling_method}")

    def call(self, inputs, training):
        if training:
            # augmentation during training
            index = tf.range(start=0, limit=inputs.shape[0], dtype=tf.int32)
            index = tf.random.shuffle(index)
            self.index = index
            self.lam = self.lam_func()
            outputs = inputs * self.lam + tf.gather(inputs,self.index,axis=0) * (1-self.lam)
        else:
            outputs = inputs

        return outputs


class VitAug(tf.keras.Model):
    """Vit with augmentation method MixUp"""

    def __init__(
            self, 
            sampling_method: Literal['beta','uniform'], 
            image_size: int, 
            patch_size: int, 
            num_classes: int, 
            dim: int, 
            depth: int, 
            num_heads: int, 
            mlp_dim: int, 
            dropout: float | None = 0.5, 
            **kwargs
    ):
        """
        Initialize the model
        
        Args:
            sampling_method (str): method to generate lambda. 'beta' indicates beta, 'uniform' indicates uniform
            image_size (int): width or height of input images
            patch_size (int): width or height of patchs
            num_classes (int): number of the classes
            dim (int): dimension of the words vector
            depth (int): number of transformer blocks
            num_heads (int): number of heads
            mlp_dim (int): dimension of mlp blocks
            dropout (int): dropout percentage
            alpha (int): parameter for beta distribution
            uniform_range (tuple): predefined range to generate lambda uniformly
        """
        
        super().__init__()
        self.aug = MixUp(sampling_method,**kwargs)
        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes, 
            dim=dim, 
            depth=depth, 
            num_heads=num_heads, 
            mlp_dim=mlp_dim,
            dropout=dropout
        )

    def call(self, inputs, training = False):
        outputs = self.aug(inputs, training = training)
        outputs = self.vit(outputs, training = training)

        return outputs