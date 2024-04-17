from functools import partial

import flax.linen as nn
import jax.numpy as jnp

# from dreamerv3 import ninjax as nj


class MLP(nn.Module):
    hidden_features: int = None
    out_features: int = None
    act_layer: callable = nn.gelu
    drop: float = 0.0

    @nn.compact
    def __call__(self, x, training=False):
        out_features = self.out_features or x.shape[-1]
        hidden_features = self.hidden_features or x.shape[-1]
        x = nn.Dense(hidden_features)(x)
        x = self.act_layer(x)
        x = nn.Dropout(self.drop, deterministic=not training)(x)
        x = nn.Dense(out_features)(x)
        x = nn.Dropout(self.drop, deterministic=not training)(x)
        return x


class Attention(nn.Module):
    num_heads: int = 8
    qkv_bias: bool = False
    qk_scale: float = None
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    use_sdpa: bool = True

    @nn.compact
    def __call__(self, x, mask=None):
        B, N, C = x.shape
        qkv = nn.Dense(C * 3, use_bias=self.qkv_bias)(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(
            (2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]
        if self.use_sdpa:
            x = nn.dot_product_attention(q, k, v, dropout_rate=self.proj_drop)
            attn = None
        else:
            attn = (
                jnp.matmul(q, k.transpose(0, 1, 3, 2)) * self.qk_scale
            )  # [B, num_heads, D, D]
            attn = nn.softmax(attn, axis=-1)
            attn = nn.Dropout(self.attn_drop)(attn)
            x = jnp.matmul(attn, v)
        x = x.transpose(0, 2, 1, 3).reshape(B, N, C)
        x = nn.Dense(C, kernel_init=nn.initializers.normal(stddev=self.proj_drop))(x)
        return x, attn


class Block(nn.Module):
    num_heads: int
    mlp_ratio: float = 4.0
    qkv_bias: bool = False
    qk_scale: float = None
    drop: float = 0.0
    attn_drop: float = 0.0
    act_layer: callable = nn.gelu
    norm_layer: callable = nn.LayerNorm

    @nn.compact
    def __call__(self, x, return_attention=False, mask=None, training=False):
        y, attn = Attention(
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            attn_drop=self.attn_drop,
            proj_drop=self.drop,
        )(self.norm_layer()(x), mask=mask)
        if return_attention:
            return attn
        x = x + y
        x = x + MLP(
            hidden_features=int(x.shape[-1] * self.mlp_ratio),
            act_layer=self.act_layer,
            drop=self.drop,
        )(self.norm_layer()(x), training=training)
        return x


class CrossAttention(nn.Module):
    num_heads: int = 12
    qkv_bias: bool = False

    @nn.compact
    def __call__(self, q, x):
        B, n, C = q.shape
        q = (
            nn.Dense(C, use_bias=self.qkv_bias)(q)
            .reshape(B, n, self.num_heads, C // self.num_heads)
            .transpose((0, 2, 1, 3))
        )
        B, N, C = x.shape
        kv = (
            nn.Dense(int(C * 2), use_bias=self.qkv_bias)(x)
            .reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .transpose((2, 0, 3, 1, 4))
        )
        k, v = kv[0], kv[1]  # (batch_size, num_heads, seq_len, feature_dim_per_head)

        xattn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * (C // self.num_heads) ** -0.5
        xattn = nn.softmax(
            xattn, axis=-1
        )  # (batch_size, num_heads, query_len, seq_len)
        q = jnp.matmul(xattn, v)
        q = q.transpose(0, 2, 1, 3).reshape(B, n, C)
        q = nn.Dense(C)(q)
        return q


class CrossAttentionBlock(nn.Module):
    num_heads: int
    mlp_ratio: float = 4.0
    qkv_bias: bool = False
    act_layer: callable = nn.gelu
    norm_layer: callable = nn.LayerNorm

    @nn.compact
    def __call__(self, q, x, training=False):
        y = CrossAttention(num_heads=self.num_heads, qkv_bias=self.qkv_bias)(
            q, self.norm_layer()(x)
        )
        q = q + y
        q = q + MLP(
            hidden_features=int(q.shape[-1] * self.mlp_ratio), act_layer=self.act_layer
        )(self.norm_layer()(q), training=training)
        return q


class AttentivePooler(nn.Module):
    num_queries: int = 1
    num_heads: int = 12
    mlp_ratio: float = 4.0
    depth: int = 1
    norm_layer: callable = nn.LayerNorm
    init_std: float = 0.02
    qkv_bias: bool = True
    complete_block: bool = True

    @nn.compact
    def __call__(self, x, training=False):
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], -1, x.shape[-1])

        query_tokens = self.param(
            "query_tokens",
            nn.initializers.normal(stddev=self.init_std),
            (1, self.num_queries, x.shape[-1]),
        )
        if self.complete_block:
            cross_attention_block = CrossAttentionBlock(
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                norm_layer=self.norm_layer,
            )
            cross_attention_block = partial(cross_attention_block, training=training)
        else:
            cross_attention_block = CrossAttention(
                num_heads=self.num_heads, qkv_bias=self.qkv_bias
            )
        blocks = None
        if self.depth > 1:
            blocks = [
                Block(
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    qk_scale=False,
                    norm_layer=self.norm_layer,
                )
                for i in range(self.depth - 1)
            ]
        q = jnp.repeat(query_tokens, len(x), axis=0)
        q = cross_attention_block(q, x)
        if blocks is not None:
            for blk in blocks:
                q = blk(q, training=training)
        return q


# _njAttentivePooler = functools.partial(nj.FlaxModule, AttentivePooler)


# class ImageAttentivePooler(nj.Module):
#     def __call__(self, x):
#         x = self.get("attn_pool", _njAttentivePooler)(x)
#         return self.linear(x)


# pooler = AttentivePooler()
# params = pooler.init(jax.random.PRNGKey(42), jnp.empty((1, 1, 768)))
# x = jnp.zeros((4, 10, 768))
# q = pooler.apply(params, x)
# print(q.shape)


# attentive_pooler = nj.jit(nj.pure(ImageAttentivePooler(name="attn")))
# state = {}
# state, out = attentive_pooler(state, 42, x)
