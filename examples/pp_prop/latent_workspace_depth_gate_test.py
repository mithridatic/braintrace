"""Tests for the preregistered Example 21 demonstrated-depth Gate B."""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import inspect
import msgspec_json
import math
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import brainstate
import braintrace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from braintrace._testing.oracle import (
    chunked_online_param_gradients,
    flat_gradient_leaves,
    gradient_norm,
    relative_deviation,
)
from examples.pp_prop import latent_workspace_binding_control as legacy
from examples.pp_prop import latent_workspace_binding_gate_launcher as launcher
from examples.pp_prop import latent_workspace_depth_gate as depth
from examples.pp_prop.latent_workspace_model import LatentWorkspaceModel
from examples.pp_prop.latent_workspace_task import (
    ArcGrid,
    ArcPair,
    ArcTask,
    RowEventConfig,
    associative_memory_feature_indices,
    encode_query_episode,
)


_PRODUCTION_ENCODED_GLOBAL_SHA256 = {
    "events": "a1937b7f8d5d4da5f30216847cc63d022d9ec46d5cf152b25f5a30a59a1eb84f",
    "targets": "4082d2fd1440e9d14b0c81c754158f05b8056137a9116aee667f8d112312184c",
    "loss_weights": "044616bf9dd86cbdc1d472184ede8027bf9ff65d65834b15ec619bf3095d2e31",
    "advance_masks": "2fc1b2acd9f73e567684d2a85f44c4009c5941ce262a527589066117ec27a4cc",
    "mapping_ids": "78c2d8aaa9e874dbcc1c25363875ff8aec0356a711d2426e09f2e79c76c72cb7",
    "efforts": "c7ca75132501bda8e6b5695a48a1ae5cde22da587f4658f7721bd4e3adcd58e6",
    "query_colors": "38b4cecef323dce16b0478fdd3874c9383804c913c39aaf017ce34554dcd37cb",
    "presentation_orders": "0650be382b381d7ab14b642c6fcdb16ae410e70a4c5821b10643bce41e3f7ca5",
}
_PRODUCTION_CHUNK_SHA256_MANIFEST = {
    "events": "c73cb65b9b774f07e617000f531511a561c2672cd5c5dc0bfb5c95328bded771",
    "targets": "e969705e4d9ab68581cde897f5f8fd780a716964d9a96a842b3e9850db294f3e",
    "loss_weights": "968eed06f7bc37d52d68300f2ef803666320c2ffaa094843876f4c3b39101b9c",
    "advance_masks": "226ee64d8c3ffbe5eec66bb8cc01c7030b2d3c700eb2c75da91b5391e659cfd8",
    "mapping_ids": "d198e47d766ad69c2691243fd759835d4a3d5d79377f8d47cc9bbf8488306a84",
    "efforts": "267aef30ba47f5e67c17a36d299ffe2a41acfe57c92a078f4d04cc8f7cea903b",
    "query_colors": "f67148579356209f6846d681eeaafa3e00ccf9206023f59a7837d1c351a58da9",
    "presentation_orders": "60b52825e54e2b2b5efbc9f2cbaea65ef3737577b7b4c2e51b924ea9e907a8ef",
}
_PRODUCTION_CHUNK_SHA256_HEX = {
    "events": (
        "a8027e2fa69ab19d331edee4e30e4723ae7b03b4bdc7f4443ce4f14f9ae6c83c3e601beb31dd7832a5fb206e1040911c"
        "a358adf3744a01c8b5137cbc62ae3d36c632355bb8ee7eb44ac03d02bea164c13b33f2d2fd4b839fab09e80abf125a46"
        "2f7bd21477706acc7aa678fb72b61dc6ff82c20a02f189fffeca568c3a19169f8296f93761e59ef623ef8ac2a222144f"
        "da67f9ee53bf95c6ec13b32604e95aecc2e55cab6c3ba2cb258e2b4654aaa3c7cf30939a1ae9a59d5f7966ce8c56ad56"
        "8580904870102c4f9205016b2e93dbb0a968963a9a0494173894abe789ba529383677a80882c980c50dfaf7bc67ba468"
        "7042032640b27dff5356f72aae1b01b2b28b5f21b07c51ca59dff65b798290ef24064b34ddc258192b94ff33fbd560a3"
        "e66abb9a06cee08f706bfcdcf5e72b0c9875c166194ac97b2a2ce2921779880e2e20d629584f31daeceb476f47de6a2c"
        "74b2ce277335c5827568ade205bf0aaa233f6ddb91c5f8af7745033be2b8d502bffc920fd048986e0284e1a80273e85a"
        "79111f838054ed86a15d18c2d1cc6800b19eeec2fd6d86a8d2b620ff897aee39ec4eff277148ff342986677369c53084"
        "e6df816a93bb70fe576da7f24efd68987b1bd33e576a9613ee9e0d2af7bb8686e5c8951e28b25ce3bdeafbaf71c07f13"
        "6c8147a2df44f8d7b6f433d685451be2b8d04a18f71997188f3746f5c910f5a3adb8671b511bfa5a96a8df8d90dce8a2"
        "2b74983c3d7c24e91d813128c90b7ac53bd25e38f1fd0b0e272d814e73391e7765fb22dc8982c766e7bf4339278c3dbd"
        "015a7c17347712857ea7d914e6dc2416d56f76f0350539cf0b84b96be57c693c8816ee5b104f6a7dcfaadb3cd9e7e858"
        "5d2c8b9c916eeee0720b589f7393afe46855f2275330bdfb2f680eece71e47cc6507f61f71daf77839fd07c2f261cb77"
        "b54266cf29a7ecc30a5ac72869560fa8b19236790a6c17ce7191f8e10967ef798ba9d63d8f1abe0449e0fc1069a19d3f"
        "2a4dc841a65e04d05e14d7489a7e47d1f6b33f6bfc8e0cfd678fdc4298c8b99ab5dff64cd33aaaa9e333e394f4ff64db"
        "7f25dc4961a4717b2392405c6cb2b2b275f162996fd2aa6ff92c90812a89e152778b1870328f5c8e9e337ebd3898617e"
        "462d692a0cfff2a9f9008c10df56c2fd7ee1c4491d05b036366e2bfac72a05756742d19ff8b8c1c813d9367c27d5650a"
        "a54a00140d2888cf05d1296031ea7707fb5233844351f1a86374c7176d01baf9fd17ac264c7f676c0af2ccf18a59a4ec"
        "a8102e3aa812116646a7ef0686c2cc64a4a04781b5890976d75d1e2bc845e17fa9046daa6b73122ef5e0918fad5d3ce4"
        "cd6fae8973482c21ae1257b36594677287036f38a2a169908a1522c06950475b3fd4494a24c8fa93a34cfe3c742ffa1f"
        "6d176a9192fa6765f685fcc156a61a2e"
    ),
    "targets": (
        "58d8473bd04e96e6ce64866259a667024474f34c624cd903281a47f69e067c3f9cd244792dfe22929764392b89f3c4c2"
        "8940c3adcfdbc3beaad4c31d9fb0f470d0a623ef79aa656d9e92e6a918f9f219211db564f5f7bc8c88766fb10a7f6066"
        "d06fb9bdc9c15f8816951a0fd04c130a7db283999ae16646882afe9718a8b3351a57725d538dfbb17b1829f1a1c11eb2"
        "bce276b445479c0919e6ec091aa0135ea322606c522213ad130612a22e1bd104330b8a2bd5797f56b563cf20298dc504"
        "3f927d5637b6f0dea0d90531c75899879e2345a37380a72aebcf093826c1f03b10886268b5567c4c6918a582e43cc596"
        "5eaa16d9aaa411d63eec66a6d0f26fd07bf430cb607d180974615c08d79ea3c748788ad1ac12d1b572ae69b2eddca749"
        "629410ca8eb72a06a83a963e85b22a38ccb3fabca0cedb515ccba362e59b14199fbac1eeb9b00cbf90d8cc6d48a34411"
        "7aa5de6f175cc12467a13e0c5340ef1ac87df42c5251e1e3226f62570910a3761f7bb1289d8174b097bed7b3941047d0"
        "46c7a98812dfdb3a4d0129ebead3df259d940163245ee0b71060839e0f5fdb8fbdacbdf7a772893503196a0fd7735b97"
        "f15aeeac6c097db45b4b7c37da94f77e9f97663417dc115d0b7759faa85916f1865c9c688bd1f74ca85ae1f43e263910"
        "026da72d56571e624febb69b0123d41dcd50ebceaa77c3b53a7f398d943dd1f506395ab0f5f8643883b3babf61f8e14e"
        "210036400db48d82104747821334b2f00de8508b32c403ef2b29f0dc0a3f0ef6c2f6a3a2f349e9fe173abe9e2182988e"
        "efaa8c3f73b5329290fa399c30b18eed2e3d533a6440be2fc6bf8cabf616e5b478a1dca14533d60a7d11c8736e786c1d"
        "3f7fe2f74c13dd44b21045fb2295669a2486e08a766d29b6191e6c313d9eaa14b1dc5af53894ac8431b839cd52b468a4"
        "a092f343c39754c00308fffc30f897e3fc4865f7e823c9f8fd6d7aa7e11502b9b65d9dd10075879be727ea9ea99b3327"
        "01789c91ae2a5e37c11f7a44ac796f829b8fb2880139ea85d5bfb620c177cdc29069c1fa530cfd9d4a36dca5f7bbcf94"
        "a24d1348dadb13decf4171e2953c0c9127062b02f5adfc6360edb550a3fb81f56ba0d141b3f34b575567740942d703bc"
        "bb76ee6b4154f6b69a63d9c216921539c36910134a04af1d9ab4a104978da5e3e9a402a258de8c11f9b609d99d128571"
        "260450c317876d163fe7b81a56a059293d2de1d43579839df696e778bab1c799ad2cd9b5b63d92ffdeae142e714d533e"
        "a6fa0b83c1e0beef822db6b0fce897948d8f7c2ecb1c026f64aea93193e7c2c216309dba3053089b40397dd4c26a7d1c"
        "40cc17dc69a9dbe9d5dcff974a856581797dfbfe9195bac86d1d8204e7b2cbd9ac7f794b69627052bdd40a93f0ca0387"
        "8dd71d16760a266d79229e0076a2515c"
    ),
    "loss_weights": (
        "4b3dab18b230348ee7c04bc8e4e2edd809314f94405e65d9cfc073ec6058eaf04b3dab18b230348ee7c04bc8e4e2edd8"
        "09314f94405e65d9cfc073ec6058eaf04b3dab18b230348ee7c04bc8e4e2edd809314f94405e65d9cfc073ec6058eaf0"
        "4b3dab18b230348ee7c04bc8e4e2edd809314f94405e65d9cfc073ec6058eaf04b3dab18b230348ee7c04bc8e4e2edd8"
        "09314f94405e65d9cfc073ec6058eaf04b3dab18b230348ee7c04bc8e4e2edd809314f94405e65d9cfc073ec6058eaf0"
        "4b3dab18b230348ee7c04bc8e4e2edd809314f94405e65d9cfc073ec6058eaf04b3dab18b230348ee7c04bc8e4e2edd8"
        "09314f94405e65d9cfc073ec6058eaf04b3dab18b230348ee7c04bc8e4e2edd809314f94405e65d9cfc073ec6058eaf0"
        "4b3dab18b230348ee7c04bc8e4e2edd809314f94405e65d9cfc073ec6058eaf04b3dab18b230348ee7c04bc8e4e2edd8"
        "09314f94405e65d9cfc073ec6058eaf04b3dab18b230348ee7c04bc8e4e2edd809314f94405e65d9cfc073ec6058eaf0"
        "4b3dab18b230348ee7c04bc8e4e2edd809314f94405e65d9cfc073ec6058eaf04b3dab18b230348ee7c04bc8e4e2edd8"
        "09314f94405e65d9cfc073ec6058eaf04b3dab18b230348ee7c04bc8e4e2edd809314f94405e65d9cfc073ec6058eaf0"
        "4b3dab18b230348ee7c04bc8e4e2edd809314f94405e65d9cfc073ec6058eaf04b3dab18b230348ee7c04bc8e4e2edd8"
        "09314f94405e65d9cfc073ec6058eaf04b3dab18b230348ee7c04bc8e4e2edd809314f94405e65d9cfc073ec6058eaf0"
        "4b3dab18b230348ee7c04bc8e4e2edd809314f94405e65d9cfc073ec6058eaf04b3dab18b230348ee7c04bc8e4e2edd8"
        "09314f94405e65d9cfc073ec6058eaf04b3dab18b230348ee7c04bc8e4e2edd809314f94405e65d9cfc073ec6058eaf0"
        "4b3dab18b230348ee7c04bc8e4e2edd809314f94405e65d9cfc073ec6058eaf04b3dab18b230348ee7c04bc8e4e2edd8"
        "09314f94405e65d9cfc073ec6058eaf04b3dab18b230348ee7c04bc8e4e2edd809314f94405e65d9cfc073ec6058eaf0"
        "4b3dab18b230348ee7c04bc8e4e2edd809314f94405e65d9cfc073ec6058eaf04b3dab18b230348ee7c04bc8e4e2edd8"
        "09314f94405e65d9cfc073ec6058eaf04b3dab18b230348ee7c04bc8e4e2edd809314f94405e65d9cfc073ec6058eaf0"
        "4b3dab18b230348ee7c04bc8e4e2edd809314f94405e65d9cfc073ec6058eaf04b3dab18b230348ee7c04bc8e4e2edd8"
        "09314f94405e65d9cfc073ec6058eaf04b3dab18b230348ee7c04bc8e4e2edd809314f94405e65d9cfc073ec6058eaf0"
        "4b3dab18b230348ee7c04bc8e4e2edd809314f94405e65d9cfc073ec6058eaf04b3dab18b230348ee7c04bc8e4e2edd8"
        "09314f94405e65d9cfc073ec6058eaf0"
    ),
    "advance_masks": (
        "23cfcb95b51ce9c86e64ca9a1b8379faac70a62233c3c07f02b502b4e294027e23cfcb95b51ce9c86e64ca9a1b8379fa"
        "ac70a62233c3c07f02b502b4e294027e23cfcb95b51ce9c86e64ca9a1b8379faac70a62233c3c07f02b502b4e294027e"
        "23cfcb95b51ce9c86e64ca9a1b8379faac70a62233c3c07f02b502b4e294027e23cfcb95b51ce9c86e64ca9a1b8379fa"
        "ac70a62233c3c07f02b502b4e294027e23cfcb95b51ce9c86e64ca9a1b8379faac70a62233c3c07f02b502b4e294027e"
        "23cfcb95b51ce9c86e64ca9a1b8379faac70a62233c3c07f02b502b4e294027e23cfcb95b51ce9c86e64ca9a1b8379fa"
        "ac70a62233c3c07f02b502b4e294027e23cfcb95b51ce9c86e64ca9a1b8379faac70a62233c3c07f02b502b4e294027e"
        "23cfcb95b51ce9c86e64ca9a1b8379faac70a62233c3c07f02b502b4e294027e23cfcb95b51ce9c86e64ca9a1b8379fa"
        "ac70a62233c3c07f02b502b4e294027e23cfcb95b51ce9c86e64ca9a1b8379faac70a62233c3c07f02b502b4e294027e"
        "23cfcb95b51ce9c86e64ca9a1b8379faac70a62233c3c07f02b502b4e294027e23cfcb95b51ce9c86e64ca9a1b8379fa"
        "ac70a62233c3c07f02b502b4e294027e23cfcb95b51ce9c86e64ca9a1b8379faac70a62233c3c07f02b502b4e294027e"
        "23cfcb95b51ce9c86e64ca9a1b8379faac70a62233c3c07f02b502b4e294027e23cfcb95b51ce9c86e64ca9a1b8379fa"
        "ac70a62233c3c07f02b502b4e294027e23cfcb95b51ce9c86e64ca9a1b8379faac70a62233c3c07f02b502b4e294027e"
        "23cfcb95b51ce9c86e64ca9a1b8379faac70a62233c3c07f02b502b4e294027e23cfcb95b51ce9c86e64ca9a1b8379fa"
        "ac70a62233c3c07f02b502b4e294027e23cfcb95b51ce9c86e64ca9a1b8379faac70a62233c3c07f02b502b4e294027e"
        "23cfcb95b51ce9c86e64ca9a1b8379faac70a62233c3c07f02b502b4e294027e23cfcb95b51ce9c86e64ca9a1b8379fa"
        "ac70a62233c3c07f02b502b4e294027e23cfcb95b51ce9c86e64ca9a1b8379faac70a62233c3c07f02b502b4e294027e"
        "23cfcb95b51ce9c86e64ca9a1b8379faac70a62233c3c07f02b502b4e294027e23cfcb95b51ce9c86e64ca9a1b8379fa"
        "ac70a62233c3c07f02b502b4e294027e23cfcb95b51ce9c86e64ca9a1b8379faac70a62233c3c07f02b502b4e294027e"
        "23cfcb95b51ce9c86e64ca9a1b8379faac70a62233c3c07f02b502b4e294027e23cfcb95b51ce9c86e64ca9a1b8379fa"
        "ac70a62233c3c07f02b502b4e294027e23cfcb95b51ce9c86e64ca9a1b8379faac70a62233c3c07f02b502b4e294027e"
        "23cfcb95b51ce9c86e64ca9a1b8379faac70a62233c3c07f02b502b4e294027e23cfcb95b51ce9c86e64ca9a1b8379fa"
        "ac70a62233c3c07f02b502b4e294027e"
    ),
    "mapping_ids": (
        "e36e60c0df3072142d8522982adfea680df4987102e8c6d046a39e69978810b3040b5ee7a7a789714e66120dbc51a815"
        "e2a11ff46262cf6a2825ed28c1df51246b1e5d8e329fb556c0efa18cd561e97370e8887123b1030f1778fdebb4fca293"
        "a7f9363eb513ae01aa4bf37ea3047ad5173da17d7826a113ed60fd4805ea0ba4c51bc5f116b8d8dbd8af8beee870973e"
        "71430404fe3df55770e59b3158fd0b8e296d50f79182afa75b34b9f3c7155a8ab3fb477f1f61674dd7d3159dc02bde80"
        "283a97f2abc42506e24744e3b42d3a402b0db712483c0e369165b560449396151e7f084d799bb532b5e285b7e6f8414f"
        "d52de1f756566806cb900dd5dea7bb62f942e01006acd148498ecacb3f52daa1f4fb9c4d9f36d5686e331d0b3b8c47a9"
        "122381ed04c9b45ece1d10bafb50e0f4d012da6cfedac0dcb3d846b89a797d56b83c08c69a4b2c888cf058f1486e038c"
        "16bc454d53942b3e585a6bc77d141e8f923536bb631dfd6d9d7ecbd9a974d4e592a9cededa3bf71049431250efe7ec28"
        "b4014be3d0b3d3c525869f77a22436abaa74bb473bfb3e40b4dd67020da8ac6df0534025042f44438830a6826e2a9dd6"
        "87d2b4f33a2c990f37a78a5752b6592d732a7d4715ce142ef2d94861a32e6d6e47f08a2dfb359073b386071040a5b44a"
        "00a72445ab6bca47d4715d44fb9e8ba6a88d513882bf63aebf6bba6f103ddd832c279035f70e3b7e3bdb6c94f2cc7b82"
        "eaed665c75f699b597a37f29263ab45022a0ffc7f9f921f778dcc96b69996520957c9a4ec23f06dea383c5f2b8114937"
        "461c2071375d62367bf83b8c62a1535808f07d8d7d25ae08392648160325e5e8c5e1a3c50648b7567e0bd61055d35421"
        "126e95782137ec2ba5b2c371c185e57ebc63468c5811cd1e540d7dbd7d3f13649abb313ca0bfb86162155fd3418bd4d2"
        "0f56d89f066de5db574aa03add1333626ad4656fa9564f4f12126152801bf63bc5f078f9bef5663bd41e5842ba7259e3"
        "5cb60d3337f23666e30348c7784f710781d83ad825474709f38194bb2aa4180057a1bcaf3bb54b76bc409366b7c92d42"
        "3dcc39f1ddcfbad7fb4f2a63a36d05d6c2283343082994b9bf2cbf4199619caccfaba91439f9257f91bb9ecc685ffa26"
        "f94b4db7cacce7f477f9c5c0d3c95a6d0b4cb635f00c8ce33b611bce9c22672a26265048907b11a935679e9d92ce0aea"
        "fcb2f1b33efe2b641a9830816ce14d3dbc2a32185cb0ea3b4ac6b3572a52b224341e1117e41f80ba703fcd796d0d3252"
        "19ae6c615291298acc8b9410b0dd0bf52e88e1ef959c891f2eb553f99859de8928628d1ad62fdb6d4f09fb9381f92207"
        "ec392d670ee6ddf32ef1f65e2aab658581e3eaefc2de942c8ab5b46622e663ff5943c6d74f762949faf7bdd959309cb1"
        "e0f7da138fc3551cb4ca1b4b93d45cba"
    ),
    "efforts": (
        "1a8466c3f574df1fd0da49707ae7c96000f394426ee51d85bcc943dd39d4203d1a8466c3f574df1fd0da49707ae7c960"
        "00f394426ee51d85bcc943dd39d4203d1a8466c3f574df1fd0da49707ae7c96000f394426ee51d85bcc943dd39d4203d"
        "1a8466c3f574df1fd0da49707ae7c96000f394426ee51d85bcc943dd39d4203d1a8466c3f574df1fd0da49707ae7c960"
        "00f394426ee51d85bcc943dd39d4203d1a8466c3f574df1fd0da49707ae7c96000f394426ee51d85bcc943dd39d4203d"
        "1a8466c3f574df1fd0da49707ae7c96000f394426ee51d85bcc943dd39d4203d1a8466c3f574df1fd0da49707ae7c960"
        "00f394426ee51d85bcc943dd39d4203d1a8466c3f574df1fd0da49707ae7c96000f394426ee51d85bcc943dd39d4203d"
        "1a8466c3f574df1fd0da49707ae7c96000f394426ee51d85bcc943dd39d4203d1a8466c3f574df1fd0da49707ae7c960"
        "00f394426ee51d85bcc943dd39d4203d1a8466c3f574df1fd0da49707ae7c96000f394426ee51d85bcc943dd39d4203d"
        "1a8466c3f574df1fd0da49707ae7c96000f394426ee51d85bcc943dd39d4203d1a8466c3f574df1fd0da49707ae7c960"
        "00f394426ee51d85bcc943dd39d4203d1a8466c3f574df1fd0da49707ae7c96000f394426ee51d85bcc943dd39d4203d"
        "1a8466c3f574df1fd0da49707ae7c96000f394426ee51d85bcc943dd39d4203d1a8466c3f574df1fd0da49707ae7c960"
        "00f394426ee51d85bcc943dd39d4203d1a8466c3f574df1fd0da49707ae7c96000f394426ee51d85bcc943dd39d4203d"
        "1a8466c3f574df1fd0da49707ae7c96000f394426ee51d85bcc943dd39d4203d1a8466c3f574df1fd0da49707ae7c960"
        "00f394426ee51d85bcc943dd39d4203d1a8466c3f574df1fd0da49707ae7c96000f394426ee51d85bcc943dd39d4203d"
        "1a8466c3f574df1fd0da49707ae7c96000f394426ee51d85bcc943dd39d4203d1a8466c3f574df1fd0da49707ae7c960"
        "00f394426ee51d85bcc943dd39d4203d1a8466c3f574df1fd0da49707ae7c96000f394426ee51d85bcc943dd39d4203d"
        "1a8466c3f574df1fd0da49707ae7c96000f394426ee51d85bcc943dd39d4203d1a8466c3f574df1fd0da49707ae7c960"
        "00f394426ee51d85bcc943dd39d4203d1a8466c3f574df1fd0da49707ae7c96000f394426ee51d85bcc943dd39d4203d"
        "1a8466c3f574df1fd0da49707ae7c96000f394426ee51d85bcc943dd39d4203d1a8466c3f574df1fd0da49707ae7c960"
        "00f394426ee51d85bcc943dd39d4203d1a8466c3f574df1fd0da49707ae7c96000f394426ee51d85bcc943dd39d4203d"
        "1a8466c3f574df1fd0da49707ae7c96000f394426ee51d85bcc943dd39d4203d1a8466c3f574df1fd0da49707ae7c960"
        "00f394426ee51d85bcc943dd39d4203d"
    ),
    "query_colors": (
        "3036ff91b3f8adc2c20e18782adf565421dcf654fd04e11e4accfb7b33fd355633f5a59c425fe8cb39233d7b3bcc9b9f"
        "2c2ac4a7698c4a6e185d32d15e743048ba7668988d41a35704e305907fb28372463201763a4c2bfa3c78a8653d3620de"
        "b02cc31ee1056f52e95dbb7294433912018b69eca52f2a511d8168f1c143ad64c2903843524eb48df939df606a69e3ba"
        "3639527b318afc655b50fb6d665ec7333036ff91b3f8adc2c20e18782adf565421dcf654fd04e11e4accfb7b33fd3556"
        "33f5a59c425fe8cb39233d7b3bcc9b9f2c2ac4a7698c4a6e185d32d15e743048ba7668988d41a35704e305907fb28372"
        "463201763a4c2bfa3c78a8653d3620deb02cc31ee1056f52e95dbb7294433912018b69eca52f2a511d8168f1c143ad64"
        "c2903843524eb48df939df606a69e3ba3639527b318afc655b50fb6d665ec7333036ff91b3f8adc2c20e18782adf5654"
        "21dcf654fd04e11e4accfb7b33fd355633f5a59c425fe8cb39233d7b3bcc9b9f2c2ac4a7698c4a6e185d32d15e743048"
        "ba7668988d41a35704e305907fb28372463201763a4c2bfa3c78a8653d3620deb02cc31ee1056f52e95dbb7294433912"
        "018b69eca52f2a511d8168f1c143ad64c2903843524eb48df939df606a69e3ba3639527b318afc655b50fb6d665ec733"
        "3036ff91b3f8adc2c20e18782adf565421dcf654fd04e11e4accfb7b33fd355633f5a59c425fe8cb39233d7b3bcc9b9f"
        "2c2ac4a7698c4a6e185d32d15e743048ba7668988d41a35704e305907fb28372463201763a4c2bfa3c78a8653d3620de"
        "b02cc31ee1056f52e95dbb7294433912018b69eca52f2a511d8168f1c143ad64c2903843524eb48df939df606a69e3ba"
        "3639527b318afc655b50fb6d665ec7333036ff91b3f8adc2c20e18782adf565421dcf654fd04e11e4accfb7b33fd3556"
        "33f5a59c425fe8cb39233d7b3bcc9b9f2c2ac4a7698c4a6e185d32d15e743048ba7668988d41a35704e305907fb28372"
        "463201763a4c2bfa3c78a8653d3620deb02cc31ee1056f52e95dbb7294433912018b69eca52f2a511d8168f1c143ad64"
        "c2903843524eb48df939df606a69e3ba3639527b318afc655b50fb6d665ec7333036ff91b3f8adc2c20e18782adf5654"
        "21dcf654fd04e11e4accfb7b33fd355633f5a59c425fe8cb39233d7b3bcc9b9f2c2ac4a7698c4a6e185d32d15e743048"
        "ba7668988d41a35704e305907fb28372463201763a4c2bfa3c78a8653d3620deb02cc31ee1056f52e95dbb7294433912"
        "018b69eca52f2a511d8168f1c143ad64c2903843524eb48df939df606a69e3ba3639527b318afc655b50fb6d665ec733"
        "3036ff91b3f8adc2c20e18782adf565421dcf654fd04e11e4accfb7b33fd355633f5a59c425fe8cb39233d7b3bcc9b9f"
        "2c2ac4a7698c4a6e185d32d15e743048"
    ),
    "presentation_orders": (
        "ceafaf9166bbe3a9b1d1311d54678644cd9c2b034eadaa5587ddda20a7a50c625b94d406b53a8b0661ba6af49d230863"
        "2db11553c07942325506bcbee057dc28194ec9e23083f2f74c1777e0a2cda3ed8372c461a86d94cb25c630e4df2b30f5"
        "d85de4beac74284d8707b7219167a0247cfd0a98c3045f0be4a21bd1eb290d38d72b81b7543208a5669486a9bb4b569e"
        "da2f3997a0e1b55163f93706395f6928aa48502f8cfb2f0868c30c83a3c5c70f146be89e7ce757ad58a64814cea6ea18"
        "4fd12c5b43f2b03577dd11b64eea49c7a388f4b7dae1a57395ee60239dd5c34e34153081783113b58faa65eb7949488e"
        "301f073133d9536def5b2b02305d3ca06be9b7f723b62e50c89274bfbc3f7c7c85eb81036947b5ca7d27a9b135064cea"
        "f852385f33ece97581b231bb5ad6f3e7ebf0faa92a551d64716b54cea2ee60e4c1ff1c59cf3c77857f8caaaa569b4624"
        "f8314aede772005b888d3da17e1d1bff8b0df93f94a99bda7bddf6b4eb09e11d888d9f3043b4f601d9bc7242e5d6fd14"
        "d7ad665e94e153b926f860d22e7b5a155b5790ddf1ccebaf91b5651fe9bd666a1acfe38954c726afc7308d18902adddf"
        "eef54582f51fed3095fdaadd8d6b58e15363c86b2a04e9e5c692f66743bb1d51b06a04d510df01e92e5f2f208eca92d3"
        "98e9c7dbd07e7d3597429e13b4660dc192fdd594190a08df8726b62f104fda12a20ef52470390e22900c7ddb48041b06"
        "c7e899188bfa34cedffa6fb3b254f192b89d2b87261250b5fec029db251e17e4ccfc1af47415c385bbcf52c70891900b"
        "3f359711b3f635e0d55ff339a257bd3ea3cb2f2df8a643795756583d34ebe3af24adeb54309e6a384b7384f21c05cfaf"
        "f19bdc91ae30640e131e84a35b06f91d8de0091becad29ebd635e297b0d2d8f668c6dba91bfca2871a64c576dbb69e2f"
        "0cf82e0bde99d9db4ccb2b7cc7cb53188b17a7cb28aa40c49005d9f5a2117bdc8b9acbc2d0dc9ecd82334597c2e12e08"
        "da34a427954da16d0bd6ac6077e86748b6307ea126312201055d1284ebc43994f0a5063ed526c387a06bcdd0514323d5"
        "5b8e40b0e0fbb1b337ceb0007846e74922e0ea03652437d7c11619496d699e7d1a68f565ff104acdfbb536f9c051b486"
        "209e636d7a154ed9fed231adc739dea6841ad9da9622062f3f84d2ab35f2123c6113ee02e1c0bb0eb6900f385d100858"
        "c81d4b7c524265383ec45f3d436ce573db4b34655d65355cf478232b0bdc3e83e23012343bf2354469454b0e6c7d1f91"
        "c36d5495e065f5f68a8c3c5ca6ac61658e2003c322db9be8694f22289a505af5f1b299d1ffae762c87bc0ee8f89220a8"
        "e505246f1c584c1338ba7de2e70a19150fd0edd219a88c896c1a500bda4ddf6016bb68e3cd203bec332830631184358b"
        "537fcea1490a72139601ff34da01c40a"
    ),
}
_PRODUCTION_VALIDATION_SHA256 = {
    "mapping_ids": "b036444e228c60116b8bfd9c10399261bcf6645d7b69d27d4b391460fae83cd8",
    "query_colors": "c7e70f56cca66d920d5d690a902b9943f2fcfdff7003fa4bbb3580070738d67e",
    "presentation_orders": "0bab5cfe3c2c87109909d36c59f88ef04983322aacbce469fea6221aa4ac37b0",
    "shuffled_shifts": "15af1f04589cc523d89b66d2f07027158d69068901d786eecfd259a156f2f2d0",
    "intact": "5683aa84aa2ef8a1ff623e5e0b60afb3451e617728f0363d3ad84f2ea52dacde",
    "shuffled": "abd5eb4ab2e2a685faeb8f6bf785ad2deb97721b00e8d194b4f65d4995516be3",
    "no_context": "45fd14d3faefad83b0ce6d908456320afa67944b361159cfe503fdfab591162d",
    "targets_by_depth": "a438d64347dc4ec5cfc639342d8b142c785e497ddf06728eb03f8ccfb42d3cd6",
    "advance_masks": "b88b3593d9df51260fbafa4a937159c3da3f56fc33335a30993c0ff8a7462ac8",
}
_VALIDATION_IDENTITY_FIELDS = (
    "mapping_ids",
    "query_colors",
    "presentation_orders",
    "shuffled_shifts",
    "intact",
    "shuffled",
    "no_context",
    "targets_by_depth",
    "advance_masks",
)


def _iterate(mapping: np.ndarray, color: int, applications: int) -> int:
    value = color
    for _ in range(applications):
        value = int(mapping[value])
    return value


def _is_single_ten_cycle(mapping: np.ndarray) -> bool:
    visited: list[int] = []
    value = 0
    for _ in range(10):
        visited.append(value)
        value = int(mapping[value])
    return value == 0 and sorted(visited) == list(range(10))


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _ordered_json_list_digest(values: list[str]) -> str:
    payload = msgspec_json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _strict_json_digest(value: Mapping[str, Any]) -> str:
    payload = (
        msgspec_json.dumps(
            value,
            allow_nan=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _validation_data_digests(data: depth.DepthValidationData) -> dict[str, str]:
    return {
        field: _array_digest(np.asarray(getattr(data, field)))
        for field in _VALIDATION_IDENTITY_FIELDS
    }


def _one_cell(color: int) -> ArcGrid:
    return ArcGrid(((int(color),),))


def _encoded_cycle_episode(
    mapping: np.ndarray,
    query_color: int,
    presentation_order: np.ndarray,
    row_config: RowEventConfig,
) -> np.ndarray:
    demonstrations = tuple(
        ArcPair(_one_cell(color), _one_cell(int(mapping[color])))
        for color in np.asarray(presentation_order, dtype=np.int32)
    )
    task = ArcTask(
        train=demonstrations,
        test=(
            ArcPair(
                _one_cell(query_color),
                _one_cell(int(mapping[query_color])),
            ),
        ),
    )
    return np.asarray(encode_query_episode(task, 0, row_config).events)


def _reduced_depth_config() -> depth.DepthGateConfig:
    return depth.DepthGateConfig(
        training_updates=1,
        batch_size=2,
        validation_episodes=2,
        neuron_count=64,
        recurrent_edges=64,
        readout_width=8,
        color_rank=1,
        staging_chunk_updates=1,
    )


class _DepthObjectiveProbe(brainstate.nn.Module):
    def __init__(self, config: depth.DepthGateConfig) -> None:
        super().__init__()
        self.reservoir = LatentWorkspaceModel(depth._model_config(config, batch_size=1))
        self.event_width = config.row_config.input_width
        self.color_rank = config.color_rank

    def update(self, packed: jax.Array) -> jax.Array:
        event = packed[:, : self.event_width]
        advance = packed[:, self.event_width] > 0.5
        target = packed[:, self.event_width + 1].astype(jnp.int32)
        loss_scale = packed[:, self.event_width + 2]
        compact = self.reservoir(event, advance)
        loss = legacy._classification_loss(compact, target, self.color_rank)
        return loss_scale * jnp.sqrt(loss)


def _depth_probe_pp_prop(model: brainstate.nn.Module):
    return braintrace.pp_prop(
        model,
        decay_or_rank=0.9,
        vjp_method="multi-step",
        config=braintrace.ETraceConfig(
            trace_factorization="io_factorized",
            recurrence_scope="diagonal",
            decay=0.9,
        ),
    )


def _depth_objective_probe_inputs(
    effort: int,
) -> tuple[depth.DepthGateConfig, jax.Array, jax.Array]:
    config = depth.DepthGateConfig(
        training_updates=1,
        batch_size=1,
        validation_episodes=1,
        neuron_count=64,
        recurrent_edges=1,
        readout_width=2,
        color_rank=1,
        context_memory_width=2,
        staging_chunk_updates=1,
    )
    mapping = np.asarray(depth.unrank_ten_cycle(12_345))
    query = 7
    events = np.zeros((19, 1, config.row_config.input_width), dtype=np.float32)
    events[:11, 0] = _encoded_cycle_episode(
        mapping,
        query,
        np.arange(10, dtype=np.int32),
        config.row_config,
    )
    contract = depth._checkpoint_contract(mapping, query, effort)
    packed = np.concatenate(
        (
            events,
            np.asarray(contract.advance_mask, dtype=np.float32)[:, None, None],
            np.asarray(contract.targets, dtype=np.float32)[:, None, None],
            np.sqrt(np.asarray(contract.loss_weights, dtype=np.float32))[:, None, None],
        ),
        axis=-1,
    )
    padded = jnp.asarray(packed)
    return config, padded, padded[: contract.active_length]


def _probe_objective(
    config: depth.DepthGateConfig,
    inputs: jax.Array,
) -> float:
    model = _DepthObjectiveProbe(config)
    brainstate.nn.init_all_states(model, batch_size=1)
    outputs = brainstate.transform.for_loop(model, inputs)
    return float(jnp.square(outputs).sum())


def _passing_depth_compiler() -> dict[str, Any]:
    paths = [
        "memory_write_scale",
        "workspace_query_projection/weight",
        "memory_read_projection/weight",
    ]
    return {
        "available": True,
        "diagnostics": [],
        "compiled_parameter_paths": paths,
        "required_direct_paths": paths,
        "direct_path_status": {path: True for path in paths},
        "direct_path_evidence": {
            "memory_write_scale": [
                {
                    "relation_key": "weight",
                    "classification": "all_direct",
                    "hidden_groups": [{"index": 1, "hidden_paths": ["context_memory"]}],
                }
            ],
            "workspace_query_projection/weight": [
                {
                    "relation_key": "weight",
                    "classification": "all_direct",
                    "hidden_groups": [
                        {"index": 3, "hidden_paths": ["reasoning_query"]}
                    ],
                }
            ],
            "memory_read_projection/weight": [
                {
                    "relation_key": "weight",
                    "classification": "all_direct",
                    "hidden_groups": [
                        {
                            "index": 0,
                            "hidden_paths": [
                                "ff_syn/post/V",
                                "workspace_carrier",
                            ],
                        }
                    ],
                }
            ],
        },
        "hidden_groups": [
            {"index": 0, "hidden_paths": ["ff_syn/post/V", "workspace_carrier"]},
            {"index": 1, "hidden_paths": ["context_memory"]},
            {"index": 2, "hidden_paths": ["query_encoding"]},
            {"index": 3, "hidden_paths": ["reasoning_query"]},
        ],
        "all_required_direct": True,
        "context_memory_isolated_from_workspace_lif": True,
    }


def _depth_metric(
    correct: int,
    *,
    checkpoint: int,
    prediction_label: str,
) -> dict[str, Any]:
    count = 512
    lower, upper = legacy._wilson_interval(correct, count)
    return {
        "correct": correct,
        "count": count,
        "accuracy": correct / count,
        "wilson_95_lower": lower,
        "wilson_95_upper": upper,
        "prediction_histogram": [52, 52, 51, 51, 51, 51, 51, 51, 51, 51],
        "prediction_sha256": hashlib.sha256(prediction_label.encode()).hexdigest(),
        "checkpoint": checkpoint,
    }


def _passing_depth_evaluation() -> dict[str, Any]:
    depths: dict[str, dict[str, Any]] = {}
    for depth_index in range(9):
        depths[str(depth_index)] = {
            "intact": _depth_metric(
                400,
                checkpoint=depth_index,
                prediction_label=f"intact-{depth_index}",
            ),
            "shuffled": _depth_metric(
                64,
                checkpoint=depth_index,
                prediction_label=f"shuffled-{depth_index}",
            ),
            "no_context": _depth_metric(
                64,
                checkpoint=depth_index,
                prediction_label=f"no-context-{depth_index}",
            ),
        }
    h0_proper = depths["0"]["intact"]
    efforts: dict[str, dict[str, Any]] = {}
    for effort in depth.QUALIFYING_EFFORTS:
        matching = depths[str(effort)]
        h0_final = _depth_metric(
            100,
            checkpoint=0,
            prediction_label="intact-0",
        )
        efforts[str(effort)] = {
            "intact": matching["intact"],
            "shuffled": matching["shuffled"],
            "no_context": matching["no_context"],
            "h0_final_target": h0_final,
            "intact_minus_h0": (matching["intact"]["accuracy"] - h0_final["accuracy"]),
            "intact_minus_shuffled": (
                matching["intact"]["accuracy"] - matching["shuffled"]["accuracy"]
            ),
        }
    return {
        "finite": True,
        "h0_proper": h0_proper,
        "depths": depths,
        "efforts": efforts,
    }


def _placeholder_schedule_report() -> dict[str, Any]:
    return {
        "chunk_count": 32,
        "chunk_updates": 128,
        "training_updates": 4_096,
        "training_episode_count": 262_144,
        "global_sha256": dict(_PRODUCTION_ENCODED_GLOBAL_SHA256),
        "chunk_sha256_manifest": dict(_PRODUCTION_CHUNK_SHA256_MANIFEST),
        "chunk_sha256": {
            field: [packed[index : index + 64] for index in range(0, len(packed), 64)]
            for field, packed in _PRODUCTION_CHUNK_SHA256_HEX.items()
        },
    }


def _passing_source_report() -> dict[str, Any]:
    commit = "a" * 40
    return {
        "commit": commit,
        "asserted_commit": commit,
        "asserted_commit_matches_head": True,
        "commit_is_valid_40_hex": True,
        "head_command_succeeded": True,
        "verified": True,
        "dirty": False,
        "asserted_dirty": False,
        "asserted_dirty_matches_worktree": True,
        "status_command_succeeded": True,
    }


def _gate_b_initialization_wrapper(
    admission: Mapping[str, Any],
) -> dict[str, Any]:
    source_head = str(admission["source_start"]["commit"])
    image_digest = str(admission["environment"]["image_digest"])
    manifest_sha256 = "3" * 64
    preflight_sha256 = "4" * 64
    result_sha256 = _strict_json_digest(admission)
    bundle_sha256 = hashlib.sha256(
        (
            "example21-launch-bundle-v1\0gate_b_init\0"
            f"{source_head}\0{preflight_sha256}\0{result_sha256}"
        ).encode()
    ).hexdigest()
    return {
        "target": "gate_b_init",
        "source_head": source_head,
        "image_digest": image_digest,
        "bundle_sha256": bundle_sha256,
        "manifest_sha256": manifest_sha256,
        "preflight_sha256": preflight_sha256,
        "result_sha256": result_sha256,
        "admission": copy.deepcopy(admission),
    }


def _refresh_gate_b_initialization_wrapper_hashes(
    wrapper: dict[str, Any],
) -> None:
    result_sha256 = _strict_json_digest(wrapper["admission"])
    wrapper["result_sha256"] = result_sha256
    wrapper["bundle_sha256"] = hashlib.sha256(
        (
            "example21-launch-bundle-v1\0gate_b_init\0"
            f"{wrapper['source_head']}\0{wrapper['preflight_sha256']}\0"
            f"{result_sha256}"
        ).encode()
    ).hexdigest()


def _passing_depth_report() -> dict[str, Any]:
    config = depth.DepthGateConfig()
    initialization_sha = "1" * 64
    telemetry_categories = (
        "logits",
        "model_states",
        "gradients",
        "pp_prop_traces",
        "adam",
        "parameters",
    )
    report = {
        "schema_version": 1,
        "control": "example21_demonstrated_depth_gate_b",
        "qualification_regime": "preregistered_full",
        "learner": {
            "algorithm": "production_pp_prop",
            "optimizer": "Adam",
            "trace_factorization": "io_factorized",
            "recurrence_scope": "diagonal",
            "trace_decay": 0.9,
            "vjp_method": "multi-step",
        },
        "config": dataclasses.asdict(config),
        "prerequisites": {
            "gate_a": {
                "qualification_passed": True,
                "result_sha256": config.gate_a_result_sha256,
                "manifest_sha256": config.gate_a_manifest_sha256,
                "source_commit": config.gate_a_source_commit,
            },
            "gate_b_initialization": {
                "fresh_model": True,
                "model_seed": config.model_seed,
                "configuration": dataclasses.asdict(config),
                "parameter_sha256": initialization_sha,
                "parameter_count": 1_000_000,
                "compiler": _passing_depth_compiler(),
                "compile_warnings": [],
            },
        },
        "data": {
            "schedule": _placeholder_schedule_report(),
            "target_contract": {
                "efforts": [1, 2, 4, 8],
                "sequence_length": 19,
                "h0_index": 10,
                "active_lengths": {"1": 12, "2": 13, "4": 15, "8": 19},
                "target_rule": "H_r=f^(r+1)(x)",
                "supervision_weights": {
                    "1": 0.5,
                    "2": 1.0 / 3.0,
                    "4": 0.2,
                    "8": 1.0 / 9.0,
                },
                "suffix_advance_false": True,
                "suffix_loss_weight_zero": True,
                "padded_compact_objective_equal": True,
                "padded_compact_pp_prop_gradients_equal": True,
                "finite_window_chunk_size": 1,
            },
            "controls": {
                "rotation_candidates": list(range(1, 10)),
                "all_validation_rotations_valid": True,
                "all_shuffled_answers_differ_at_qualifying_depths": True,
                "exact_input_output_marginals": True,
                "no_context_demonstrations_zero": True,
                "event_timing_identical": True,
            },
            "held_out_invariants": {
                "validation_episode_count": 512,
                "distinct_training_cycle_count": 262_144,
                "distinct_validation_cycle_count": 512,
                "training_validation_overlap_count": 0,
                "balanced_queries": True,
                "target_trajectories_exact": True,
                "no_copy_shortcut_at_every_effort": True,
                "cross_effort_h0_prediction_identity": True,
                "cross_effort_h0_identity_count": 512,
            },
        },
        "training": {
            "algorithm": "production_pp_prop",
            "executed_updates": 4_096,
            "batch_size": 64,
            "chunk_count": 32,
            "chunk_updates": 128,
            "effort_update_counts": {"1": 1_024, "2": 1_024, "4": 1_024, "8": 1_024},
            "initialization_parameter_sha256": initialization_sha,
            "final_parameter_sha256": "2" * 64,
            "losses": [1.0] * 4_096,
            "finite": {category: True for category in telemetry_categories},
            "max_abs": {category: 1.0 for category in telemetry_categories},
            "value_count": {category: 4_096 for category in telemetry_categories},
            "compiler": _passing_depth_compiler(),
            "compile_warnings": [],
        },
        "evaluation": {
            **_passing_depth_evaluation(),
            "initialization_parameter_sha256": initialization_sha,
        },
        "source_start": _passing_source_report(),
        "source_end": _passing_source_report(),
        "source_files": {
            "latent_workspace_model.py": config.model_source_sha256,
            "latent_workspace_task.py": config.task_source_sha256,
        },
        "environment": {
            "backend": "gpu",
            "image_digest": "sha256:" + "b" * 64,
            "devices": [{"id": 0, "platform": "gpu", "device_kind": "test GPU"}],
        },
        "qualification": {
            "passed": True,
            "criteria": {name: True for name in depth._QUALIFICATION_CRITERIA},
            "interpretation": "fabricated_do_not_trust",
        },
    }
    initialization = report["prerequisites"]["gate_b_initialization"]
    initialization["parameters_finite"] = True
    initialization_criteria = {
        "schema_and_control": True,
        "preregistered_configuration": True,
        "gate_a_prerequisite_authenticated": True,
        "source_and_gpu_authenticated": True,
        "initialization_fresh_and_finite": True,
        "compiler_paths_complete": True,
    }
    initialization_admission = {
        "schema_version": 1,
        "control": "example21_gate_b_initialization_admission",
        "qualification_regime": "preregistered_full",
        "config": dataclasses.asdict(config),
        "prerequisites": {"gate_a": report["prerequisites"]["gate_a"]},
        "initialization": initialization,
        "source_start": report["source_start"],
        "source_end": report["source_end"],
        "source_files": report["source_files"],
        "environment": report["environment"],
        "qualification": {
            "passed": True,
            "criteria": initialization_criteria,
            "interpretation": "gate_b_initialization_admission_passed",
        },
    }
    report["prerequisites"]["gate_b_initialization"] = (
        _gate_b_initialization_wrapper(initialization_admission)
    )
    report["data"]["validation"] = {
        "episode_count": 512,
        "sha256": dict(_PRODUCTION_VALIDATION_SHA256),
    }
    return report


def test_depth_gate_config_is_exact_preregistered_contract() -> None:
    config = depth.DepthGateConfig()

    assert depth.QUALIFYING_EFFORTS == (1, 2, 4, 8)
    assert depth.TEN_CYCLE_CATALOG_SIZE == math.factorial(9) == 362_880
    assert depth.STAGING_CHUNK_UPDATES == 128
    assert depth.STAGING_CHUNK_COUNT == 32
    assert depth.GATE_B_SCHEMA_VERSION == 1
    assert depth.GATE_B_CONTROL == "example21_demonstrated_depth_gate_b"
    assert config.training_updates == 4_096
    assert config.batch_size == 64
    assert config.validation_episodes == 512
    assert config.gap_steps == 8
    assert config.neuron_count == 2_048
    assert config.recurrent_edges == 16_384
    assert config.readout_width == 128
    assert config.color_rank == 16
    assert config.context_memory_width == 32
    assert config.memory_decay == 1.0
    assert config.trace_decay == 0.9
    assert config.learning_rate == 0.003
    assert config.clip_norm == 1.0
    assert config.input_gain == 4.0
    assert config.recurrent_gain == 0.8
    assert config.model_seed == 2_108
    assert config.catalog_seed == 20_260_818
    assert config.train_episode_seed == 32_021
    assert config.validation_episode_seed == 92_021
    assert config.staging_chunk_updates == 128
    assert config.row_config.max_demonstrations == 10
    assert config.row_config.input_width == 47
    assert config.sequence_length == 19
    assert config.training_episode_count == 262_144
    assert config.staging_chunk_count == 32
    assert config.qualification_regime == "preregistered_full"
    assert config.gate_a_result_sha256 == (
        "3a585e739715b31757082b50fe57b98ca50107891f7c79edaa7e5e54c90ad632"
    )
    assert config.gate_a_manifest_sha256 == (
        "69d690daa5023f5b3ce22b0e65ea09a1a6706687d792e998651422f6d6ea15cf"
    )
    assert config.gate_a_source_commit == "4737e9172b1c6ca99347af5b2c83fc795a294a16"
    assert config.model_source_sha256 == (
        "467022c79123b976dd5cebc8d5ae5da37d1373bc46477133003b0b263abd8216"
    )
    assert config.task_source_sha256 == (
        "cfaec054bd42f6dccf9fb24c5fbec0cd703fdef17ba8d3b6dd68bf78366de18b"
    )
    assert dataclasses.replace(config, training_updates=4).qualification_regime == (
        "nonqualifying_abbreviated"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("training_updates", 4_095),
        ("batch_size", 32),
        ("validation_episodes", 256),
        ("neuron_count", 1_024),
        ("recurrent_edges", 8_192),
        ("readout_width", 64),
        ("color_rank", 8),
        ("context_memory_width", 16),
        ("memory_decay", 0.95),
        ("trace_decay", 0.8),
        ("learning_rate", 0.001),
        ("clip_norm", 0.5),
        ("input_gain", 3.0),
        ("recurrent_gain", 0.7),
        ("model_seed", 2_109),
        ("catalog_seed", 20_260_819),
        ("train_episode_seed", 32_022),
        ("validation_episode_seed", 92_022),
        ("staging_chunk_updates", 64),
    ],
)
def test_any_mutable_frozen_config_coordinate_change_is_nonqualifying(
    field: str,
    value: object,
) -> None:
    changed = dataclasses.replace(depth.DepthGateConfig(), **{field: value})

    assert changed.qualification_regime == "nonqualifying_abbreviated"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gap_steps", 4),
        (
            "row_config",
            RowEventConfig(max_demonstrations=9, max_grid_size=1),
        ),
        (
            "row_config",
            RowEventConfig(max_demonstrations=10, max_grid_size=2),
        ),
    ],
)
def test_incompatible_frozen_layout_change_fails_closed(
    field: str,
    value: object,
) -> None:
    try:
        changed = dataclasses.replace(depth.DepthGateConfig(), **{field: value})
    except ValueError:
        return

    assert changed.qualification_regime == "nonqualifying_abbreviated"


@pytest.mark.parametrize(
    "changes",
    [
        {"batch_size": True},
        {"memory_decay": "invalid"},
        {"trace_decay": math.inf},
        {"training_updates": 0},
        {"validation_episodes": 0},
        {"staging_chunk_updates": 0},
        {"staging_chunk_updates": 127},
        {"training_updates": 6_000},
    ],
)
def test_depth_gate_config_rejects_invalid_numeric_and_schedule_contracts(
    changes: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(depth.DepthGateConfig(), **changes)


@pytest.mark.parametrize("mapping_id", [0, 1, 9, 10, 12_345, math.factorial(9) - 1])
def test_ten_cycle_catalog_unranking_is_bijective(mapping_id: int) -> None:
    mapping = np.asarray(depth.unrank_ten_cycle(mapping_id))

    assert mapping.shape == (10,)
    assert np.issubdtype(mapping.dtype, np.integer)
    assert sorted(mapping.tolist()) == list(range(10))
    assert _is_single_ten_cycle(mapping)
    assert depth.rank_ten_cycle(mapping) == mapping_id


@pytest.mark.parametrize(
    ("mapping_id", "expected"),
    [
        (0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
        (1, [1, 2, 3, 4, 5, 6, 7, 9, 0, 8]),
        (math.factorial(9) - 1, [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]),
    ],
)
def test_ten_cycle_unranking_uses_anchored_lexicographic_lehmer_order(
    mapping_id: int, expected: list[int]
) -> None:
    assert np.asarray(depth.unrank_ten_cycle(mapping_id)).tolist() == expected


@pytest.mark.parametrize("mapping_id", [-1, math.factorial(9)])
def test_ten_cycle_unranking_rejects_out_of_catalog_ids(mapping_id: int) -> None:
    with pytest.raises(ValueError, match="catalog|mapping"):
        depth.unrank_ten_cycle(mapping_id)


@pytest.mark.parametrize("mapping_id", [True, 1.5])
def test_ten_cycle_unranking_rejects_noninteger_ids(mapping_id: Any) -> None:
    with pytest.raises(ValueError, match="integer"):
        depth.unrank_ten_cycle(mapping_id)


@pytest.mark.parametrize(
    "mapping",
    [
        np.arange(9, dtype=np.int32),
        np.arange(10, dtype=np.float32),
        np.zeros((10,), dtype=np.int32),
        np.asarray([1, 0, 3, 2, 5, 4, 7, 6, 9, 8], dtype=np.int32),
    ],
)
def test_ten_cycle_ranking_rejects_malformed_or_disconnected_mappings(
    mapping: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="mapping|permutation|cycle"):
        depth.rank_ten_cycle(mapping)


@pytest.mark.parametrize(
    ("mapping", "query"),
    [
        (np.arange(9, dtype=np.int32), 0),
        (depth.unrank_ten_cycle(0), True),
        (depth.unrank_ten_cycle(0), 10),
    ],
)
def test_shuffled_rotation_rejects_malformed_inputs(
    mapping: np.ndarray,
    query: Any,
) -> None:
    with pytest.raises(ValueError, match="mapping|query"):
        depth._select_shuffled_rotation(mapping, query)


def test_shuffled_rotation_fails_closed_when_no_control_breaks_all_depths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(depth, "_iterate_mapping", lambda *args: 0)

    with pytest.raises(ValueError, match="no shuffled rotation"):
        depth._select_shuffled_rotation(depth.unrank_ten_cycle(0), 0)


@pytest.fixture(scope="module")
def production_schedule() -> depth.DepthSchedule:
    return depth._build_schedule(depth.DepthGateConfig())


def test_frozen_schedule_is_balanced_unique_and_disjoint(
    production_schedule: depth.DepthSchedule,
) -> None:
    schedule = production_schedule
    training_ids = np.asarray(schedule.training_mapping_ids)
    validation_ids = np.asarray(schedule.validation_mapping_ids)
    efforts = np.asarray(schedule.training_efforts)
    training_queries = np.asarray(schedule.training_query_colors)
    validation_queries = np.asarray(schedule.validation_query_colors)
    training_orders = np.asarray(schedule.training_presentation_orders)
    validation_orders = np.asarray(schedule.validation_presentation_orders)

    assert training_ids.shape == (4_096, 64)
    assert validation_ids.shape == (512,)
    assert efforts.shape == (4_096,)
    assert training_queries.shape == (4_096, 64)
    assert validation_queries.shape == (512,)
    assert training_orders.shape == (4_096, 64, 10)
    assert validation_orders.shape == (512, 10)
    assert np.array_equal(efforts, np.resize([1, 2, 4, 8], 4_096))
    assert np.array_equal(
        np.bincount(efforts, minlength=9)[[1, 2, 4, 8]],
        np.full((4,), 1_024),
    )
    flat_training_ids = training_ids.reshape(-1)
    assert np.unique(flat_training_ids).size == 262_144
    assert np.unique(validation_ids).size == 512
    assert not np.intersect1d(flat_training_ids, validation_ids).size
    assert flat_training_ids.min() >= 0
    assert validation_ids.min() >= 0
    assert flat_training_ids.max() < math.factorial(9)
    assert validation_ids.max() < math.factorial(9)
    assert flat_training_ids[:8].tolist() == [
        42_599,
        59_110,
        75_621,
        92_132,
        108_643,
        125_154,
        141_665,
        158_176,
    ]
    assert validation_ids[:4].tolist() == [232_423, 248_934, 265_445, 281_956]
    assert _array_digest(flat_training_ids) == (
        "b604a27206a0f64d222cb06530586622522b4f0951579f3aa0132a52e541381d"
    )
    assert _array_digest(validation_ids) == (
        "b036444e228c60116b8bfd9c10399261bcf6645d7b69d27d4b391460fae83cd8"
    )
    assert np.array_equal(training_queries.reshape(-1), np.arange(262_144) % 10)
    assert np.array_equal(validation_queries, (262_144 + np.arange(512)) % 10)
    for queries in (training_queries.reshape(-1), validation_queries):
        counts = np.bincount(queries, minlength=10)
        assert int(counts.max() - counts.min()) <= 1
    assert training_orders.reshape(-1, 10)[:3].tolist() == [
        [3, 5, 0, 1, 4, 6, 2, 8, 7, 9],
        [3, 9, 2, 8, 0, 7, 4, 1, 6, 5],
        [6, 4, 7, 1, 0, 3, 5, 8, 2, 9],
    ]
    assert validation_orders[:3].tolist() == [
        [6, 2, 5, 3, 8, 7, 4, 9, 0, 1],
        [3, 4, 2, 7, 5, 9, 0, 1, 6, 8],
        [3, 7, 8, 9, 4, 0, 5, 2, 6, 1],
    ]
    assert _array_digest(training_orders.reshape(-1, 10)) == (
        "79f8c7385699f29595e6fd99ff2e2e0feb56e6f69469ae3b16b4d7f4b8ae588d"
    )
    assert _array_digest(validation_orders) == (
        "0bab5cfe3c2c87109909d36c59f88ef04983322aacbce469fea6221aa4ac37b0"
    )


def test_frozen_schedule_is_sensitive_to_each_declared_seed(
    production_schedule: depth.DepthSchedule,
) -> None:
    config = depth.DepthGateConfig()
    catalog_changed = depth._build_schedule(
        dataclasses.replace(config, catalog_seed=config.catalog_seed + 1)
    )
    train_changed = depth._build_schedule(
        dataclasses.replace(config, train_episode_seed=config.train_episode_seed + 1)
    )
    validation_changed = depth._build_schedule(
        dataclasses.replace(
            config,
            validation_episode_seed=config.validation_episode_seed + 1,
        )
    )

    assert not np.array_equal(
        catalog_changed.training_mapping_ids,
        production_schedule.training_mapping_ids,
    )
    assert not np.array_equal(
        train_changed.training_presentation_orders,
        production_schedule.training_presentation_orders,
    )
    assert not np.array_equal(
        validation_changed.validation_presentation_orders,
        production_schedule.validation_presentation_orders,
    )
    assert np.array_equal(
        train_changed.training_mapping_ids,
        production_schedule.training_mapping_ids,
    )
    assert np.array_equal(
        validation_changed.validation_mapping_ids,
        production_schedule.validation_mapping_ids,
    )


def test_schedule_is_deterministic_and_chunking_is_identity(
    production_schedule: depth.DepthSchedule,
) -> None:
    config = depth.DepthGateConfig()
    repeated = depth._build_schedule(config)
    for field in dataclasses.fields(depth.DepthSchedule):
        assert np.array_equal(
            np.asarray(getattr(production_schedule, field.name)),
            np.asarray(getattr(repeated, field.name)),
        )

    chunks = tuple(depth._iter_schedule_chunks(production_schedule, config))
    assert len(chunks) == 32
    assert all(
        np.asarray(chunk.training_mapping_ids).shape == (128, 64) for chunk in chunks
    )
    assert all(np.asarray(chunk.training_efforts).shape == (128,) for chunk in chunks)
    assert all(
        np.asarray(chunk.training_presentation_orders).shape == (128, 64, 10)
        for chunk in chunks
    )
    assert np.array_equal(
        np.concatenate([np.asarray(chunk.training_mapping_ids) for chunk in chunks]),
        np.asarray(production_schedule.training_mapping_ids),
    )
    assert np.array_equal(
        np.concatenate([np.asarray(chunk.training_efforts) for chunk in chunks]),
        np.asarray(production_schedule.training_efforts),
    )
    assert np.array_equal(
        np.concatenate([np.asarray(chunk.training_query_colors) for chunk in chunks]),
        np.asarray(production_schedule.training_query_colors),
    )
    assert np.array_equal(
        np.concatenate(
            [np.asarray(chunk.training_presentation_orders) for chunk in chunks]
        ),
        np.asarray(production_schedule.training_presentation_orders),
    )


def test_encoded_schedule_report_matches_conceptual_chunk_concatenation() -> None:
    config = dataclasses.replace(
        _reduced_depth_config(),
        training_updates=4,
        staging_chunk_updates=2,
    )
    schedule = depth._build_schedule(config)
    encoded_chunks = tuple(
        depth._encode_training_chunk(chunk, config)
        for chunk in depth._iter_schedule_chunks(schedule, config)
    )
    fields = (
        "events",
        "targets",
        "loss_weights",
        "advance_masks",
        "mapping_ids",
        "efforts",
        "query_colors",
        "presentation_orders",
    )
    expected_globals = {
        field: _array_digest(
            np.concatenate(
                [np.asarray(getattr(chunk, field)) for chunk in encoded_chunks],
                axis=0,
            )
        )
        for field in fields
    }
    expected_chunks = {
        field: [
            _array_digest(np.asarray(getattr(chunk, field))) for chunk in encoded_chunks
        ]
        for field in fields
    }
    expected_chunk_manifests = {
        field: _ordered_json_list_digest(digests)
        for field, digests in expected_chunks.items()
    }

    report = depth._encoded_schedule_report(schedule, config)

    assert report == {
        "chunk_count": 2,
        "chunk_updates": 2,
        "training_updates": 4,
        "training_episode_count": 8,
        "global_sha256": expected_globals,
        "chunk_sha256_manifest": expected_chunk_manifests,
        "chunk_sha256": expected_chunks,
    }


def test_production_encoded_schedule_global_digests_are_pinned() -> None:
    config = depth.DepthGateConfig()

    report = depth._encoded_schedule_report(depth._build_schedule(config), config)

    assert report["chunk_count"] == 32
    assert report["chunk_updates"] == 128
    assert report["training_updates"] == 4_096
    assert report["training_episode_count"] == 262_144
    assert report["global_sha256"] == _PRODUCTION_ENCODED_GLOBAL_SHA256
    assert (
        report["chunk_sha256_manifest"]
        == _PRODUCTION_CHUNK_SHA256_MANIFEST
    )
    assert set(report["chunk_sha256"]) == set(_PRODUCTION_ENCODED_GLOBAL_SHA256)
    for field, digests in report["chunk_sha256"].items():
        assert len(digests) == 32, field
        assert all(
            len(digest) == 64 and set(digest) <= set("0123456789abcdef")
            for digest in digests
        )
        assert (
            _ordered_json_list_digest(digests)
            == _PRODUCTION_CHUNK_SHA256_MANIFEST[field]
        )


def test_production_validation_schedule_and_event_digests_are_pinned(
    production_schedule: depth.DepthSchedule,
) -> None:
    data = depth._encode_validation_data(
        production_schedule,
        depth.DepthGateConfig(),
    )

    assert _validation_data_digests(data) == _PRODUCTION_VALIDATION_SHA256


def test_encoded_schedule_hashing_rejects_empty_staging_chunk() -> None:
    config = _reduced_depth_config()
    schedule = depth._build_schedule(config)
    encoded = depth._encode_training_chunk(
        next(depth._iter_schedule_chunks(schedule, config)),
        config,
    )
    empty = dataclasses.replace(encoded, efforts=np.empty((0,), dtype=np.int32))

    with pytest.raises(ValueError, match="at least one update"):
        depth._update_encoded_schedule_hash_state(
            depth._new_encoded_schedule_hash_state(),
            empty,
            config,
        )


@pytest.mark.parametrize("mutation", ["update_count", "chunk_count"])
def test_encoded_schedule_hashing_rejects_incomplete_coverage(mutation: str) -> None:
    config = _reduced_depth_config()
    state = depth._new_encoded_schedule_hash_state()
    state.encoded_updates = config.training_updates
    state.chunk_count = config.staging_chunk_count
    if mutation == "update_count":
        state.encoded_updates -= 1
    else:
        state.chunk_count -= 1

    with pytest.raises(ValueError, match="encoded chunk"):
        depth._finish_encoded_schedule_report(state, config)


def test_training_chunk_encoding_matches_row_events_and_checkpoint_contracts() -> None:
    config = _reduced_depth_config()
    schedule = depth._build_schedule(config)
    schedule_chunk = next(depth._iter_schedule_chunks(schedule, config))

    encoded = depth._encode_training_chunk(schedule_chunk, config)

    assert isinstance(encoded, depth.DepthTrainingChunk)
    assert np.asarray(encoded.events).shape == (1, 19, 2, 47)
    assert np.asarray(encoded.targets).shape == (1, 19, 2)
    assert np.asarray(encoded.loss_weights).shape == (1, 19)
    assert np.asarray(encoded.advance_masks).shape == (1, 19, 2)
    assert np.array_equal(encoded.mapping_ids, schedule_chunk.training_mapping_ids)
    assert np.array_equal(encoded.efforts, schedule_chunk.training_efforts)
    assert np.array_equal(encoded.query_colors, schedule_chunk.training_query_colors)
    assert np.array_equal(
        encoded.presentation_orders,
        schedule_chunk.training_presentation_orders,
    )

    effort = int(schedule_chunk.training_efforts[0])
    for batch_index in range(config.batch_size):
        mapping = np.asarray(
            depth.unrank_ten_cycle(
                int(schedule_chunk.training_mapping_ids[0, batch_index])
            )
        )
        query = int(schedule_chunk.training_query_colors[0, batch_index])
        expected_events = np.zeros((19, 47), dtype=np.float32)
        expected_events[:11] = _encoded_cycle_episode(
            mapping,
            query,
            schedule_chunk.training_presentation_orders[0, batch_index],
            config.row_config,
        )
        contract = depth._checkpoint_contract(mapping, query, effort)

        assert np.array_equal(encoded.events[0, :, batch_index], expected_events)
        assert np.array_equal(encoded.targets[0, :, batch_index], contract.targets)
        assert np.array_equal(
            encoded.advance_masks[0, :, batch_index],
            contract.advance_mask,
        )
        assert np.array_equal(encoded.loss_weights[0], contract.loss_weights)


def test_batched_cycle_decoder_matches_scalar_unrank_at_catalog_boundaries() -> None:
    mapping_ids = np.asarray(
        [[0, 1, 9, 10], [12_345, math.factorial(9) - 2, math.factorial(9) - 1, 42]],
        dtype=np.int64,
    )

    decoded = depth._decode_ten_cycle_batch(mapping_ids)
    expected = np.asarray(
        [
            depth.unrank_ten_cycle(int(mapping_id))
            for mapping_id in mapping_ids.reshape(-1)
        ],
        dtype=np.int32,
    ).reshape(mapping_ids.shape + (10,))

    assert decoded.dtype == np.int32
    assert decoded.shape == (2, 4, 10)
    assert decoded.tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    "mapping_id",
    [0, 1, 9, 10, 12_345, math.factorial(9) - 1],
)
def test_batched_cycle_encoder_matches_scalar_for_all_queries_and_orders(
    mapping_id: int,
) -> None:
    config = _reduced_depth_config()
    mapping_ids = np.full((30,), mapping_id, dtype=np.int64)
    queries = np.tile(np.arange(10, dtype=np.int32), 3)
    base_orders = (
        np.arange(10, dtype=np.int32),
        np.arange(9, -1, -1, dtype=np.int32),
        np.roll(np.arange(10, dtype=np.int32), 3),
    )
    orders = np.asarray(
        [order for order in base_orders for _ in range(10)],
    )
    mappings = depth._decode_ten_cycle_batch(mapping_ids)

    batched = depth._encode_cycle_batch(
        mappings,
        queries,
        orders,
        config.row_config,
    )
    expected = np.stack(
        [
            depth._encode_cycle_episode(
                mappings[index],
                int(queries[index]),
                orders[index],
                config.row_config,
            )
            for index in range(mapping_ids.size)
        ],
        axis=1,
    )

    assert batched.shape == (11, mapping_ids.size, config.row_config.input_width)
    assert batched.dtype == np.float32
    assert batched.tobytes() == expected.tobytes()


@pytest.mark.parametrize("effort", [1, 2, 4, 8])
def test_batched_checkpoint_tensors_match_scalar_for_all_efforts(
    effort: int,
) -> None:
    config = _reduced_depth_config()
    ranks = np.asarray([0, 1, 12_345, math.factorial(9) - 1], dtype=np.int64)
    mapping_ids = np.repeat(ranks, 10).reshape(4, 10)
    queries = np.tile(np.arange(10, dtype=np.int32), (4, 1))
    efforts = np.full((4,), effort, dtype=np.int32)
    mappings = depth._decode_ten_cycle_batch(mapping_ids)

    targets, weights, advances = depth._checkpoint_tensors_batch(
        mappings,
        queries,
        efforts,
        sequence_length=config.sequence_length,
    )
    for update_index, effort_value in enumerate(efforts):
        for batch_index in range(10):
            contract = depth._checkpoint_contract(
                mappings[update_index, batch_index],
                int(queries[update_index, batch_index]),
                int(effort_value),
            )
            assert np.array_equal(
                targets[update_index, :, batch_index],
                contract.targets,
            )
            assert np.array_equal(
                advances[update_index, :, batch_index],
                contract.advance_mask,
            )
        assert np.array_equal(weights[update_index], contract.loss_weights)


@pytest.mark.parametrize(
    "mutation",
    [
        "mapping_rank",
        "batch_size",
        "effort_shape",
        "query_shape",
        "presentation_shape",
    ],
)
def test_training_encoder_rejects_inconsistent_schedule_shapes(
    mutation: str,
) -> None:
    config = _reduced_depth_config()
    chunk = next(depth._iter_schedule_chunks(depth._build_schedule(config), config))
    replacements: dict[str, np.ndarray] = {}
    if mutation == "mapping_rank":
        replacements["training_mapping_ids"] = np.asarray(
            chunk.training_mapping_ids
        ).reshape(-1)
    elif mutation == "batch_size":
        replacements["training_mapping_ids"] = np.asarray(
            chunk.training_mapping_ids
        )[:, :1]
    elif mutation == "effort_shape":
        replacements["training_efforts"] = np.asarray(
            chunk.training_efforts
        ).reshape(1, 1)
    elif mutation == "query_shape":
        replacements["training_query_colors"] = np.asarray(
            chunk.training_query_colors
        )[:, :1]
    elif mutation == "presentation_shape":
        replacements["training_presentation_orders"] = np.asarray(
            chunk.training_presentation_orders
        )[..., :9]

    malformed = dataclasses.replace(chunk, **replacements)

    with pytest.raises(ValueError, match="shape|batch size"):
        depth._encode_training_chunk(malformed, config)


def test_training_encoder_rejects_wrong_row_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _reduced_depth_config()
    chunk = next(depth._iter_schedule_chunks(depth._build_schedule(config), config))
    monkeypatch.setattr(
        depth,
        "_encode_cycle_batch",
        lambda *args: np.zeros(
            (10, config.batch_size, config.row_config.input_width),
            dtype=np.float32,
        ),
    )

    with pytest.raises(ValueError, match="exactly 11 rows"):
        depth._encode_training_chunk(chunk, config)


def test_training_encoder_rejects_undeclared_batch_effort() -> None:
    config = _reduced_depth_config()
    chunk = next(depth._iter_schedule_chunks(depth._build_schedule(config), config))
    malformed = dataclasses.replace(
        chunk,
        training_efforts=np.asarray([3], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="effort"):
        depth._encode_training_chunk(malformed, config)


def test_validation_encoding_contains_exact_intact_and_control_episodes() -> None:
    config = _reduced_depth_config()
    schedule = depth._build_schedule(config)

    data = depth._encode_validation_data(schedule, config)

    assert isinstance(data, depth.DepthValidationData)
    assert np.asarray(data.intact).shape == (19, 2, 47)
    assert np.asarray(data.shuffled).shape == (19, 2, 47)
    assert np.asarray(data.no_context).shape == (19, 2, 47)
    assert np.asarray(data.targets_by_depth).shape == (9, 2)
    assert np.asarray(data.advance_masks).shape == (19, 2)
    assert np.all(data.advance_masks)
    assert np.array_equal(data.mapping_ids, schedule.validation_mapping_ids)
    assert np.array_equal(data.query_colors, schedule.validation_query_colors)
    assert np.array_equal(
        data.presentation_orders,
        schedule.validation_presentation_orders,
    )
    assert np.asarray(data.shuffled_shifts).shape == (2,)

    for episode_index in range(config.validation_episodes):
        mapping = np.asarray(
            depth.unrank_ten_cycle(int(schedule.validation_mapping_ids[episode_index]))
        )
        query = int(schedule.validation_query_colors[episode_index])
        order = schedule.validation_presentation_orders[episode_index]
        shift, shuffled_mapping = depth._select_shuffled_rotation(mapping, query)
        expected_intact = np.zeros((19, 47), dtype=np.float32)
        expected_shuffled = np.zeros_like(expected_intact)
        expected_no_context = np.zeros_like(expected_intact)
        expected_intact[:11] = _encoded_cycle_episode(
            mapping,
            query,
            order,
            config.row_config,
        )
        expected_shuffled[:11] = _encoded_cycle_episode(
            shuffled_mapping,
            query,
            order,
            config.row_config,
        )
        expected_no_context[10] = expected_intact[10]
        expected_targets = np.asarray(
            [_iterate(mapping, query, depth_index + 1) for depth_index in range(9)],
            dtype=np.int32,
        )

        assert int(data.shuffled_shifts[episode_index]) == shift
        assert np.array_equal(data.intact[:, episode_index], expected_intact)
        assert np.array_equal(data.shuffled[:, episode_index], expected_shuffled)
        assert np.array_equal(data.no_context[:, episode_index], expected_no_context)
        assert np.array_equal(data.targets_by_depth[:, episode_index], expected_targets)
        assert np.array_equal(
            data.intact[10, episode_index], data.shuffled[10, episode_index]
        )
        assert np.array_equal(
            data.intact[10, episode_index], data.no_context[10, episode_index]
        )


@pytest.mark.parametrize("mutation", ["identity_shape", "presentation_shape"])
def test_validation_encoder_rejects_inconsistent_schedule_shapes(
    mutation: str,
) -> None:
    config = _reduced_depth_config()
    schedule = depth._build_schedule(config)
    if mutation == "identity_shape":
        schedule = dataclasses.replace(
            schedule,
            validation_mapping_ids=np.asarray(schedule.validation_mapping_ids)[:1],
        )
    else:
        schedule = dataclasses.replace(
            schedule,
            validation_presentation_orders=np.asarray(
                schedule.validation_presentation_orders
            )[..., :9],
        )

    with pytest.raises(ValueError, match="validation"):
        depth._encode_validation_data(schedule, config)


def test_validation_encoder_rejects_control_that_retains_intact_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _reduced_depth_config()
    schedule = depth._build_schedule(config)

    def unchanged_control(
        mapping: np.ndarray,
        query_color: int,
    ) -> tuple[int, np.ndarray]:
        return 1, np.asarray(mapping)

    monkeypatch.setattr(depth, "_select_shuffled_rotation", unchanged_control)

    with pytest.raises(ValueError, match="retains an intact"):
        depth._encode_validation_data(schedule, config)


def test_model_config_preserves_gate_b_semantic_indices_and_topology() -> None:
    config = _reduced_depth_config()
    model_config = depth._model_config(config, batch_size=3)
    indices = associative_memory_feature_indices(config.row_config)

    assert model_config.input_width == 47
    assert model_config.batch_size == 3
    assert model_config.neuron_count == config.neuron_count
    assert model_config.recurrent_edges == config.recurrent_edges
    assert model_config.max_latent_steps == 8
    assert model_config.readout_width == config.readout_width
    assert model_config.color_rank == config.color_rank
    assert model_config.context_memory_width == config.context_memory_width
    assert model_config.memory_decay == config.memory_decay
    assert model_config.trace_decay == config.trace_decay
    assert model_config.event_valid_index == config.row_config.valid_slice.start
    assert model_config.demonstration_phase_index == config.row_config.phase_slice.start
    assert model_config.query_phase_index == config.row_config.phase_slice.start + 1
    assert (
        model_config.input_side_valid_index == config.row_config.side_valid_slice.start
    )
    assert (
        model_config.output_side_valid_index
        == config.row_config.side_valid_slice.start + 1
    )
    assert model_config.memory_key_indices == indices.key_indices
    assert model_config.memory_value_indices == indices.value_indices
    assert model_config.seed == config.model_seed


def test_training_driver_is_one_jit_with_internal_brainstate_for_loop() -> None:
    source = inspect.getsource(depth._make_pp_prop_trainer)
    function = ast.parse(source).body[0]
    assert isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
    nested = {
        node.name: node
        for node in ast.walk(function)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    train_chunk = nested["train_chunk"]

    assert len(train_chunk.decorator_list) == 1
    assert ast.unparse(train_chunk.decorator_list[0]) == "brainstate.transform.jit"
    assert not any(
        isinstance(node, (ast.For, ast.While)) for node in ast.walk(train_chunk)
    )
    for_loop_calls = [
        node
        for node in ast.walk(train_chunk)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "brainstate.transform.for_loop"
    ]
    assert len(for_loop_calls) == 1
    assert ast.unparse(for_loop_calls[0].args[0]) == "train_one"
    assert ast.unparse(for_loop_calls[0].args[1]) == (
        "(events, targets, loss_weights, advance_masks)"
    )


@pytest.mark.parametrize("effort", [1, 2, 4, 8])
def test_padded_t19_matches_compact_prefix_finite_window_pp_prop(
    effort: int,
) -> None:
    config, padded, compact = _depth_objective_probe_inputs(effort)

    padded_objective = _probe_objective(config, padded)
    compact_objective = _probe_objective(config, compact)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        padded_gradients = chunked_online_param_gradients(
            lambda: _DepthObjectiveProbe(config),
            padded,
            algo_factory=_depth_probe_pp_prop,
            chunk_size=1,
        )
        compact_gradients = chunked_online_param_gradients(
            lambda: _DepthObjectiveProbe(config),
            compact,
            algo_factory=_depth_probe_pp_prop,
            chunk_size=1,
        )

    assert padded_objective == pytest.approx(compact_objective, rel=0.0, abs=1e-7)
    assert gradient_norm(compact_gradients) > 1e-8
    assert relative_deviation(padded_gradients, compact_gradients) == pytest.approx(
        0.0,
        abs=1e-7,
    )
    padded_leaves = flat_gradient_leaves(padded_gradients)
    compact_leaves = flat_gradient_leaves(compact_gradients)
    assert padded_leaves.keys() == compact_leaves.keys()
    for path in padded_leaves:
        np.testing.assert_allclose(
            padded_leaves[path],
            compact_leaves[path],
            rtol=0.0,
            atol=1e-7,
            err_msg=path,
        )


@pytest.fixture(scope="module")
def reduced_gate_run() -> dict[str, Any]:
    config = _reduced_depth_config()
    schedule = depth._build_schedule(config)
    chunk = depth._encode_training_chunk(
        next(depth._iter_schedule_chunks(schedule, config)),
        config,
    )
    validation = depth._encode_validation_data(schedule, config)
    model = LatentWorkspaceModel(
        depth._model_config(config, batch_size=config.batch_size)
    )

    initialization = depth._initialization_report(model, config)
    trainer = depth._make_pp_prop_trainer(model, config)
    telemetry = trainer.train_chunk(
        chunk.events,
        chunk.targets,
        chunk.loss_weights,
        chunk.advance_masks,
    )
    evaluation = depth._evaluate_model(model, validation, config)
    return {
        "config": config,
        "initialization": initialization,
        "trainer": trainer,
        "telemetry": telemetry,
        "evaluation": evaluation,
    }


def test_tree_telemetry_rejects_subject_without_array_leaves() -> None:
    with pytest.raises(RuntimeError, match="no array leaves"):
        depth._tree_telemetry(())


def test_reduced_gate_b_runs_real_pp_prop_train_and_evaluation_smoke(
    reduced_gate_run: dict[str, Any],
) -> None:
    config = reduced_gate_run["config"]
    initialization = reduced_gate_run["initialization"]
    trainer = reduced_gate_run["trainer"]
    telemetry = reduced_gate_run["telemetry"]
    evaluation = reduced_gate_run["evaluation"]

    assert initialization["fresh_model"] is True
    assert initialization["model_seed"] == config.model_seed
    assert initialization["configuration"] == dataclasses.asdict(config)
    assert initialization["parameter_count"] > 0
    assert len(initialization["parameter_sha256"]) == 64
    assert initialization["compiler"]["available"] is True
    assert trainer.algorithm == "production_pp_prop"
    assert trainer.compiler["available"] is True
    assert np.asarray(telemetry["loss"]).shape == (1,)
    assert np.isfinite(np.asarray(telemetry["loss"])).all()
    assert isinstance(evaluation, Mapping)
    assert evaluation["finite"] is True
    assert set(evaluation) == {"finite", "h0_proper", "depths", "efforts"}
    assert set(evaluation["efforts"]) == {"1", "2", "4", "8"}
    h0_hash = evaluation["h0_proper"]["prediction_sha256"]
    assert evaluation["h0_proper"]["checkpoint"] == 0
    assert len(h0_hash) == 64
    for effort in depth.QUALIFYING_EFFORTS:
        evidence = evaluation["efforts"][str(effort)]
        assert set(evidence) == {
            "intact",
            "shuffled",
            "no_context",
            "h0_final_target",
            "intact_minus_h0",
            "intact_minus_shuffled",
        }
        assert evidence["intact"]["checkpoint"] == effort
        assert evidence["shuffled"]["checkpoint"] == effort
        assert evidence["no_context"]["checkpoint"] == effort
        assert evidence["h0_final_target"]["checkpoint"] == 0
        assert evidence["h0_final_target"]["prediction_sha256"] == h0_hash
        for metric_name in ("intact", "shuffled", "no_context", "h0_final_target"):
            metric = evidence[metric_name]
            assert metric["count"] == config.validation_episodes
            assert len(metric["prediction_sha256"]) == 64
            assert np.isfinite(metric["accuracy"])
        assert np.isfinite(evidence["intact_minus_h0"])
        assert np.isfinite(evidence["intact_minus_shuffled"])


def test_train_chunk_retains_complete_finite_telemetry(
    reduced_gate_run: dict[str, Any],
) -> None:
    telemetry = reduced_gate_run["telemetry"]
    categories = {
        "logits",
        "model_states",
        "gradients",
        "pp_prop_traces",
        "adam",
        "parameters",
    }

    assert set(telemetry) == {"loss", "finite", "max_abs", "value_count"}
    for section in ("finite", "max_abs", "value_count"):
        assert set(telemetry[section]) == categories
    for category in categories:
        finite = np.asarray(telemetry["finite"][category])
        maximum = np.asarray(telemetry["max_abs"][category])
        count = np.asarray(telemetry["value_count"][category])
        assert finite.shape == maximum.shape == count.shape == (1,)
        assert np.issubdtype(finite.dtype, np.bool_)
        assert np.all(finite)
        assert np.all(np.isfinite(maximum))
        assert np.all(maximum >= 0.0)
        assert np.issubdtype(count.dtype, np.integer)
        assert np.all(count > 0)


def test_train_depth_gate_streams_chunks_through_one_prebuilt_trainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = dataclasses.replace(
        _reduced_depth_config(),
        training_updates=2,
        staging_chunk_updates=1,
    )
    schedule = depth._build_schedule(config)
    expected_schedule_report = depth._encoded_schedule_report(schedule, config)
    model = LatentWorkspaceModel(
        depth._model_config(config, batch_size=config.batch_size)
    )
    initial_sha = legacy._array_digest(legacy._parameter_values(model))
    calls = {"make_trainer": 0, "encode_chunk": 0, "train_chunk": 0}
    real_make_trainer = depth._make_pp_prop_trainer
    real_encode_chunk = depth._encode_training_chunk

    def make_trainer(*args: Any, **kwargs: Any) -> Any:
        calls["make_trainer"] += 1
        trainer = real_make_trainer(*args, **kwargs)
        real_train_chunk = trainer.train_chunk

        def train_chunk(*chunk_args: Any, **chunk_kwargs: Any) -> Any:
            calls["train_chunk"] += 1
            return real_train_chunk(*chunk_args, **chunk_kwargs)

        trainer.train_chunk = train_chunk
        return trainer

    def encode_chunk(*args: Any, **kwargs: Any) -> depth.DepthTrainingChunk:
        calls["encode_chunk"] += 1
        return real_encode_chunk(*args, **kwargs)

    monkeypatch.setattr(depth, "_make_pp_prop_trainer", make_trainer)
    monkeypatch.setattr(depth, "_encode_training_chunk", encode_chunk)

    training, schedule_report = depth._train_depth_gate(model, schedule, config)

    assert calls == {"make_trainer": 1, "encode_chunk": 2, "train_chunk": 2}
    assert schedule_report == expected_schedule_report
    assert set(training) == {
        "algorithm",
        "executed_updates",
        "batch_size",
        "chunk_count",
        "chunk_updates",
        "effort_update_counts",
        "initialization_parameter_sha256",
        "final_parameter_sha256",
        "losses",
        "finite",
        "max_abs",
        "value_count",
        "compiler",
        "compile_warnings",
    }
    assert training["algorithm"] == "production_pp_prop"
    assert training["executed_updates"] == 2
    assert training["batch_size"] == 2
    assert training["chunk_count"] == 2
    assert training["chunk_updates"] == 1
    assert training["effort_update_counts"] == {"1": 1, "2": 1, "4": 0, "8": 0}
    assert training["initialization_parameter_sha256"] == initial_sha
    assert training["final_parameter_sha256"] != initial_sha
    assert len(training["losses"]) == 2
    assert np.isfinite(training["losses"]).all()
    assert training["compiler"]["available"] is True
    categories = {
        "logits",
        "model_states",
        "gradients",
        "pp_prop_traces",
        "adam",
        "parameters",
    }
    assert set(training["finite"]) == categories
    assert set(training["max_abs"]) == categories
    assert set(training["value_count"]) == categories
    assert all(training["finite"].values())
    assert all(
        np.isfinite(value) and value >= 0.0 for value in training["max_abs"].values()
    )
    assert all(value > 0 for value in training["value_count"].values())


def test_train_depth_gate_host_loop_never_calls_model_or_reencodes_schedule() -> None:
    source = inspect.getsource(depth._train_depth_gate)
    function = ast.parse(source).body[0]
    assert isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
    loops = [node for node in ast.walk(function) if isinstance(node, ast.For)]
    trainer_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "_make_pp_prop_trainer"
    ]

    assert len(loops) == 1
    assert ast.unparse(loops[0].iter).startswith("_iter_schedule_chunks(")
    assert len(trainer_calls) == 1
    assert trainer_calls[0].lineno < loops[0].lineno
    assert "trainer.train_chunk(" in source
    assert "_encode_training_chunk(" in source
    assert not any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "_encoded_schedule_report"
        for node in ast.walk(function)
    )
    assert "learner.etrace_grad(" not in source
    forbidden_model_calls = {
        "model",
        "model.update",
        "model.cell_step",
        "learner",
        "learner.etrace_grad",
    }
    assert not any(
        isinstance(node, ast.Call) and ast.unparse(node.func) in forbidden_model_calls
        for node in ast.walk(function)
    )
    assert not any(isinstance(node, ast.While) for node in ast.walk(function))


def test_gate_b_initialization_admission_recomputes_full_authenticated_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = depth.DepthGateConfig()
    model = LatentWorkspaceModel(
        depth._model_config(config, batch_size=config.batch_size)
    )
    passing = _passing_depth_report()
    gate_a = passing["prerequisites"]["gate_a"]
    source_start = passing["source_start"]
    source_end = passing["source_end"]
    source_files = passing["source_files"]
    environment = passing["environment"]
    lifecycle: list[str] = []
    real_initialization_report = depth._initialization_report

    def initialization_report(
        actual_model: LatentWorkspaceModel,
        actual_config: depth.DepthGateConfig,
    ) -> dict[str, Any]:
        result = real_initialization_report(actual_model, actual_config)
        lifecycle.append("initialization_complete")
        return result

    def source_end_reporter() -> Mapping[str, Any]:
        lifecycle.append("source_end_captured")
        return source_end

    monkeypatch.setattr(depth, "_initialization_report", initialization_report)

    report = depth._gate_b_initialization_report(
        model,
        config,
        gate_a=gate_a,
        source_start=source_start,
        source_end_reporter=source_end_reporter,
        source_files=source_files,
        environment=environment,
    )

    assert lifecycle == ["initialization_complete", "source_end_captured"]
    assert set(report) == {
        "schema_version",
        "control",
        "qualification_regime",
        "config",
        "prerequisites",
        "initialization",
        "source_start",
        "source_end",
        "source_files",
        "environment",
        "qualification",
    }
    assert report["schema_version"] == 1
    assert report["control"] == "example21_gate_b_initialization_admission"
    assert report["qualification_regime"] == "preregistered_full"
    assert report["config"] == dataclasses.asdict(config)
    assert report["prerequisites"] == {"gate_a": gate_a}
    assert report["source_start"] == source_start
    assert report["source_end"] == source_end
    assert report["source_files"] == source_files
    assert report["environment"] == environment
    assert report["initialization"]["fresh_model"] is True
    assert report["initialization"]["model_seed"] == config.model_seed
    assert report["initialization"]["configuration"] == dataclasses.asdict(config)
    assert report["initialization"]["parameter_count"] > 0
    assert len(report["initialization"]["parameter_sha256"]) == 64
    assert report["initialization"]["compiler"]["available"] is True
    assert report["qualification"]["passed"] is True
    assert all(report["qualification"]["criteria"].values())
    assert report["qualification"]["interpretation"] == (
        "gate_b_initialization_admission_passed"
    )


def test_run_depth_gate_captures_source_end_once_after_evaluation() -> None:
    signature = inspect.signature(depth.run_depth_gate)
    assert "source_end_reporter" in signature.parameters
    assert "source_end" not in signature.parameters

    function = ast.parse(inspect.getsource(depth.run_depth_gate)).body[0]
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    evaluations = [
        node for node in calls if ast.unparse(node.func) == "_evaluate_model"
    ]
    source_end_captures = [
        node for node in calls if ast.unparse(node.func) == "source_end_reporter"
    ]

    assert len(evaluations) == 1
    assert len(source_end_captures) == 1
    assert evaluations[0].lineno < source_end_captures[0].lineno


def test_run_depth_gate_rejects_precomputed_source_end_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_model(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("model constructed before reporter validation")

    monkeypatch.setattr(depth, "LatentWorkspaceModel", forbidden_model)

    with pytest.raises(TypeError, match="callable|source_end_reporter"):
        depth.run_depth_gate(
            _reduced_depth_config(),
            prerequisites={},
            source_start={},
            source_end_reporter={},
            source_files={},
            environment={},
        )


@pytest.mark.parametrize(
    "mutation",
    ["raw_initialization", "unwrapped_admission", "missing_finite", "stored_qualification"],
)
def test_run_depth_gate_rejects_unauthenticated_initialization_before_training(
    mutation: str,
    reduced_gate_run: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = reduced_gate_run["config"]
    passing = _passing_depth_report()
    admission = {
        "schema_version": 1,
        "control": "example21_gate_b_initialization_admission",
        "qualification_regime": config.qualification_regime,
        "config": dataclasses.asdict(config),
        "prerequisites": {"gate_a": passing["prerequisites"]["gate_a"]},
        "initialization": copy.deepcopy(reduced_gate_run["initialization"]),
        "source_start": passing["source_start"],
        "source_end": passing["source_end"],
        "source_files": passing["source_files"],
        "environment": passing["environment"],
    }
    admission["qualification"] = depth._gate_b_initialization_qualification(
        admission,
        config,
    )
    wrapper: Any = _gate_b_initialization_wrapper(admission)
    if mutation == "raw_initialization":
        wrapper = wrapper["admission"]["initialization"]
    elif mutation == "unwrapped_admission":
        wrapper = wrapper["admission"]
    elif mutation == "missing_finite":
        del wrapper["admission"]["initialization"]["parameters_finite"]
        _refresh_gate_b_initialization_wrapper_hashes(wrapper)
    elif mutation == "stored_qualification":
        wrapper["admission"]["qualification"]["passed"] = True
        wrapper["admission"]["qualification"]["interpretation"] = (
            "gate_b_initialization_admission_passed"
        )
        _refresh_gate_b_initialization_wrapper_hashes(wrapper)

    training_reached = False
    source_end_calls = 0

    def forbidden_training(*args: Any, **kwargs: Any) -> Any:
        nonlocal training_reached
        training_reached = True
        raise AssertionError("training reached before authentication")

    def source_end_reporter() -> Mapping[str, Any]:
        nonlocal source_end_calls
        source_end_calls += 1
        return passing["source_end"]

    monkeypatch.setattr(depth, "_train_depth_gate", forbidden_training)

    with pytest.raises(RuntimeError, match="initialization|admission|authentication"):
        depth.run_depth_gate(
            config,
            prerequisites={
                "gate_a": passing["prerequisites"]["gate_a"],
                "gate_b_initialization": wrapper,
            },
            source_start=passing["source_start"],
            source_end_reporter=source_end_reporter,
            source_files=passing["source_files"],
            environment=passing["environment"],
        )

    assert training_reached is False
    assert source_end_calls == 0


def test_run_depth_gate_rejects_nonmapping_source_end_after_evaluation(
    reduced_gate_run: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = reduced_gate_run["config"]
    passing = _passing_depth_report()
    admission = {
        "schema_version": 1,
        "control": "example21_gate_b_initialization_admission",
        "qualification_regime": config.qualification_regime,
        "config": dataclasses.asdict(config),
        "prerequisites": {"gate_a": passing["prerequisites"]["gate_a"]},
        "initialization": copy.deepcopy(reduced_gate_run["initialization"]),
        "source_start": passing["source_start"],
        "source_end": passing["source_end"],
        "source_files": passing["source_files"],
        "environment": passing["environment"],
    }
    admission["qualification"] = depth._gate_b_initialization_qualification(
        admission,
        config,
    )
    lifecycle: list[str] = []

    def evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        lifecycle.append("evaluation")
        return {}

    def source_end_reporter() -> list[Any]:
        lifecycle.append("source_end")
        return []

    monkeypatch.setattr(depth, "_train_depth_gate", lambda *args: ({}, {}))
    monkeypatch.setattr(depth, "_encode_validation_data", lambda *args: object())
    monkeypatch.setattr(depth, "_evaluate_model", evaluate)

    with pytest.raises(TypeError, match="must return a mapping"):
        depth.run_depth_gate(
            config,
            prerequisites={
                "gate_a": passing["prerequisites"]["gate_a"],
                "gate_b_initialization": _gate_b_initialization_wrapper(admission),
            },
            source_start=passing["source_start"],
            source_end_reporter=source_end_reporter,
            source_files=passing["source_files"],
            environment=passing["environment"],
        )

    assert lifecycle == ["evaluation", "source_end"]


def test_reduced_run_depth_gate_executes_real_pp_prop_and_assembles_strict_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _reduced_depth_config()
    passing = _passing_depth_report()
    initialization_model = LatentWorkspaceModel(
        depth._model_config(config, batch_size=config.batch_size)
    )
    initialization_admission = depth._gate_b_initialization_report(
        initialization_model,
        config,
        gate_a=passing["prerequisites"]["gate_a"],
        source_start=passing["source_start"],
        source_end_reporter=lambda: passing["source_end"],
        source_files=passing["source_files"],
        environment=passing["environment"],
    )
    initialization = initialization_admission["initialization"]
    initialization_wrapper = _gate_b_initialization_wrapper(
        initialization_admission
    )
    validation = depth._encode_validation_data(depth._build_schedule(config), config)
    expected_validation_digests = _validation_data_digests(validation)
    prerequisites = {
        "gate_a": passing["prerequisites"]["gate_a"],
        "gate_b_initialization": initialization_wrapper,
    }
    lifecycle: list[str] = []
    real_evaluate_model = depth._evaluate_model

    def evaluate_model(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = real_evaluate_model(*args, **kwargs)
        lifecycle.append("evaluation_finished")
        return result

    def source_end_reporter() -> Mapping[str, Any]:
        lifecycle.append("source_end_captured")
        return passing["source_end"]

    monkeypatch.setattr(depth, "_evaluate_model", evaluate_model)

    report = depth.run_depth_gate(
        config,
        prerequisites=prerequisites,
        source_start=passing["source_start"],
        source_end_reporter=source_end_reporter,
        source_files=passing["source_files"],
        environment=passing["environment"],
    )

    assert lifecycle == ["evaluation_finished", "source_end_captured"]
    assert set(report) == set(_passing_depth_report()) | {"total_wall_seconds"}
    assert report["schema_version"] == 1
    assert report["control"] == "example21_demonstrated_depth_gate_b"
    assert report["qualification_regime"] == "nonqualifying_abbreviated"
    assert report["config"] == dataclasses.asdict(config)
    assert report["prerequisites"] == prerequisites
    assert report["learner"]["algorithm"] == "production_pp_prop"
    assert report["training"]["executed_updates"] == 1
    assert report["training"]["batch_size"] == 2
    assert report["training"]["chunk_count"] == 1
    assert len(report["training"]["losses"]) == 1
    assert np.isfinite(report["training"]["losses"]).all()
    assert (
        report["training"]["initialization_parameter_sha256"]
        == (initialization["parameter_sha256"])
    )
    assert (
        report["evaluation"]["initialization_parameter_sha256"]
        == (initialization["parameter_sha256"])
    )
    assert set(report["evaluation"]["depths"]) == {str(index) for index in range(9)}
    assert report["data"]["schedule"]["chunk_count"] == 1
    assert report["data"]["schedule"]["training_updates"] == 1
    assert report["data"]["validation"]["episode_count"] == 2
    assert report["data"]["validation"]["sha256"] == expected_validation_digests
    assert report["data"]["held_out_invariants"]["validation_episode_count"] == 2
    assert report["qualification"]["passed"] is False
    assert report["qualification"]["interpretation"] == (
        "nonqualifying_abbreviated_no_capability_conclusion"
    )
    assert np.isfinite(report["total_wall_seconds"])
    assert report["total_wall_seconds"] >= 0.0


def test_evaluation_retains_all_checkpoint_metrics_from_shared_predictions(
    reduced_gate_run: dict[str, Any],
) -> None:
    config = reduced_gate_run["config"]
    evaluation = reduced_gate_run["evaluation"]
    metric_keys = {
        "correct",
        "count",
        "accuracy",
        "wilson_95_lower",
        "wilson_95_upper",
        "prediction_histogram",
        "prediction_sha256",
        "checkpoint",
    }

    assert set(evaluation["depths"]) == {str(index) for index in range(9)}
    assert evaluation["h0_proper"] == evaluation["depths"]["0"]["intact"]
    for depth_index in range(9):
        evidence = evaluation["depths"][str(depth_index)]
        assert set(evidence) == {"intact", "shuffled", "no_context"}
        for arm in ("intact", "shuffled", "no_context"):
            metric = evidence[arm]
            assert set(metric) == metric_keys
            assert metric["checkpoint"] == depth_index
            assert metric["count"] == config.validation_episodes
            assert 0 <= metric["correct"] <= metric["count"]
            assert metric["accuracy"] == pytest.approx(
                metric["correct"] / metric["count"],
                rel=0.0,
                abs=0.0,
            )
            assert len(metric["prediction_histogram"]) == 10
            assert sum(metric["prediction_histogram"]) == metric["count"]
            assert len(metric["prediction_sha256"]) == 64
            assert 0.0 <= metric["wilson_95_lower"] <= metric["accuracy"]
            assert metric["accuracy"] <= math.nextafter(
                metric["wilson_95_upper"], math.inf
            )
            assert metric["wilson_95_upper"] <= 1.0

    h0_hash = evaluation["h0_proper"]["prediction_sha256"]
    for effort in depth.QUALIFYING_EFFORTS:
        effort_evidence = evaluation["efforts"][str(effort)]
        depth_evidence = evaluation["depths"][str(effort)]
        for arm in ("intact", "shuffled", "no_context"):
            assert effort_evidence[arm] == depth_evidence[arm]
        assert effort_evidence["h0_final_target"]["prediction_sha256"] == h0_hash


def test_first_valid_rotation_preserves_marginal_and_breaks_every_depth(
    production_schedule: depth.DepthSchedule,
) -> None:
    for mapping_id, query_color in zip(
        np.asarray(production_schedule.validation_mapping_ids),
        np.asarray(production_schedule.validation_query_colors),
        strict=True,
    ):
        mapping = np.asarray(depth.unrank_ten_cycle(int(mapping_id)))
        shift, shuffled = depth._select_shuffled_rotation(mapping, int(query_color))
        shuffled = np.asarray(shuffled)
        valid_shifts = [
            candidate
            for candidate in range(1, 10)
            if np.all((mapping + candidate) % 10 != mapping)
            and all(
                _iterate((mapping + candidate) % 10, int(query_color), effort + 1)
                != _iterate(mapping, int(query_color), effort + 1)
                for effort in depth.QUALIFYING_EFFORTS
            )
        ]

        assert valid_shifts
        assert shift == valid_shifts[0]
        assert np.array_equal(shuffled, (mapping + shift) % 10)
        assert sorted(shuffled.tolist()) == list(range(10))
        assert np.all(shuffled != mapping)
        assert all(
            _iterate(shuffled, int(query_color), effort + 1)
            != _iterate(mapping, int(query_color), effort + 1)
            for effort in depth.QUALIFYING_EFFORTS
        )


@pytest.mark.parametrize("effort", [1, 2, 4, 8])
def test_checkpoint_contract_has_exact_targets_masks_and_compact_prefix(
    effort: int,
) -> None:
    mapping = np.asarray(depth.unrank_ten_cycle(12_345))
    query_color = 7
    contract = depth._checkpoint_contract(mapping, query_color, effort)
    targets = np.asarray(contract.targets)
    weights = np.asarray(contract.loss_weights)
    advances = np.asarray(contract.advance_mask)
    active_length = 11 + effort

    assert isinstance(contract, depth.CheckpointContract)
    assert targets.shape == weights.shape == advances.shape == (19,)
    assert np.issubdtype(targets.dtype, np.integer)
    assert np.issubdtype(weights.dtype, np.floating)
    assert np.issubdtype(advances.dtype, np.bool_)
    assert contract.active_length == active_length
    assert np.all(advances[:active_length])
    assert not np.any(advances[active_length:])
    assert np.all(weights[:10] == 0.0)
    assert np.all(weights[active_length:] == 0.0)
    assert np.allclose(
        weights[10:active_length],
        np.full((effort + 1,), 1.0 / (effort + 1)),
        rtol=0.0,
        atol=0.0,
    )
    assert math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-7)
    expected = np.asarray(
        [
            _iterate(mapping, query_color, applications)
            for applications in range(1, effort + 2)
        ]
    )
    assert np.array_equal(targets[10:active_length], expected)
    assert targets[10] == mapping[query_color]
    assert targets[10 + effort] != targets[10]


@pytest.mark.parametrize("effort", [0, 3, 9])
def test_checkpoint_contract_rejects_undeclared_effort(effort: int) -> None:
    with pytest.raises(ValueError, match="effort"):
        depth._checkpoint_contract(np.asarray(depth.unrank_ten_cycle(0)), 0, effort)


@pytest.mark.parametrize(
    ("mapping", "query"),
    [
        (np.arange(9, dtype=np.int32), 0),
        (depth.unrank_ten_cycle(0), True),
        (depth.unrank_ten_cycle(0), -1),
    ],
)
def test_checkpoint_contract_rejects_malformed_mapping_or_query(
    mapping: np.ndarray,
    query: Any,
) -> None:
    with pytest.raises(ValueError, match="mapping|query"):
        depth._checkpoint_contract(mapping, query, 1)


def test_missing_gate_b_evidence_fails_closed() -> None:
    qualification = depth._qualification_report({}, config=depth.DepthGateConfig())

    assert qualification["passed"] is False
    assert qualification["interpretation"] == (
        "gate_b_failed_stop_no_capability_conclusion"
    )
    assert qualification["criteria"]
    assert not any(qualification["criteria"].values())


def test_gate_b_qualification_passes_only_complete_recomputed_evidence() -> None:
    report = _passing_depth_report()

    qualification = depth._qualification_report(
        report,
        config=depth.DepthGateConfig(),
    )

    assert qualification["passed"] is True
    assert all(qualification["criteria"].values())
    assert qualification["interpretation"] == (
        "gate_b_passed_demonstrated_depth_application"
    )
    assert qualification != report["qualification"]


@pytest.mark.parametrize("mutation", ["schema", "counter_type", "chunk_type"])
def test_schedule_evidence_rejects_malformed_structure_and_types(
    mutation: str,
) -> None:
    schedule = copy.deepcopy(_placeholder_schedule_report())
    if mutation == "schema":
        schedule["unexpected"] = True
    elif mutation == "counter_type":
        schedule["chunk_count"] = True
    else:
        schedule["chunk_sha256"] = []

    assert depth._schedule_evidence_complete(schedule) is False


@pytest.mark.parametrize("mutation", ["schema", "count_type", "digest_type"])
def test_validation_evidence_rejects_malformed_structure_and_types(
    mutation: str,
) -> None:
    validation: dict[str, Any] = {
        "episode_count": 512,
        "sha256": dict(_PRODUCTION_VALIDATION_SHA256),
    }
    if mutation == "schema":
        validation["unexpected"] = True
    elif mutation == "count_type":
        validation["episode_count"] = True
    else:
        validation["sha256"] = []

    assert depth._validation_evidence_complete(validation) is False


@pytest.mark.parametrize("mutation", ["provenance", "admission_type", "schema"])
def test_initialization_wrapper_rejects_malformed_provenance_or_admission(
    mutation: str,
) -> None:
    report = _passing_depth_report()
    wrapper = copy.deepcopy(report["prerequisites"]["gate_b_initialization"])
    if mutation == "provenance":
        wrapper["target"] = "gate_b_formal"
    elif mutation == "admission_type":
        wrapper["admission"] = []
    else:
        del wrapper["admission"]["initialization"]

    with pytest.raises(ValueError, match="provenance|admission"):
        depth._validated_initialization_admission(
            wrapper,
            depth.DepthGateConfig(),
            source_start=report["source_start"],
            environment=report["environment"],
            require_pass=True,
        )


def test_initialization_wrapper_requires_passing_admission_for_formal_use() -> None:
    config = _reduced_depth_config()
    report = _passing_depth_report()
    admission = copy.deepcopy(
        report["prerequisites"]["gate_b_initialization"]["admission"]
    )
    admission["qualification_regime"] = config.qualification_regime
    admission["config"] = dataclasses.asdict(config)
    admission["initialization"]["configuration"] = dataclasses.asdict(config)
    admission["qualification"] = depth._gate_b_initialization_qualification(
        admission,
        config,
    )
    wrapper = _gate_b_initialization_wrapper(admission)

    with pytest.raises(ValueError, match="did not pass"):
        depth._validated_initialization_admission(
            wrapper,
            config,
            source_start=report["source_start"],
            environment=report["environment"],
            require_pass=True,
        )


@pytest.mark.parametrize(
    "mutation",
    ["telemetry_schema", "finite", "maximum", "count"],
)
def test_training_evidence_rejects_invalid_finite_telemetry(mutation: str) -> None:
    report = _passing_depth_report()
    training = report["training"]
    if mutation == "telemetry_schema":
        training["finite"].pop("logits")
    elif mutation == "finite":
        training["finite"]["logits"] = False
    elif mutation == "maximum":
        training["max_abs"]["logits"] = -1.0
    else:
        training["value_count"]["logits"] = True

    assert depth._training_evidence_complete(
        report,
        depth.DepthGateConfig(),
    ) is False


@pytest.mark.parametrize(
    "mutation",
    [
        "finite",
        "depths_schema",
        "arms_schema",
        "metric_schema",
        "h0_mismatch",
        "efforts_schema",
        "effort_schema",
        "effort_arm_mismatch",
        "h0_final_mismatch",
        "delta_type",
        "delta_value",
    ],
)
def test_evaluation_evidence_rejects_malformed_or_inconsistent_metrics(
    mutation: str,
) -> None:
    evaluation = copy.deepcopy(_passing_depth_evaluation())
    if mutation == "finite":
        evaluation["finite"] = False
    elif mutation == "depths_schema":
        evaluation["depths"] = []
    elif mutation == "arms_schema":
        evaluation["depths"]["0"] = []
    elif mutation == "metric_schema":
        evaluation["depths"]["0"]["intact"].pop("checkpoint")
    elif mutation == "h0_mismatch":
        evaluation["h0_proper"] = _depth_metric(
            399,
            checkpoint=0,
            prediction_label="different-h0",
        )
    elif mutation == "efforts_schema":
        evaluation["efforts"] = []
    elif mutation == "effort_schema":
        evaluation["efforts"]["1"].pop("intact_minus_h0")
    elif mutation == "effort_arm_mismatch":
        evaluation["efforts"]["1"]["intact"] = _depth_metric(
            399,
            checkpoint=1,
            prediction_label="different-effort",
        )
    elif mutation == "h0_final_mismatch":
        evaluation["efforts"]["1"]["h0_final_target"][
            "prediction_sha256"
        ] = "0" * 64
    elif mutation == "delta_type":
        evaluation["efforts"]["1"]["intact_minus_h0"] = True
    else:
        evaluation["efforts"]["1"]["intact_minus_h0"] += 0.01

    assert depth._evaluation_evidence_complete(
        evaluation,
        depth.DepthGateConfig(),
    ) is False


@pytest.mark.parametrize(
    ("mutation", "criterion"),
    [
        ("schema_control", "schema_and_control"),
        ("config", "preregistered_configuration"),
        ("config_bool", "preregistered_configuration"),
        ("gate_a", "gate_a_prerequisite_authenticated"),
        ("source_files", "gate_a_prerequisite_authenticated"),
        ("initialization", "gate_b_initialization_authenticated"),
        ("initialization_raw", "gate_b_initialization_authenticated"),
        ("initialization_unwrapped_admission", "gate_b_initialization_authenticated"),
        ("initialization_parameters_finite", "gate_b_initialization_authenticated"),
        ("initialization_qualification", "gate_b_initialization_authenticated"),
        ("initialization_source_head", "gate_b_initialization_authenticated"),
        ("initialization_image", "gate_b_initialization_authenticated"),
        ("initialization_result_hash", "gate_b_initialization_authenticated"),
        ("initialization_bundle_hash", "gate_b_initialization_authenticated"),
        ("compiler_topology", "gate_b_initialization_authenticated"),
        ("source_start", "gate_b_initialization_authenticated"),
        ("environment", "gate_b_initialization_authenticated"),
        ("schedule", "cycle_catalog_and_schedule_complete"),
        ("chunk_digest", "cycle_catalog_and_schedule_complete"),
        ("chunk_manifest", "cycle_catalog_and_schedule_complete"),
        ("validation_digest", "cycle_catalog_and_schedule_complete"),
        ("validation_cherry_pick", "cycle_catalog_and_schedule_complete"),
        ("target_contract", "checkpoint_targets_and_controls_complete"),
        ("controls", "checkpoint_targets_and_controls_complete"),
        ("learner", "training_complete_and_finite"),
        ("training", "training_complete_and_finite"),
        ("runtime_compiler", "training_complete_and_finite"),
        ("source_end", "training_complete_and_finite"),
        ("evaluation", "evaluation_complete_and_finite"),
        ("matching_depth", "matching_depth_above_chance_at_every_effort"),
        ("improvement", "at_least_two_depths_improve_over_h0"),
        ("intact_shuffled", "intact_exceeds_shuffled_at_every_effort"),
        ("controls_chance", "controls_not_demonstrably_above_chance"),
        ("h0_chance", "h0_one_step_above_chance"),
        ("held_out", "held_out_invariants_complete"),
    ],
)
def test_gate_b_qualification_fails_closed_for_mutated_evidence(
    mutation: str,
    criterion: str,
) -> None:
    report = copy.deepcopy(_passing_depth_report())
    if mutation == "schema_control":
        report["schema_version"] = True
    elif mutation == "config":
        report["config"]["learning_rate"] = 0.01
    elif mutation == "config_bool":
        report["config"]["memory_decay"] = True
    elif mutation == "gate_a":
        report["prerequisites"]["gate_a"]["result_sha256"] = "0" * 64
    elif mutation == "source_files":
        report["source_files"]["latent_workspace_model.py"] = "0" * 64
    elif mutation == "initialization":
        wrapper = report["prerequisites"]["gate_b_initialization"]
        wrapper["admission"]["initialization"]["parameter_sha256"] = "0" * 64
        _refresh_gate_b_initialization_wrapper_hashes(wrapper)
    elif mutation == "initialization_raw":
        wrapper = report["prerequisites"]["gate_b_initialization"]
        report["prerequisites"]["gate_b_initialization"] = wrapper["admission"][
            "initialization"
        ]
    elif mutation == "initialization_unwrapped_admission":
        wrapper = report["prerequisites"]["gate_b_initialization"]
        report["prerequisites"]["gate_b_initialization"] = wrapper["admission"]
    elif mutation == "initialization_parameters_finite":
        wrapper = report["prerequisites"]["gate_b_initialization"]
        del wrapper["admission"]["initialization"]["parameters_finite"]
        _refresh_gate_b_initialization_wrapper_hashes(wrapper)
    elif mutation == "initialization_qualification":
        wrapper = report["prerequisites"]["gate_b_initialization"]
        wrapper["admission"]["qualification"]["criteria"][
            "compiler_paths_complete"
        ] = False
        _refresh_gate_b_initialization_wrapper_hashes(wrapper)
    elif mutation == "initialization_source_head":
        wrapper = report["prerequisites"]["gate_b_initialization"]
        wrapper["source_head"] = "b" * 40
        _refresh_gate_b_initialization_wrapper_hashes(wrapper)
    elif mutation == "initialization_image":
        report["prerequisites"]["gate_b_initialization"]["image_digest"] = (
            "sha256:" + "c" * 64
        )
    elif mutation == "initialization_result_hash":
        wrapper = report["prerequisites"]["gate_b_initialization"]
        wrapper["result_sha256"] = "0" * 64
        wrapper["bundle_sha256"] = hashlib.sha256(
            (
                "example21-launch-bundle-v1\0gate_b_init\0"
                f"{wrapper['source_head']}\0{wrapper['preflight_sha256']}\0"
                f"{wrapper['result_sha256']}"
            ).encode()
        ).hexdigest()
    elif mutation == "initialization_bundle_hash":
        report["prerequisites"]["gate_b_initialization"]["bundle_sha256"] = (
            "0" * 64
        )
    elif mutation == "compiler_topology":
        wrapper = report["prerequisites"]["gate_b_initialization"]
        compiler = wrapper["admission"]["initialization"]["compiler"]
        compiler["direct_path_evidence"]["memory_write_scale"][0]["hidden_groups"] = [
            {"index": 3, "hidden_paths": ["reasoning_query"]}
        ]
        _refresh_gate_b_initialization_wrapper_hashes(wrapper)
    elif mutation == "source_start":
        report["source_start"].update(dirty=True, verified=False)
    elif mutation == "environment":
        report["environment"]["backend"] = "cpu"
    elif mutation == "schedule":
        report["data"]["schedule"]["global_sha256"]["events"] = "0" * 64
    elif mutation == "chunk_digest":
        report["data"]["schedule"]["chunk_sha256"]["events"][7] = "0" * 64
    elif mutation == "chunk_manifest":
        report["data"]["schedule"]["chunk_sha256_manifest"]["events"] = "0" * 64
    elif mutation == "validation_digest":
        report["data"]["validation"]["sha256"]["targets_by_depth"] = "0" * 64
    elif mutation == "validation_cherry_pick":
        report["data"]["validation"]["sha256"]["intact"] = report["data"][
            "validation"
        ]["sha256"]["shuffled"]
    elif mutation == "target_contract":
        report["data"]["target_contract"]["supervision_weights"]["8"] = 0.2
    elif mutation == "controls":
        report["data"]["controls"][
            "all_shuffled_answers_differ_at_qualifying_depths"
        ] = False
    elif mutation == "learner":
        report["learner"]["algorithm"] = "bptt"
    elif mutation == "training":
        report["training"]["losses"][0] = True
    elif mutation == "runtime_compiler":
        report["training"]["compiler"]["direct_path_evidence"][
            "memory_write_scale"
        ][0]["hidden_groups"] = [
            {"index": 3, "hidden_paths": ["reasoning_query"]}
        ]
    elif mutation == "source_end":
        report["source_end"]["commit"] = "b" * 40
        report["source_end"]["asserted_commit"] = "b" * 40
    elif mutation == "evaluation":
        report["evaluation"]["depths"]["3"]["intact"]["accuracy"] = True
    elif mutation == "matching_depth":
        metric = _depth_metric(64, checkpoint=8, prediction_label="intact-8")
        report["evaluation"]["depths"]["8"]["intact"] = metric
        report["evaluation"]["efforts"]["8"]["intact"] = metric
        report["evaluation"]["efforts"]["8"]["intact_minus_h0"] = (
            metric["accuracy"]
            - report["evaluation"]["efforts"]["8"]["h0_final_target"]["accuracy"]
        )
        report["evaluation"]["efforts"]["8"]["intact_minus_shuffled"] = 0.0
    elif mutation == "improvement":
        for effort in (2, 4, 8):
            evidence = report["evaluation"]["efforts"][str(effort)]
            h0_final = _depth_metric(
                350,
                checkpoint=0,
                prediction_label="intact-0",
            )
            evidence["h0_final_target"] = h0_final
            evidence["intact_minus_h0"] = (
                evidence["intact"]["accuracy"] - h0_final["accuracy"]
            )
    elif mutation == "intact_shuffled":
        metric = _depth_metric(350, checkpoint=4, prediction_label="shuffled-4")
        report["evaluation"]["depths"]["4"]["shuffled"] = metric
        report["evaluation"]["efforts"]["4"]["shuffled"] = metric
        report["evaluation"]["efforts"]["4"]["intact_minus_shuffled"] = (
            report["evaluation"]["efforts"]["4"]["intact"]["accuracy"]
            - metric["accuracy"]
        )
    elif mutation == "controls_chance":
        metric = _depth_metric(100, checkpoint=2, prediction_label="no-context-2")
        report["evaluation"]["depths"]["2"]["no_context"] = metric
        report["evaluation"]["efforts"]["2"]["no_context"] = metric
    elif mutation == "h0_chance":
        metric = _depth_metric(64, checkpoint=0, prediction_label="intact-0")
        report["evaluation"]["h0_proper"] = metric
        report["evaluation"]["depths"]["0"]["intact"] = metric
    elif mutation == "held_out":
        report["data"]["held_out_invariants"]["cross_effort_h0_prediction_identity"] = (
            False
        )

    qualification = depth._qualification_report(
        report,
        config=depth.DepthGateConfig(),
    )

    assert qualification["passed"] is False
    assert qualification["criteria"][criterion] is False
    assert qualification["interpretation"] == (
        "gate_b_failed_stop_no_capability_conclusion"
    )


def test_gate_b_qualification_labels_abbreviated_regime_without_capability() -> None:
    config = _reduced_depth_config()
    qualification = depth._qualification_report(
        _passing_depth_report(),
        config=config,
    )

    assert qualification["passed"] is False
    assert qualification["interpretation"] == (
        "nonqualifying_abbreviated_no_capability_conclusion"
    )


def _depth_cli_host_argv(
    target: str,
) -> tuple[launcher.LaunchConfig, launcher.TargetPaths, list[str]]:
    repo_root = Path(depth.__file__).resolve().parents[2]
    config = launcher.LaunchConfig(
        target=target,
        repo_root=repo_root,
        output_dir=repo_root / "var" / "example21-depth-gate",
    )
    head = "a" * 40
    paths = launcher.target_paths(config, head, target)
    gate_a = launcher._gate_a_artifact_paths(config)
    argv = [
        "--target",
        target,
        "--gate-a-result",
        str(gate_a.result),
        "--gate-a-manifest",
        str(gate_a.manifest),
    ]
    if target == "formal_gate_b":
        argv.extend(
            [
                "--gate-b-init-manifest",
                str(launcher.target_paths(config, head, "gate_b_init").manifest),
            ]
        )
    argv.extend(["--output", str(paths.result)])
    return config, paths, argv


def test_depth_artifact_writer_is_atomic_deterministic_and_strict(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "nested" / "depth.json"
    value = {"z": 1, "a": [True, None]}

    written = depth.write_artifact(value, destination)

    expected = (
        msgspec_json.dumps(
            value,
            allow_nan=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n"
    )
    assert written == destination
    assert destination.read_text(encoding="utf-8") == expected
    assert not destination.with_suffix(".json.tmp").exists()

    invalid = tmp_path / "nonfinite.json"
    with pytest.raises(ValueError, match="JSON|range|compliant"):
        depth.write_artifact({"loss": math.nan}, invalid)
    assert not invalid.exists()
    assert not invalid.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize("target", ["gate_b_init", "formal_gate_b"])
def test_depth_parser_accepts_exact_launcher_module_argv(target: str) -> None:
    config, paths, _ = _depth_cli_host_argv(target)
    command = launcher.gate_command(
        config,
        image_id="sha256:" + "b" * 64,
        head="a" * 40,
        paths=paths,
        git_dir_in_container="/git-common/worktrees/depth",
        admission_manifests=None,
    )
    python_index = command.index("python")
    assert command[python_index : python_index + 3] == [
        "python",
        "-m",
        "examples.pp_prop.latent_workspace_depth_gate",
    ]
    argv = command[python_index + 3 :]

    parsed = depth._parser().parse_args(argv)

    assert set(vars(parsed)) == {
        "target",
        "gate_a_result",
        "gate_a_manifest",
        "gate_b_init_manifest",
        "output",
    }
    assert parsed.target == target
    assert parsed.output == Path(str(paths.container_result))
    assert parsed.gate_b_init_manifest is None if target == "gate_b_init" else (
        parsed.gate_b_init_manifest is not None
    )


def test_depth_parser_exposes_no_topology_or_budget_overrides() -> None:
    _, _, argv = _depth_cli_host_argv("gate_b_init")

    with pytest.raises(SystemExit) as error:
        depth._parser().parse_args([*argv, "--neuron-count", "64"])

    assert error.value.code == 2


@pytest.mark.parametrize(
    "mutation",
    ["wrong_gate_a_path", "init_with_init_manifest", "formal_without_init_manifest"],
)
def test_depth_cli_rejects_nonfixed_or_target_incompatible_paths(
    mutation: str,
) -> None:
    target = "formal_gate_b" if mutation == "formal_without_init_manifest" else (
        "gate_b_init"
    )
    _, paths, argv = _depth_cli_host_argv(target)
    if mutation == "wrong_gate_a_path":
        argv[argv.index("--gate-a-result") + 1] = str(paths.result)
    elif mutation == "init_with_init_manifest":
        argv.extend(["--gate-b-init-manifest", str(paths.manifest)])
    else:
        index = argv.index("--gate-b-init-manifest")
        del argv[index : index + 2]

    with pytest.raises(ValueError, match="fixed|manifest|target"):
        depth.main(argv)


def test_gate_b_init_cli_loads_gate_a_and_emits_full_admission(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, paths, argv = _depth_cli_host_argv("gate_b_init")
    passing = _passing_depth_report()
    admission = copy.deepcopy(
        passing["prerequisites"]["gate_b_initialization"]["admission"]
    )
    gate_a = copy.deepcopy(passing["prerequisites"]["gate_a"])
    source_reports = iter([passing["source_start"], passing["source_end"]])
    events: list[str] = []
    captured: dict[str, Any] = {}
    model = object()

    def source_report() -> dict[str, Any]:
        value = copy.deepcopy(next(source_reports))
        events.append("source_start" if not events else "source_end")
        return value

    def environment_report() -> dict[str, Any]:
        events.append("environment")
        return copy.deepcopy(passing["environment"])

    def require_launch(source: Mapping[str, Any], environment: Mapping[str, Any]) -> None:
        events.append("gpu_authenticated")
        assert source == passing["source_start"]
        assert environment == passing["environment"]

    def load_gate_a(actual: launcher.LaunchConfig) -> dict[str, Any]:
        events.append("gate_a_loaded")
        assert actual.target == "gate_b_init"
        assert actual.repo_root == config.repo_root
        assert actual.output_dir == config.output_dir
        return copy.deepcopy(gate_a)

    def make_model(model_config: Any) -> object:
        del model_config
        events.append("fresh_model")
        return model

    def initialization_report(
        actual_model: object,
        actual_config: depth.DepthGateConfig,
        **kwargs: Any,
    ) -> dict[str, Any]:
        events.append("initialization_report")
        captured.update(model=actual_model, config=actual_config, **kwargs)
        reporter = kwargs["source_end_reporter"]
        assert callable(reporter)
        assert reporter() == passing["source_end"]
        return copy.deepcopy(admission)

    def write_artifact(value: dict[str, Any], output: Path) -> Path:
        events.append("artifact_written")
        captured.update(value=value, output=output)
        return output

    monkeypatch.setattr(depth.gate, "_source_report", source_report)
    monkeypatch.setattr(
        depth,
        "_source_files_report",
        lambda actual_config: copy.deepcopy(passing["source_files"]),
    )
    monkeypatch.setattr(depth.gate, "_environment_report", environment_report)
    monkeypatch.setattr(depth.gate, "_require_authenticated_gpu_launch", require_launch)
    monkeypatch.setattr(launcher, "_load_gate_a_prerequisite", load_gate_a)
    monkeypatch.setattr(depth, "LatentWorkspaceModel", make_model)
    monkeypatch.setattr(depth, "_gate_b_initialization_report", initialization_report)
    monkeypatch.setattr(depth, "write_artifact", write_artifact, raising=False)

    assert depth.main(argv) == 0

    assert events == [
        "source_start",
        "environment",
        "gpu_authenticated",
        "gate_a_loaded",
        "fresh_model",
        "initialization_report",
        "source_end",
        "artifact_written",
    ]
    assert captured["model"] is model
    assert captured["config"] == depth.DepthGateConfig()
    assert captured["gate_a"] == gate_a
    assert captured["source_start"] == passing["source_start"]
    assert callable(captured["source_end_reporter"])
    assert captured["source_files"] == passing["source_files"]
    assert captured["environment"] == passing["environment"]
    assert captured["value"] == admission
    assert captured["output"] == paths.result
    stdout = capsys.readouterr().out
    assert str(paths.result) in stdout
    assert '"passed": true' in stdout


def test_formal_gate_b_cli_loads_both_prerequisites_and_passes_postflight_reporter(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, paths, argv = _depth_cli_host_argv("formal_gate_b")
    passing = _passing_depth_report()
    gate_a = copy.deepcopy(passing["prerequisites"]["gate_a"])
    init_bundle = copy.deepcopy(passing["prerequisites"]["gate_b_initialization"])
    source_reports = iter([passing["source_start"], passing["source_end"]])
    events: list[str] = []
    captured: dict[str, Any] = {}

    def source_report() -> dict[str, Any]:
        value = copy.deepcopy(next(source_reports))
        events.append("source_start" if not events else "source_end")
        return value

    def environment_report() -> dict[str, Any]:
        events.append("environment")
        return copy.deepcopy(passing["environment"])

    def require_launch(source: Mapping[str, Any], environment: Mapping[str, Any]) -> None:
        del source, environment
        events.append("gpu_authenticated")

    def load_gate_a(actual: launcher.LaunchConfig) -> dict[str, Any]:
        events.append("gate_a_loaded")
        assert actual.target == "formal_gate_b"
        return copy.deepcopy(gate_a)

    def load_init(
        actual: launcher.LaunchConfig,
        *,
        head: str,
        image_id: str,
    ) -> dict[str, Any]:
        events.append("initialization_loaded")
        assert actual.target == "formal_gate_b"
        assert head == passing["source_start"]["commit"]
        assert image_id == passing["environment"]["image_digest"]
        return copy.deepcopy(init_bundle)

    def run_gate(
        actual_config: depth.DepthGateConfig,
        **kwargs: Any,
    ) -> dict[str, Any]:
        events.append("core_run")
        captured.update(config=actual_config, **kwargs)
        reporter = kwargs["source_end_reporter"]
        assert callable(reporter)
        source_end = reporter()
        result = copy.deepcopy(passing)
        result["prerequisites"] = copy.deepcopy(kwargs["prerequisites"])
        result["source_start"] = copy.deepcopy(kwargs["source_start"])
        result["source_end"] = copy.deepcopy(source_end)
        result["qualification"] = {
            "passed": False,
            "criteria": {},
            "interpretation": "gate_b_failed_stop_no_capability_conclusion",
        }
        return result

    def write_artifact(value: dict[str, Any], output: Path) -> Path:
        events.append("artifact_written")
        captured.update(value=value, output=output)
        return output

    monkeypatch.setattr(depth.gate, "_source_report", source_report)
    monkeypatch.setattr(
        depth,
        "_source_files_report",
        lambda actual_config: copy.deepcopy(passing["source_files"]),
    )
    monkeypatch.setattr(depth.gate, "_environment_report", environment_report)
    monkeypatch.setattr(depth.gate, "_require_authenticated_gpu_launch", require_launch)
    monkeypatch.setattr(launcher, "_load_gate_a_prerequisite", load_gate_a)
    monkeypatch.setattr(launcher, "_load_gate_b_init_manifest", load_init)
    monkeypatch.setattr(depth, "run_depth_gate", run_gate)
    monkeypatch.setattr(depth, "write_artifact", write_artifact, raising=False)

    assert depth.main(argv) == 0

    assert events == [
        "source_start",
        "environment",
        "gpu_authenticated",
        "gate_a_loaded",
        "initialization_loaded",
        "core_run",
        "source_end",
        "artifact_written",
    ]
    assert captured["config"] == depth.DepthGateConfig()
    assert captured["prerequisites"] == {
        "gate_a": gate_a,
        "gate_b_initialization": init_bundle,
    }
    assert captured["source_start"] == passing["source_start"]
    assert captured["source_files"] == passing["source_files"]
    assert captured["environment"] == passing["environment"]
    assert captured["output"] == paths.result
    assert captured["value"]["qualification"]["passed"] is False
    stdout = capsys.readouterr().out
    assert str(paths.result) in stdout
    assert '"passed": false' in stdout


def test_depth_cli_propagates_duplicate_json_authentication_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, argv = _depth_cli_host_argv("gate_b_init")
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"gate": 1, "gate": 2}\n', encoding="utf-8")
    passing = _passing_depth_report()

    monkeypatch.setattr(
        depth.gate,
        "_source_report",
        lambda: copy.deepcopy(passing["source_start"]),
    )
    monkeypatch.setattr(
        depth.gate,
        "_environment_report",
        lambda: copy.deepcopy(passing["environment"]),
    )
    monkeypatch.setattr(
        depth.gate,
        "_require_authenticated_gpu_launch",
        lambda *args: None,
    )
    monkeypatch.setattr(
        launcher,
        "_load_gate_a_prerequisite",
        lambda config: launcher.load_strict_json(duplicate),
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        depth.main(argv)
