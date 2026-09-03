# Copied from TRELLIS.2 (https://github.com/microsoft/TRELLIS.2/blob/main/trellis2/modules/sparse/conv/config.py)
# Copyright (c) Microsoft Corporation. Licensed under the MIT License.

SPCONV_ALGO = 'auto'                                # 'auto', 'implicit_gemm', 'native'
FLEX_GEMM_ALGO = 'masked_implicit_gemm_splitk'      # 'explicit_gemm', 'implicit_gemm', 'implicit_gemm_splitk', 'masked_implicit_gemm', 'masked_implicit_gemm_splitk'
FLEX_GEMM_HASHMAP_RATIO = 2.0                       # Ratio of hashmap size to input size
