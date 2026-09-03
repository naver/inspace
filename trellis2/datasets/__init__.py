# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license
#
# Modified from TRELLIS.2 (https://github.com/microsoft/TRELLIS.2)
# Copyright (c) Microsoft Corporation. Licensed under the MIT License.

import importlib

__attributes = {
    'FlexiDualGridDataset': 'flexi_dual_grid',
    'SparseVoxelPbrDataset':'sparse_voxel_pbr',

    'SparseStructureLatent': 'sparse_structure_latent',
    'TextConditionedSparseStructureLatent': 'sparse_structure_latent',
    'ImageConditionedSparseStructureLatent': 'sparse_structure_latent',

    'SLat': 'structured_latent',
    'ImageConditionedSLat': 'structured_latent',
    'SLatShape': 'structured_latent_shape',
    'ImageConditionedSLatShape': 'structured_latent_shape',
    'SLatPbr': 'structured_latent_svpbr',
    'ImageConditionedSLatPbr': 'structured_latent_svpbr',

    # ERP datasets for panorama-to-3D scene generation
    'ERPSparseStructureLatent': 'erp_sparse_structure_latent',
    'ERPCubemapConditionedSparseStructureLatent': 'erp_sparse_structure_latent',
    'ERPImageConditionedSparseStructureLatent': 'erp_sparse_structure_latent',

    # ERP bounding box estimation dataset
    'ERPBBoxDataset': 'erp_bbox_estimation',

    # ERP structured latent datasets for asset-aware shape/texture generation
    'ERPStructuredLatentBase': 'erp_structured_latent',
    'ERPStructuredLatentShape': 'erp_structured_latent',
    'ERPStructuredLatentTexture': 'erp_structured_latent',
    'ERPImageConditionedSLatShape': 'erp_structured_latent',
    'ERPImageConditionedSLatTexture': 'erp_structured_latent',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# For Pylance
if __name__ == '__main__':    
    from .flexi_dual_grid import FlexiDualGridDataset
    from .sparse_voxel_pbr import SparseVoxelPbrDataset
    
    from .sparse_structure_latent import SparseStructureLatent, ImageConditionedSparseStructureLatent
    from .structured_latent import SLat, ImageConditionedSLat
    from .structured_latent_shape import SLatShape, ImageConditionedSLatShape
    from .structured_latent_svpbr import SLatPbr, ImageConditionedSLatPbr
    