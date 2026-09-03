# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

from .asset_attention_mask import (
    create_overall_spatial_mask_sparse,
    # Per-part cross-attention masks for ERP OmniPart
    create_per_part_cross_attn_masks,
    # Visibility filtering
    is_point_inside_obb,
    calculate_asset_visibility,
    filter_visible_assets,
    # 3D bbox overlap detection
    obb_to_corners,
    check_obb_overlap_sat,
    compute_overlap_groups,
    create_intra_asset_attention_mask,
)

# from .asset_attention_mask import (
#     create_asset_cross_attention_mask,
#     create_overall_spatial_attention_mask,
#     create_overall_spatial_mask_sparse,
#     create_combined_attention_masks,
#     create_batch_combined_attention_masks,
#     # Per-part cross-attention masks for ERP OmniPart
#     create_per_part_cross_attn_masks,
#     # Visibility filtering
#     is_point_inside_obb,
#     calculate_asset_visibility,
#     filter_visible_assets,
#     # 3D bbox overlap detection
#     obb_to_corners,
#     check_obb_overlap_sat,
#     compute_overlap_groups,
#     create_intra_asset_attention_mask,
# )
