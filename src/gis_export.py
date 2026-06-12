#!/usr/bin/env python3
"""
src/gis_export.py — GIS export module for AeroSeg.

Converts segmentation masks to GeoJSON polygons with per-class area
measurements. Supports GeoJSON and Shapefile export formats.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
import rasterio.transform
from shapely.geometry import shape as shapely_shape


def mask_to_geojson(
    mask: np.ndarray,
    config: Dict,
    transform: Optional[rasterio.transform.Affine] = None,
    crs: str = "EPSG:4326",
) -> str:
    """Convert a segmentation mask to a GeoJSON FeatureCollection string.

    Polygonizes each class in the mask, filters small polygons, simplifies
    geometries, and returns a GeoJSON string.

    Args:
        mask: 2D array of class indices (H×W, int).
        config: Configuration dictionary with 'gis', 'inference', 'data',
                and 'class_names' sections.
        transform: Optional rasterio Affine transform. If None, a dummy
                   transform is created using default_resolution_m.
        crs: Coordinate reference system string (default: EPSG:4326).

    Returns:
        GeoJSON FeatureCollection string.
    """
    num_classes = config["data"]["num_classes"]
    class_names = config["class_names"]
    resolution = config["gis"]["default_resolution_m"]
    min_area_px = config["inference"]["min_polygon_area_px"]
    simplify_tol = config["gis"]["simplify_tolerance"]

    # Create dummy affine transform if not provided
    if transform is None:
        h, w = mask.shape
        transform = rasterio.transform.from_bounds(
            0, 0, w * resolution, h * resolution, w, h
        )

    min_area_m2 = min_area_px * resolution * resolution

    # Collect features for all classes
    records: List[Dict] = []

    for class_idx in range(num_classes):
        # Create binary mask for this class
        binary_mask = (mask == class_idx).astype(np.uint8)

        # Skip if class not present
        if not np.any(binary_mask):
            continue

        try:
            # Polygonize
            shapes = list(
                rasterio.features.shapes(
                    binary_mask, transform=transform
                )
            )

            for geom_dict, value in shapes:
                if value != 1:
                    continue

                geom = shapely_shape(geom_dict)

                # Filter small polygons
                if geom.area < min_area_m2:
                    continue

                # Simplify geometry
                geom = geom.simplify(simplify_tol)

                if geom.is_empty:
                    continue

                records.append({
                    "class_id": int(class_idx),
                    "class_name": class_names[class_idx],
                    "area_m2": round(geom.area, 2),
                    "geometry": geom,
                })

        except Exception as e:
            print(f"Warning: Error polygonizing class {class_idx} "
                  f"({class_names[class_idx]}): {e}")
            continue

    # Build GeoDataFrame
    if records:
        gdf = gpd.GeoDataFrame(records, geometry="geometry")
        gdf = gdf.set_crs(crs)
    else:
        # Return empty GeoDataFrame with correct schema
        gdf = gpd.GeoDataFrame(
            columns=["class_id", "class_name", "area_m2", "geometry"],
            geometry="geometry",
        )
        gdf = gdf.set_crs(crs)

    return gdf.to_json()


def mask_to_geodataframe(
    mask: np.ndarray,
    config: Dict,
    transform: Optional[rasterio.transform.Affine] = None,
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Convert a segmentation mask to a GeoDataFrame.

    Args:
        mask: 2D array of class indices (H×W, int).
        config: Configuration dictionary.
        transform: Optional rasterio Affine transform.
        crs: Coordinate reference system string.

    Returns:
        GeoDataFrame with class_id, class_name, area_m2, geometry columns.
    """
    geojson_str = mask_to_geojson(mask, config, transform, crs)
    gdf = gpd.read_file(geojson_str, driver="GeoJSON")
    return gdf


def export_geojson(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """Export a GeoDataFrame as a GeoJSON file.

    Args:
        gdf: GeoDataFrame to export.
        output_path: Path for the output .geojson file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        gdf.to_file(str(output_path), driver="GeoJSON")
        print(f"Exported GeoJSON: {output_path}")
    except Exception as e:
        raise RuntimeError(f"Error exporting GeoJSON to {output_path}: {e}")


def export_shapefile(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """Export a GeoDataFrame as a Shapefile.

    Truncates column names to ≤10 characters for Shapefile compatibility.

    Args:
        gdf: GeoDataFrame to export.
        output_path: Path for the output .shp file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Truncate column names for Shapefile compatibility (max 10 chars)
        gdf_copy = gdf.copy()
        rename_map = {}
        for col in gdf_copy.columns:
            if col != "geometry" and len(col) > 10:
                rename_map[col] = col[:10]
        if rename_map:
            gdf_copy = gdf_copy.rename(columns=rename_map)

        gdf_copy.to_file(str(output_path), driver="ESRI Shapefile")
        print(f"Exported Shapefile: {output_path}")
    except Exception as e:
        raise RuntimeError(f"Error exporting Shapefile to {output_path}: {e}")


def compute_area_stats(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Compute per-class area statistics from a GeoDataFrame.

    Args:
        gdf: GeoDataFrame with 'class_name' and 'area_m2' columns.

    Returns:
        DataFrame with class_name, total_area_m2, polygon_count, pct_of_total.
    """
    if gdf.empty or "class_name" not in gdf.columns:
        return pd.DataFrame(
            columns=["class_name", "total_area_m2", "polygon_count", "pct_of_total"]
        )

    stats = gdf.groupby("class_name").agg(
        total_area_m2=("area_m2", "sum"),
        polygon_count=("area_m2", "count"),
    ).reset_index()

    total_area = stats["total_area_m2"].sum()
    if total_area > 0:
        stats["pct_of_total"] = (stats["total_area_m2"] / total_area * 100).round(2)
    else:
        stats["pct_of_total"] = 0.0

    return stats
