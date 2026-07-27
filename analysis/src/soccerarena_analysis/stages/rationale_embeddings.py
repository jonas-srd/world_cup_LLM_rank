from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from ..config import AnalysisConfig, canonical_json, sha256_bytes, sha256_file
from ..manifest import Manifest
from ..reporting.figures import apply_style, model_color, save_figure
from .annotations import annotation_corpus

MODEL_PROVIDER_ICONS = {
    "anthropic/claude-opus-4.8": "anthropic.png",
    "deepseek/deepseek-v4-pro": "deepseek-color.png",
    "google/gemini-3.1-pro-preview": "google-color.png",
    "x-ai/grok-4.3": "xai.png",
    "openai/gpt-5.5": "openai.png",
    "mistralai/mistral-large-2512": "mistral-color.png",
    "qwen/qwen3.7-max": "alibaba-color.png",
}


def _add_model_icon_legend(
    axis: plt.Axes,
    models: list[str],
    config: AnalysisConfig,
    tsne: dict[str, object],
) -> None:
    icon_directory = Path(__file__).resolve().parents[3] / "assets" / "brand_icons"
    icon_paths = {model: icon_directory / MODEL_PROVIDER_ICONS[model] for model in models}
    missing_icons = [str(path) for path in icon_paths.values() if not path.is_file()]
    if missing_icons:
        raise FileNotFoundError(f"Missing model provider icons: {missing_icons}")

    x_positions = np.linspace(
        float(tsne["icon_legend_x_min"]),
        float(tsne["icon_legend_x_max"]),
        len(models),
    )
    line_y = float(tsne["icon_legend_line_y"])
    line_half_width = float(tsne["icon_legend_line_half_width"])
    for x_position, model in zip(x_positions, models):
        axis.add_artist(
            Line2D(
                [x_position - line_half_width, x_position + line_half_width],
                [line_y, line_y],
                transform=axis.transAxes,
                color=model_color(config, model),
                linewidth=float(tsne["icon_legend_line_width"]),
                solid_capstyle="round",
                clip_on=False,
            )
        )
        axis.add_artist(
            AnnotationBbox(
                OffsetImage(
                    plt.imread(icon_paths[model]),
                    zoom=float(tsne["icon_legend_zoom"]),
                    interpolation="lanczos",
                ),
                (x_position, float(tsne["icon_legend_y"])),
                xycoords=axis.transAxes,
                box_alignment=(0.5, 0.5),
                frameon=False,
                pad=0.0,
                annotation_clip=False,
            )
        )


def _corpus_identity(corpus: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    identity = corpus[
        ["prediction_id", "model_id", "access_condition", "match_id", "rationale_text"]
    ].copy()
    identity["annotation_id"] = identity["prediction_id"].map(
        lambda value: sha256_bytes(f"{config.master_seed}|{value}".encode())[:20]
    )
    identity["rationale_hash"] = identity["rationale_text"].map(
        lambda value: sha256_bytes(value.encode("utf-8"))
    )
    identity["embedding_row"] = np.arange(len(identity), dtype=np.int64)
    return identity


def _embedding_metadata(
    identity: pd.DataFrame, config: AnalysisConfig, dimension: int | None = None
) -> dict[str, object]:
    settings = config.section("annotation")["embedding"]
    corpus_records = identity[["annotation_id", "rationale_hash"]].to_dict("records")
    return {
        "model": settings["model"],
        "revision": settings["revision"],
        "normalize": bool(settings["normalize"]),
        "corpus_hash": sha256_bytes(canonical_json(corpus_records).encode("utf-8")),
        "rows": len(identity),
        "dimension": dimension,
    }


def _load_or_encode(
    corpus: pd.DataFrame, config: AnalysisConfig, directory: Path
) -> tuple[np.ndarray, pd.DataFrame, Path, Path, Path]:
    settings = config.section("annotation")["embedding"]
    identity = _corpus_identity(corpus, config)
    vector_path = directory / "rationale_embeddings.npy"
    index_path = directory / "rationale_embeddings_index.parquet"
    metadata_path = directory / "rationale_embeddings_metadata.json"
    expected = _embedding_metadata(identity, config)
    if vector_path.exists() and index_path.exists() and metadata_path.exists():
        frozen = json.loads(metadata_path.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in frozen.items() if key != "dimension"}
        expected_comparable = {key: value for key, value in expected.items() if key != "dimension"}
        if comparable != expected_comparable:
            raise ValueError("Frozen rationale embedding cache does not match the scoped corpus")
        cached_index = pd.read_parquet(index_path)
        if not cached_index.equals(identity):
            raise ValueError("Frozen rationale embedding index does not match the scoped corpus")
        vectors = np.load(vector_path, allow_pickle=False)
        if vectors.shape != (len(identity), int(frozen["dimension"])):
            raise ValueError("Frozen rationale embedding array has an invalid shape")
        return vectors, identity, vector_path, index_path, metadata_path

    from sentence_transformers import SentenceTransformer

    runtime_cache = config.resolve_path("artifacts") / "runtime" / "huggingface"
    runtime_cache.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(
        str(settings["model"]),
        revision=str(settings["revision"]),
        cache_folder=str(runtime_cache),
    )
    vectors = model.encode(
        corpus["rationale_text"].tolist(),
        batch_size=int(settings["batch_size"]),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=bool(settings["normalize"]),
    ).astype(np.float32)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(vector_path, vectors, allow_pickle=False)
    identity.to_parquet(index_path, index=False)
    metadata = _embedding_metadata(identity, config, dimension=int(vectors.shape[1]))
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return vectors, identity, vector_path, index_path, metadata_path


def _coordinates(
    vectors: np.ndarray, identity: pd.DataFrame, config: AnalysisConfig
) -> pd.DataFrame:
    from sklearn.manifold import TSNE

    settings = config.section("annotation")["embedding"]["tsne"]
    projection = TSNE(
        n_components=2,
        perplexity=float(settings["perplexity"]),
        learning_rate=settings["learning_rate"],
        init=str(settings["initialization"]),
        metric=str(settings["metric"]),
        max_iter=int(settings["max_iterations"]),
        random_state=config.derived_seed("rq5.tsne"),
    ).fit_transform(vectors)
    return identity.drop(columns="rationale_text").assign(
        tsne_1=projection[:, 0], tsne_2=projection[:, 1]
    )


def _plot(
    coordinates: pd.DataFrame,
    config: AnalysisConfig,
    manifest: Manifest,
    source_hashes: dict[str, str],
) -> None:
    annotation = config.section("annotation")
    tsne = annotation["embedding"]["tsne"]
    reporting = config.section("reporting")
    access_markers = dict(
        zip(config.section("design")["access_conditions"], annotation["access_markers"])
    )
    apply_style(config)
    figure, axis = plt.subplots(
        figsize=(float(reporting["figure_width_double"]), float(tsne["figure_height"]))
    )
    models = list(config.section("design")["complete_panel"])
    for model_id in models:
        for access in config.section("design")["access_conditions"]:
            subset = coordinates[
                coordinates["model_id"].eq(model_id) & coordinates["access_condition"].eq(access)
            ]
            axis.scatter(
                subset["tsne_1"],
                subset["tsne_2"],
                s=float(tsne["point_size"]),
                alpha=float(tsne["point_alpha"]),
                marker=access_markers[access],
                color=model_color(config, model_id),
                linewidths=0,
                rasterized=True,
            )
    for access in config.section("design")["access_conditions"]:
        label = reporting["labels"]["access"][access]
        axis.scatter(
            [],
            [],
            color=reporting["palette"]["neutral"],
            marker=access_markers[access],
            label=label,
        )
    axis.tick_params(
        axis="both",
        labelsize=float(tsne["tick_label_size"]),
    )
    axis.set_xlabel(
        "t-SNE dimension 1 (arbitrary units)",
        fontsize=float(tsne["axis_label_size"]),
    )
    axis.set_ylabel(
        "t-SNE dimension 2 (arbitrary units)",
        fontsize=float(tsne["axis_label_size"]),
    )
    axis.set_title("")
    _add_model_icon_legend(axis, models, config, tsne)
    axis.legend(
        loc="lower right",
        bbox_to_anchor=(float(tsne["legend_anchor_x"]), float(tsne["legend_anchor_y"])),
        ncol=int(tsne["legend_columns"]),
        markerscale=float(tsne["legend_marker_scale"]),
        fontsize=float(tsne["legend_font_size"]),
    )
    axis.grid(False)
    figure.subplots_adjust(**tsne["margins"])
    save_figure(
        figure,
        config,
        manifest,
        "fig_rq5_rationale_tsne_appendix",
        "rq5",
        source_hashes,
    )


def run(config: AnalysisConfig, manifest: Manifest) -> pd.DataFrame:
    corpus_record = manifest.require("derived_rationale_corpus")
    corpus = annotation_corpus(pd.read_parquet(corpus_record.path), config)
    directory = config.resolve_path("annotations")
    vectors, identity, vector_path, index_path, metadata_path = _load_or_encode(
        corpus, config, directory
    )
    sources = {"rationale_corpus": corpus_record.sha256}
    manifest.add(
        "rationale_embedding_vectors",
        vector_path,
        "npy",
        "rq5",
        sources,
        {"rows": len(identity), "dimensions": int(vectors.shape[1])},
    )
    manifest.add(
        "rationale_embedding_index", index_path, "parquet", "rq5", sources, {"rows": len(identity)}
    )
    manifest.add("rationale_embedding_metadata", metadata_path, "json", "rq5", sources)
    coordinates = _coordinates(vectors, identity, config)
    coordinate_path = directory / "rationale_tsne_coordinates.parquet"
    coordinates.to_parquet(coordinate_path, index=False)
    coordinate_record = manifest.add(
        "rq5_tsne_coordinates",
        coordinate_path,
        "parquet",
        "rq5",
        {
            "embedding_vectors": sha256_file(vector_path),
            "embedding_index": sha256_file(index_path),
        },
        {"rows": len(coordinates), "interpretation": "exploratory visualization only"},
    )
    manifest.write()
    _plot(coordinates, config, manifest, {"tsne_coordinates": coordinate_record.sha256})
    return coordinates
