"""Model Registry & Versioning Management Module.

Persists trained model artifacts, scalers, and metadata in the models/ directory.
Provides tracking, retrieval, active model resolution, and comparison reporting.
"""

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib


MODELS_DIR = Path("d:/StockMind-AI/models")
REGISTRY_FILE = MODELS_DIR / "registry.json"


class ModelRegistry:
    """Manages versioned storage and retrieval of trained machine learning models."""

    def __init__(
        self,
        base_dir: Optional[Union[Path, str]] = None,
        registry_dir: Optional[Union[Path, str]] = None,
    ) -> None:
        chosen_dir = registry_dir if registry_dir is not None else (base_dir if base_dir is not None else MODELS_DIR)
        self.base_dir = Path(chosen_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.base_dir / "registry.json"
        self._ensure_registry_file()

    def save_model(
        self,
        symbol: str,
        model_artifact: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Convenience method to register and save model."""
        meta = metadata or {}
        model_type = meta.get("model_type", "model")
        target_type = meta.get("target_type", "direction")
        metrics = meta.get("metrics", {})
        features = meta.get("features", [])
        m_id = self.register_model(
            model_artifact=model_artifact,
            symbol=symbol,
            model_type=model_type,
            target_type=target_type,
            metrics=metrics,
            features=features,
        )
        registry = self._load_registry()
        return registry.get(m_id, {}).get("version", m_id)

    def load_model(
        self,
        symbol: str,
        version: Optional[str] = None,
    ) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """Load model artifact and metadata by symbol and optional version."""
        registry = self._load_registry()
        symbol = symbol.upper()
        matching = [
            m for m in registry.values()
            if m.get("symbol") == symbol and (version is None or m.get("version") == version)
        ]
        if not matching:
            return None, None
        meta = matching[-1]
        bundle = self.load_model_bundle(meta["model_id"])
        if bundle:
            return bundle.get("model"), meta
        return None, meta

    def get_model_status(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Return readiness and version status of models in registry."""
        registry = self._load_registry()
        if symbol:
            symbol = symbol.upper()
            sym_models = [m for m in registry.values() if m.get("symbol") == symbol]
            is_ready = len(sym_models) > 0
            latest_version = sym_models[-1].get("version", "v0.0") if sym_models else "none"
            return {
                "symbol": symbol,
                "is_ready": is_ready,
                "latest_version": latest_version,
                "total_models": len(sym_models),
            }
        return {
            "is_ready": len(registry) > 0,
            "total_registered_models": len(registry),
            "total_models_registered": len(registry),
            "registered_symbols": list({m.get("symbol") for m in registry.values() if m.get("symbol")}),
            "engine_status": "OPERATIONAL",
        }

    def _ensure_registry_file(self) -> None:
        if not self.registry_file.exists():
            self._save_registry({})

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        try:
            if self.registry_file.exists():
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_registry(self, data: Dict[str, Dict[str, Any]]) -> None:
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def register_model(
        self,
        model_artifact: Any,
        symbol: str,
        model_type: str,
        target_type: str,
        metrics: Dict[str, Any],
        features: List[str],
        scaler: Optional[Any] = None,
        selector: Optional[Any] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        is_active: bool = True,
    ) -> str:
        """Serialize and register a trained model artifact with full metadata."""
        symbol = symbol.upper()
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{symbol.lower()}_{model_type.lower()}_{timestamp_str}"
        artifact_filename = f"{model_id}.joblib"
        artifact_path = self.base_dir / artifact_filename

        # Bundle model, scaler, and selector together
        bundle = {
            "model": model_artifact,
            "scaler": scaler,
            "selector": selector,
            "features": features,
            "symbol": symbol,
            "model_type": model_type,
            "target_type": target_type,
        }
        joblib.dump(bundle, artifact_path, compress=3)

        registry = self._load_registry()

        # If setting this model as active, mark other models of same symbol & target as inactive
        if is_active:
            for m_id, meta in registry.items():
                if meta.get("symbol") == symbol and meta.get("model_type") == model_type and meta.get("target_type") == target_type:
                    meta["is_active"] = False

        metadata = {
            "model_id": model_id,
            "version": f"v1.{len(registry) + 1}.0",
            "symbol": symbol,
            "model_type": model_type,
            "target_type": target_type,
            "created_at": datetime.now().isoformat(),
            "artifact_path": str(artifact_path.relative_to(self.base_dir.parent)),
            "features": features,
            "feature_count": len(features),
            "metrics": metrics,
            "hyperparameters": hyperparameters or {},
            "is_active": is_active,
        }

        registry[model_id] = metadata
        self._save_registry(registry)
        return model_id

    def load_model_bundle(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Load bundled model artifact, scaler, and feature selector."""
        registry = self._load_registry()
        meta = registry.get(model_id)
        if not meta:
            return None

        artifact_rel = meta.get("artifact_path", "")
        artifact_path = Path("d:/StockMind-AI") / artifact_rel
        if not artifact_path.exists():
            artifact_path = self.base_dir / Path(artifact_rel).name
        if not artifact_path.exists():
            return None

        bundle = joblib.load(artifact_path)
        bundle["metadata"] = meta
        return bundle

    def get_active_model(
        self,
        symbol: str,
        target_type: str = "direction",
        preferred_model_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Resolve the active production model for a symbol."""
        registry = self._load_registry()
        symbol = symbol.upper()

        matching = [
            meta for meta in registry.values()
            if meta.get("symbol") == symbol and meta.get("target_type") == target_type
        ]

        if not matching:
            return None

        # Filter by preferred model type if specified
        if preferred_model_type:
            typed_matches = [m for m in matching if m.get("model_type", "").lower() == preferred_model_type.lower()]
            if typed_matches:
                matching = typed_matches

        # Check for explicitly marked active model
        active_candidates = [m for m in matching if m.get("is_active")]
        if active_candidates:
            # Sort by creation date descending
            target_meta = sorted(active_candidates, key=lambda x: x.get("created_at", ""), reverse=True)[0]
        else:
            # Fall back to highest accuracy / lowest RMSE
            target_meta = sorted(matching, key=lambda x: x.get("metrics", {}).get("accuracy", 0.0), reverse=True)[0]

        return self.load_model_bundle(target_meta["model_id"])

    def list_models(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """List registered models with performance metrics, optionally filtered by symbol."""
        registry = self._load_registry()
        models = list(registry.values())
        if symbol:
            models = [m for m in models if m.get("symbol") == symbol.upper()]
        return sorted(models, key=lambda x: x.get("created_at", ""), reverse=True)

    def get_model_comparison(self, symbol: str) -> List[Dict[str, Any]]:
        """Return side-by-side metric comparison across all registered models for a symbol."""
        models = self.list_models(symbol=symbol)
        comparison = []
        for m in models:
            metrics = m.get("metrics", {})
            comparison.append({
                "model_id": m.get("model_id"),
                "model_type": m.get("model_type"),
                "version": m.get("version"),
                "target_type": m.get("target_type"),
                "accuracy": metrics.get("accuracy", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1": metrics.get("f1", 0.0),
                "mae": metrics.get("mae", 0.0),
                "rmse": metrics.get("rmse", 0.0),
                "r2": metrics.get("r2", 0.0),
                "is_active": m.get("is_active", False),
                "created_at": m.get("created_at"),
            })
        return comparison


# Singleton registry instance
model_registry = ModelRegistry()
