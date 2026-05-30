# OD-KGC/scripts/check_entity_class_quality.py

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.kg_loader import KGLoader


# ============================================================
# Basic helpers
# ============================================================

def is_numeric_like(value: Any) -> bool:
    """
    判断一个值是否像纯数字 class id。

    Examples:
        52      -> True
        "52"    -> True
        "0789"  -> True
        "city"  -> False
        "music group" -> False
    """

    if value is None:
        return False

    value = str(value).strip()

    if value == "":
        return False

    return value.isdigit()


def is_empty(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, str):
        return value.strip() == ""

    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0

    return False


def to_list(value: Any) -> List[str]:
    """
    将各种字段统一转为 list[str]。
    """

    if value is None:
        return []

    if isinstance(value, list):
        return [str(v) for v in value if v is not None and str(v).strip() != ""]

    if isinstance(value, tuple) or isinstance(value, set):
        return [str(v) for v in value if v is not None and str(v).strip() != ""]

    if isinstance(value, dict):
        results = []

        for k, v in value.items():
            if isinstance(v, (list, tuple, set)):
                results.extend([str(x) for x in v if x is not None and str(x).strip() != ""])
            elif v is not None and str(v).strip() != "":
                results.append(str(v))
            elif k is not None and str(k).strip() != "":
                results.append(str(k))

        return results

    return [str(value)]


def normalize_text(value: Any) -> str:
    value = str(value).strip()
    value = value.replace("_", " ")
    value = value.replace("/", " / ")
    value = " ".join(value.split())
    return value.lower()


def deduplicate(values: List[str]) -> List[str]:
    results = []
    seen = set()

    for value in values:
        value = normalize_text(value)

        if not value:
            continue

        if value not in seen:
            seen.add(value)
            results.append(value)

    return results


def get_raw_class_fields(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    从 raw entity item 中抽取 class/type 相关字段。
    """

    results = {}

    for key, value in raw.items():
        key_lower = str(key).lower()

        if "class" in key_lower or "type" in key_lower:
            results[key] = value

    return results


def get_preferred_class_names_from_entity(ent) -> List[str]:
    """
    模拟后续 ontology_filter 推荐使用的 class 提取策略：

    1. 优先使用自然语言 classname / class_name / types；
    2. 如果存在非数字 class name，就不使用 class_id；
    3. 只有完全没有自然语言 class 时，才 fallback 到 class_id/classid。
    """

    classes = []

    if getattr(ent, "classname", None):
        classes.extend(to_list(ent.classname))

    raw = getattr(ent, "raw", None) or {}

    for key in ["classname", "class_name", "classes", "types", "type"]:
        if key in raw:
            classes.extend(to_list(raw[key]))

    classes = deduplicate(classes)

    non_numeric_classes = [
        c for c in classes
        if not is_numeric_like(c)
    ]

    if non_numeric_classes:
        return non_numeric_classes

    fallback_classes = []

    if getattr(ent, "class_id", None):
        fallback_classes.extend(to_list(ent.class_id))

    for key in ["class", "classid", "class_id"]:
        if key in raw:
            fallback_classes.extend(to_list(raw[key]))

    return deduplicate(fallback_classes)


def get_current_loader_classes(ent) -> List[str]:
    """
    检查 kg_loader 加载出来的 classname + class_id。
    这个函数不代表最终应该怎么用，只用于定位问题。
    """

    classes = []

    if getattr(ent, "classname", None):
        classes.extend(to_list(ent.classname))

    if getattr(ent, "class_id", None):
        classes.extend(to_list(ent.class_id))

    return deduplicate(classes)


# ============================================================
# Main inspection function
# ============================================================

def inspect_entity_class_quality(
    data_path: str | Path,
    output_path: Optional[str | Path] = None,
    num_examples: int = 30,
) -> Dict[str, Any]:
    """
    检查 entity class 是否在数据加载阶段就变成了数字。

    重点输出：
        1. kg_loader 加载后的 classname 是否为数字；
        2. kg_loader 加载后的 class_id 是否为数字；
        3. raw entity.json 中 class/type 相关字段分布；
        4. 如果后续 filtering 使用 classname + class_id，会不会引入数字 class；
        5. 推荐使用的 preferred_classes 是否仍包含数字。
    """

    data_path = Path(data_path)

    print("=" * 100)
    print("[Entity Class Quality Check]")
    print(f"Dataset path: {data_path}")
    print("=" * 100)

    loader = KGLoader(data_path)
    dataset = loader.load()

    total_entities = len(dataset.entities)

    stats = {
        "total_entities": total_entities,

        "has_classname": 0,
        "missing_classname": 0,
        "numeric_classname": 0,
        "non_numeric_classname": 0,

        "has_class_id": 0,
        "missing_class_id": 0,
        "numeric_class_id": 0,
        "non_numeric_class_id": 0,

        "current_loader_classes_contain_numeric": 0,
        "preferred_classes_contain_numeric": 0,
        "preferred_classes_missing": 0,

        "raw_class_keys_counter": {},
    }

    examples = {
        "numeric_classname_examples": [],
        "numeric_class_id_examples": [],
        "current_loader_numeric_class_examples": [],
        "preferred_numeric_class_examples": [],
        "missing_classname_examples": [],
        "normal_classname_examples": [],
        "raw_class_field_examples": [],
    }

    for ent_id, ent in dataset.entities.items():
        label = getattr(ent, "label", None)
        classname = getattr(ent, "classname", None)
        class_id = getattr(ent, "class_id", None)
        raw = getattr(ent, "raw", None) or {}

        raw_class_fields = get_raw_class_fields(raw)

        for key in raw_class_fields.keys():
            stats["raw_class_keys_counter"][key] = (
                stats["raw_class_keys_counter"].get(key, 0) + 1
            )

        # ----------------------------------------------------
        # Check classname
        # ----------------------------------------------------

        if is_empty(classname):
            stats["missing_classname"] += 1

            if len(examples["missing_classname_examples"]) < num_examples:
                examples["missing_classname_examples"].append({
                    "entity_id": ent_id,
                    "label": label,
                    "classname": classname,
                    "class_id": class_id,
                    "raw_class_fields": raw_class_fields,
                })

        else:
            stats["has_classname"] += 1

            classname_values = to_list(classname)
            classname_has_numeric = any(is_numeric_like(v) for v in classname_values)

            if classname_has_numeric:
                stats["numeric_classname"] += 1

                if len(examples["numeric_classname_examples"]) < num_examples:
                    examples["numeric_classname_examples"].append({
                        "entity_id": ent_id,
                        "label": label,
                        "classname": classname,
                        "class_id": class_id,
                        "raw_class_fields": raw_class_fields,
                    })
            else:
                stats["non_numeric_classname"] += 1

                if len(examples["normal_classname_examples"]) < num_examples:
                    examples["normal_classname_examples"].append({
                        "entity_id": ent_id,
                        "label": label,
                        "classname": classname,
                        "class_id": class_id,
                    })

        # ----------------------------------------------------
        # Check class_id
        # ----------------------------------------------------

        if is_empty(class_id):
            stats["missing_class_id"] += 1
        else:
            stats["has_class_id"] += 1

            class_id_values = to_list(class_id)
            class_id_has_numeric = any(is_numeric_like(v) for v in class_id_values)

            if class_id_has_numeric:
                stats["numeric_class_id"] += 1

                if len(examples["numeric_class_id_examples"]) < num_examples:
                    examples["numeric_class_id_examples"].append({
                        "entity_id": ent_id,
                        "label": label,
                        "classname": classname,
                        "class_id": class_id,
                        "raw_class_fields": raw_class_fields,
                    })
            else:
                stats["non_numeric_class_id"] += 1

        # ----------------------------------------------------
        # Check current loader classes
        # ----------------------------------------------------

        current_classes = get_current_loader_classes(ent)

        if any(is_numeric_like(c) for c in current_classes):
            stats["current_loader_classes_contain_numeric"] += 1

            if len(examples["current_loader_numeric_class_examples"]) < num_examples:
                examples["current_loader_numeric_class_examples"].append({
                    "entity_id": ent_id,
                    "label": label,
                    "current_classes": current_classes,
                    "classname": classname,
                    "class_id": class_id,
                    "raw_class_fields": raw_class_fields,
                })

        # ----------------------------------------------------
        # Check preferred classes
        # ----------------------------------------------------

        preferred_classes = get_preferred_class_names_from_entity(ent)

        if not preferred_classes:
            stats["preferred_classes_missing"] += 1

        if any(is_numeric_like(c) for c in preferred_classes):
            stats["preferred_classes_contain_numeric"] += 1

            if len(examples["preferred_numeric_class_examples"]) < num_examples:
                examples["preferred_numeric_class_examples"].append({
                    "entity_id": ent_id,
                    "label": label,
                    "preferred_classes": preferred_classes,
                    "classname": classname,
                    "class_id": class_id,
                    "raw_class_fields": raw_class_fields,
                })

        # ----------------------------------------------------
        # Save raw examples
        # ----------------------------------------------------

        if raw_class_fields and len(examples["raw_class_field_examples"]) < num_examples:
            examples["raw_class_field_examples"].append({
                "entity_id": ent_id,
                "label": label,
                "raw_class_fields": raw_class_fields,
            })

    report = {
        "dataset_name": dataset.dataset_name,
        "dataset_dir": str(dataset.dataset_dir),
        "stats": stats,
        "examples": examples,
        "diagnosis": diagnose_class_problem(stats),
    }

    print_report(report)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n[Saved] Report saved to: {output_path}")

    return report


# ============================================================
# Diagnosis
# ============================================================

def diagnose_class_problem(stats: Dict[str, Any]) -> str:
    numeric_classname = stats["numeric_classname"]
    current_numeric = stats["current_loader_classes_contain_numeric"]
    preferred_numeric = stats["preferred_classes_contain_numeric"]

    if numeric_classname > 0:
        return (
            "Problem likely occurs in the data loading stage or raw entity.json: "
            "some entity.classname values are numeric. Please inspect entity.json "
            "and KGLoader._load_entities()."
        )

    if numeric_classname == 0 and current_numeric > 0 and preferred_numeric == 0:
        return (
            "KGLoader itself is mostly fine: classname is not numeric. "
            "The numeric classes are introduced when class_id/classid is mixed with classname. "
            "Use preferred class extraction in ontology_filter.py and avoid using numeric class_id "
            "when natural-language classname exists."
        )

    if preferred_numeric > 0:
        return (
            "Even the preferred class extraction still contains numeric classes. "
            "This means many entities do not have natural-language classname, "
            "so class_id needs an id-to-name mapping before sending to LLM."
        )

    return (
        "No serious numeric class-name problem detected. "
        "If LLM still receives numbers, check ontology_filter.py get_entity_classes()."
    )


def print_report(report: Dict[str, Any]) -> None:
    stats = report["stats"]
    examples = report["examples"]

    print("\n" + "=" * 100)
    print("[Summary]")
    print("=" * 100)
    print(f"Dataset: {report['dataset_name']}")
    print(f"Total entities: {stats['total_entities']}")

    print("\n[Loaded classname]")
    print(f"Has classname: {stats['has_classname']}")
    print(f"Missing classname: {stats['missing_classname']}")
    print(f"Numeric classname: {stats['numeric_classname']}")
    print(f"Non-numeric classname: {stats['non_numeric_classname']}")

    print("\n[Loaded class_id]")
    print(f"Has class_id: {stats['has_class_id']}")
    print(f"Missing class_id: {stats['missing_class_id']}")
    print(f"Numeric class_id: {stats['numeric_class_id']}")
    print(f"Non-numeric class_id: {stats['non_numeric_class_id']}")

    print("\n[Class extraction simulation]")
    print(
        "Current loader classes contain numeric: "
        f"{stats['current_loader_classes_contain_numeric']}"
    )
    print(
        "Preferred classes contain numeric: "
        f"{stats['preferred_classes_contain_numeric']}"
    )
    print(
        "Preferred classes missing: "
        f"{stats['preferred_classes_missing']}"
    )

    print("\n[Raw class-related keys]")
    raw_counter = stats["raw_class_keys_counter"]

    if raw_counter:
        for key, count in sorted(raw_counter.items(), key=lambda x: x[1], reverse=True):
            print(f"{key}: {count}")
    else:
        print("No class/type related keys found in raw entity data.")

    print("\n[Diagnosis]")
    print(report["diagnosis"])

    print("\n" + "=" * 100)
    print("[Examples: numeric classname]")
    print("=" * 100)
    print_examples(examples["numeric_classname_examples"])

    print("\n" + "=" * 100)
    print("[Examples: numeric class_id]")
    print("=" * 100)
    print_examples(examples["numeric_class_id_examples"])

    print("\n" + "=" * 100)
    print("[Examples: current loader classes contain numeric]")
    print("=" * 100)
    print_examples(examples["current_loader_numeric_class_examples"])

    print("\n" + "=" * 100)
    print("[Examples: preferred classes still contain numeric]")
    print("=" * 100)
    print_examples(examples["preferred_numeric_class_examples"])

    print("\n" + "=" * 100)
    print("[Examples: normal classname]")
    print("=" * 100)
    print_examples(examples["normal_classname_examples"])


def print_examples(items: List[Dict[str, Any]]) -> None:
    if not items:
        print("None")
        return

    for idx, item in enumerate(items, start=1):
        print(f"\n#{idx}")
        print(json.dumps(item, ensure_ascii=False, indent=2))


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Check whether entity class names are numeric in OD-KGC data loading."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/FB15k-237",
        help="Dataset path, e.g., data/FB15k-237 or data/WN18RR.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the inspection report JSON.",
    )

    parser.add_argument(
        "--num_examples",
        type=int,
        default=20,
        help="Number of examples to print/save for each problem type.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data_path)
    dataset_name = data_path.name

    if args.output_path is None:
        output_path = (
            PROJECT_ROOT
            / "import"
            / "debug"
            / f"{dataset_name}_entity_class_quality_report.json"
        )
    else:
        output_path = Path(args.output_path)

    inspect_entity_class_quality(
        data_path=data_path,
        output_path=output_path,
        num_examples=args.num_examples,
    )


if __name__ == "__main__":
    main()