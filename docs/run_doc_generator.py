# -*- coding: utf-8 -*-
"""
ROI Pipeline Documentation Generator
====================================
1) 파이프라인 실행 + 중간 과정 이미지 생성 (generate_images.py)
2) 이미지를 활용한 PPT 생성 (create_ppt.py)

Usage:
    cd docs/
    python run_doc_generator.py
"""
import sys
import os
from pathlib import Path

DOCS_DIR = Path(__file__).parent
PROJECT_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from docs directory explicitly to avoid conflict with root create_ppt.py
import importlib.util
_spec_gi = importlib.util.spec_from_file_location("doc_generate_images", str(DOCS_DIR / "generate_images.py"))
_mod_gi = importlib.util.module_from_spec(_spec_gi)
_spec_gi.loader.exec_module(_mod_gi)
run_pipeline_and_generate_images = _mod_gi.run_pipeline_and_generate_images

_spec_cp = importlib.util.spec_from_file_location("doc_create_ppt", str(DOCS_DIR / "create_ppt.py"))
_mod_cp = importlib.util.module_from_spec(_spec_cp)
_spec_cp.loader.exec_module(_mod_cp)
create_ppt = _mod_cp.create_ppt


def main():
    print("=" * 60)
    print("ROI Pipeline Documentation Generator")
    print("=" * 60)

    # Step 1: Generate all intermediate images
    print("\n[Phase 1] Running pipeline & generating images...")
    all_images, pipeline_data = run_pipeline_and_generate_images()

    # Step 2: Create PPT from images
    print("\n[Phase 2] Creating PPT presentation...")
    ppt_path = create_ppt(pipeline_data)

    print("\n" + "=" * 60)
    print("Documentation generation complete!")
    print(f"PPT: {ppt_path}")
    print(f"Images: {Path(__file__).parent / 'images'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
