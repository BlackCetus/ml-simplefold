#!/usr/bin/env python3
import argparse
import os
import sys
import urllib.request
import urllib.error

def download_cif_files(id_file, output_dir, source, max_downloads=None):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(id_file):
        print(f"Error: Input file '{id_file}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(id_file, 'r') as f:
        ids = [line.strip().split(',')[0] for line in f if line.strip()]
        if ids and (ids[0].lower() in ['id', 'uniprot', 'target', 'accession']):
            ids = ids[1:]  # Skip header row

    if max_downloads:
        ids = ids[:max_downloads]

    total = len(ids)
    print(f"Loaded {total} IDs from {id_file}. Starting downloads from {source.upper()}...")

    success_count = 0
    fail_count = 0

    for idx, item_id in enumerate(ids, 1):
        
        # 1. Intercept Metagenomic ESM targets (MGYP)
        if item_id.lower().startswith("mgyp"):
            url = f"https://api.esmatlas.com/fetchPredictedStructure/{item_id.upper()}"
            # Note: ESM Atlas API natively returns .pdb files, not .cif
            dest_filename = f"{item_id.upper()}.pdb"
            
        # 2. Handle AlphaFold targets
        elif source == 'alphafold':
            if "-model_v" in item_id:
                base_id = item_id.split("-model_v")[0].upper()
                current_id = f"{base_id}-model_v6"
            elif item_id.startswith("AF-"):
                current_id = f"{item_id.upper()}-model_v6" if "-model" not in item_id else item_id
            else:
                current_id = f"AF-{item_id.upper()}-F1-model_v6"
            url = f"https://alphafold.ebi.ac.uk/files/{current_id}.cif"
            dest_filename = f"{current_id}.cif"
            
        # 3. Handle standard PDB targets
        else:
            url = f"https://files.rcsb.org/download/{item_id.upper()}.cif"
            dest_filename = f"{item_id.lower()}.cif"

        dest_path = os.path.join(output_dir, dest_filename)

        if os.path.exists(dest_path):
            success_count += 1
            print(f"[{idx}/{total}] Skipped (Already exists): {dest_filename}")
            continue

        try:
            urllib.request.urlretrieve(url, dest_path)
            success_count += 1
            print(f"[{idx}/{total}] Downloaded: {dest_filename}")
        except urllib.error.HTTPError as e:
            fail_count += 1
            print(f"[{idx}/{total}] Failed: {item_id} (HTTP Error {e.code}) -> Target URL: {url}", file=sys.stderr)
        except Exception as e:
            fail_count += 1
            print(f"[{idx}/{total}] Error downloading {item_id}: {e}", file=sys.stderr)

        print("\n--- Download Summary ---")
        print(f"Successfully processed/downloaded: {success_count}")
        print(f"Failed downloads: {fail_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk-download mmCIF files with auto-version upgrading.")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", default="data/raw_mmcif")
    parser.add_argument("-s", "--source", choices=["alphafold", "pdb"], default="alphafold")
    parser.add_argument("-n", "--limit", type=int, default=None)

    args = parser.parse_args()
    download_cif_files(args.input, args.output, args.source, args.limit)