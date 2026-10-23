"""Freeze source identities, declared licenses and untouched final datasets."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.source_adapters import CACHE, json_download


def main():
    cases = []
    registry = json.loads((ROOT / 'benchmarks/kaggle_cases.json').read_text())
    for key, specification in registry.items():
        try:
            metadata, provenance = json_download('https://www.kaggle.com/api/v1/datasets/view/' + specification['dataset'],
                CACHE / 'kaggle' / (key + '-metadata.json'))
            license_name = metadata.get('licenseName')
            # The Kaggle mirror says Unknown; the identified original dataset
            # has an explicit UCI CC BY 4.0 declaration, checked 2026-10-22 MSK.
            original_license = None
            if key == 'concrete_strength' and str(license_name).lower() == 'unknown':
                original_license = {'license': 'CC BY 4.0', 'author': 'I-Cheng Yeh',
                    'source': 'https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength',
                    'identity': '1030 rows, eight ingredients/age features, strength target; Kaggle mirrors UCI 165'}
                license_name = original_license['license']
            record = {'case': key, 'source': 'kaggle', 'dataset': specification['dataset'], 'dataset_key': key,
                'license': license_name, 'version': metadata.get('currentVersionNumber'),
                'source_url': 'https://www.kaggle.com/datasets/' + specification['dataset'],
                'target': specification['target'], 'metric': specification['metric'],
                'reserved_phase': 'development', 'status': 'metadata_available', 'metadata_provenance': provenance,
                'historical_final_lockbox': 'already used in prior development; not a new blind test'}
            if original_license:
                record.update(kaggle_declared_license=metadata.get('licenseName'), original_license_evidence=original_license)
        except Exception as error:
            record = {'case': key, 'source': 'kaggle', 'dataset_key': key, 'reserved_phase': 'development',
                'status': 'download_blocked', 'error': str(error)[:300]}
        cases.append(record)
        print(json.dumps({key: record.get(key) for key in ['case', 'license', 'status']}), flush=True)
    beyond = {}
    for filename in ['beyond-source-metadata.json', 'beyond-source-metadata-extra.json', 'beyond-source-reservations-extra.json']:
        path = ROOT / 'reports/performance_v2' / filename
        if path.exists():
            for record in json.loads(path.read_text()):
                beyond[record['dataset']] = record
    development = ['airfoil_self_noise', 'parkinsons_biomedical_voice_measurements', 'asp_potassco_classification',
        'garments_worker_productivity', 'healthcare_insurance_expenses', 'cardiotocography', 'thyroid_discordant', 'ecoli_proteins']
    reserved = ['bank_customer_churn', 'online_shoppers_purchasing_intention_dataset',
        'predict_students_dropout_and_academic_success', 'telemonitoring_parkinsons_biomedical_voice_measurements',
        'forest_fires', 'qsar_fish_toxicity', 'student_portuguese_performance',
        'biomechanical_orthopaedic_prediction', 'emscad', '5g_energy_consumption']
    for name, original in beyond.items():
        record = dict(original)
        record.setdefault('case', 'beyond_' + name)
        record.setdefault('source', 'beyondarena')
        record['reserved_phase'] = 'development' if name in development else 'final_lockbox' if name in reserved else 'excluded'
        if str(record.get('license', '')).lower() in {'', 'none', 'unknown'} and record.get('status') == 'metadata_available':
            record.update(status='dataset_not_licensed', reserved_phase='excluded', error='Curated metadata has no declared license')
        cases.append(record)
    for task_id in [3, 6, 12, 31, 37, 54]:
        cases.append({'case': 'openml_' + str(task_id), 'source': 'openml', 'task_id': task_id,
            'dataset': 'OpenML task ' + str(task_id) + ' (metadata unavailable)',
            'reserved_phase': 'development', 'status': 'download_blocked', 'metric': 'f1_macro',
            'metric_policy': 'Planned Macro F1 study variant on original task splits',
            'source_url': 'https://www.openml.org/t/' + str(task_id), 'license': None,
            'metadata_request_attempted': task_id in {3,31,37},
            'error': 'OpenML API timed out in direct requests; API also inaccessible through web tool. Source/split/labels not retrieved.'})
    manifest = {'schema_version': 2, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'seeds': [42, 53, 71], 'source_policy': 'Original curated splits; parent-only external labels; no source data in release',
        'cases': cases, 'reserved_final_cases': ['beyond_' + name for name in reserved],
        'lockbox_policy': 'Reserved datasets not loaded/evaluated before final code freeze; one joint baseline/candidate opening',
        'promotion_policy': 'Paired development results, failure-aware; final metrics never tune the frozen code'}
    (ROOT / 'benchmarks/manifest_v2.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'development_available': sum(row['reserved_phase'] == 'development' and row['status'] == 'metadata_available' for row in cases),
        'final_reserved': len(reserved), 'unavailable': sum(row['status'] != 'metadata_available' for row in cases)}))


if __name__ == '__main__':
    main()
