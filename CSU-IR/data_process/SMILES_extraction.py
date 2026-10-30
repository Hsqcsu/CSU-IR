import time
import urllib.parse
import requests

# Try to import local pyopsin module
try:
    import pyopsin

    HAS_PYOPSIN = True
except ImportError:
    HAS_PYOPSIN = False


def fetch_smiles_pubchem(substance_name, retries=3):
    """Query SMILES from PubChem PUG REST API by chemical name or CAS registry number.

    Note: The 'substance_name' parameter (and the resulting 'encoded_name') can
    be either a chemical substance name or a CAS Registry Number (e.g., '50-00-0'),
    as PubChem's compound name endpoint supports both.
    """
    if not substance_name:
        return None
    encoded_name = urllib.parse.quote(substance_name.strip(), safe="")
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/property/IsomericSMILES/TXT"
    for _ in range(retries):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                smi = resp.text.strip().split("\n")[0].strip()
                return smi if smi else None
            elif resp.status_code == 404:
                return None
            elif resp.status_code in [503, 429]:
                time.sleep(1.0)
        except Exception:
            time.sleep(0.5)
    return None


def name_to_smiles_iupac(name):
    """Convert chemical name to SMILES using IUPAC / OPSIN.

    (Prefers local pyopsin if available, falls back to the official OPSIN Web
    API).
    """
    if not name or not name.strip():
        return None
    name_clean = name.strip()

    # 1. Resolve via local pyopsin module
    if HAS_PYOPSIN:
        try:
            smi = pyopsin.name_to_smiles(name_clean)
            if smi and len(smi.strip()) > 0:
                return smi.strip()
        except Exception:
            pass

    # 2. Fallback to the official OPSIN Web API
    try:
        encoded_name = urllib.parse.quote(name_clean, safe="")
        url = f"https://opsin.ch.cam.ac.uk/opsin/{encoded_name}.smi"
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200 and resp.text.strip():
            return resp.text.strip().split()[0]
    except Exception:
        pass

    return None
