"""Read-only, cached numeric world-pack assets. No supplied loader code is executed."""
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import numpy as np

PACK_ROOT=Path(__file__).resolve().parent/'assets'/'No_Mans_Fly_WorldPack_v2'
TEMPLATES=('earthlike_rocky','marslike_rocky','moonlike_rocky','gas_giant','trans_neptunian','rocky_vi_the_voyage_home')
WAVELENGTHS=np.array([330.,345.,355.,375.,405.,437.,478.,508.,550.,600.])

@lru_cache(maxsize=1)
def available():
    return (PACK_ROOT/'manifest.json').is_file()

@lru_cache(maxsize=1024)
def template_for(body_id):
    if str(body_id).endswith('-bootstrap'):return TEMPLATES[0]
    return TEMPLATES[int.from_bytes(hashlib.sha256(str(body_id).encode()).digest()[:4],'little')%len(TEMPLATES)]

@lru_cache(maxsize=6)
def metadata(template):
    if template not in TEMPLATES:raise ValueError('Unknown world template')
    path=PACK_ROOT/'templates'/template/f'{template}.json'
    value=json.loads(path.read_text(encoding='utf-8'))
    if value.get('id')!=template:raise ValueError('World template identity mismatch')
    return value

@lru_cache(maxsize=6)
def spectrum(template):
    root=(PACK_ROOT/'templates'/template).resolve()
    path=(root/metadata(template)['maps']['spectral_reflectance']['file']).resolve()
    if root not in path.parents:raise ValueError('World map escapes template directory')
    with np.load(path,allow_pickle=False) as data:
        if not np.array_equal(data['wavelength_nm'],WAVELENGTHS):raise ValueError('Incompatible world spectrum wavelengths')
        array=np.asarray(data['reflectance'],dtype=np.float32)
    if array.ndim!=3 or array.shape[2]!=10 or min(array.shape[:2])<2:raise ValueError('Invalid world spectrum shape')
    if not np.all(np.isfinite(array)) or np.any(array<0) or np.any(array>1):raise ValueError('Invalid world reflectance')
    array.setflags(write=False)
    return array
