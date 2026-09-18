from pathlib import Path
import json, numpy as np
from PIL import Image

class WorldTemplate:
    def __init__(self, root, template_id):
        self.root=Path(root)
        self.dir=self.root/"templates"/template_id
        self.meta=json.loads((self.dir/f"{template_id}.json").read_text())
        self._cache={}
    def _img(self, rel):
        if rel not in self._cache:
            self._cache[rel]=np.array(Image.open(self.dir/rel))
        return self._cache[rel]
    @staticmethod
    def uv_from_lonlat(longitude_deg, latitude_deg):
        u=((float(longitude_deg)+180.0)%360.0)/360.0
        v=(90.0-float(latitude_deg))/180.0
        return u, max(0.0,min(1.0,v))
    def _sample(self, arr, u, v):
        h,w=arr.shape[:2]
        x=u*(w-1); y=v*(h-1)
        x0=int(np.floor(x)); y0=int(np.floor(y))
        x1=(x0+1)%w; y1=min(h-1,y0+1)
        tx=x-x0; ty=y-y0
        a=arr[y0,x0]*(1-tx)+arr[y0,x1]*tx
        b=arr[y1,x0]*(1-tx)+arr[y1,x1]*tx
        return a*(1-ty)+b*ty
    def sample_environment(self, longitude_deg, latitude_deg):
        u,v=self.uv_from_lonlat(longitude_deg, latitude_deg)
        out={}
        for key in ("moisture_source","volatile_emission_source","food_density","mineral_resource_density"):
            rel=self.meta["maps"][key]["file"]
            out[key]=float(self._sample(self._img(rel).astype(np.float32)/255.0,u,v))
        return out
    def sample_elevation_normalized(self, longitude_deg, latitude_deg):
        u,v=self.uv_from_lonlat(longitude_deg, latitude_deg)
        rel=self.meta["maps"]["elevation"]["file"]
        val=float(self._sample(self._img(rel).astype(np.float32)/65535.0,u,v))
        return val*2.0-1.0
    def sample_spectrum(self, longitude_deg, latitude_deg):
        u,v=self.uv_from_lonlat(longitude_deg, latitude_deg)
        rel=self.meta["maps"]["spectral_reflectance"]["file"]
        key="__spectral__"
        if key not in self._cache:
            z=np.load(self.dir/rel)
            self._cache[key]=(z["wavelength_nm"].astype(np.float32), z["reflectance"].astype(np.float32))
        wl,a=self._cache[key]
        return wl, self._sample(a,u,v)
