import itertools
import iris
import os
import numpy as np
import cartopy.crs as ccrs
from datetime import datetime, timedelta
from sklearn.neighbors import NearestNeighbors

iris.FUTURE.netcdf_promote = True
iris.FUTURE.netcdf_no_unlimited = True

def make_data_object_name(dataset_name, year, month, day, hour, realization, forecast_period):
    template_string = "prods_op_{}_{:02d}{:02d}{:02d}_{:02d}_{:02d}_{:03d}.nc"
    return template_string.format(dataset_name, year, month, day, hour, realization, forecast_period)

class GridcellDataset():
    def __init__(self, filenames, scale_factor, frac=1.0, mode='X', 
                 altitude_file='surface_altitude.nc', root='../data/'):
        filenames.sort()
        self.filenames = filenames
        self.scale_factor = scale_factor
        self.root = root
        self.mode = mode
        self.altitude_file = altitude_file
        times = [d for f in self.filenames for d in self._expand_date(self._extract_date(f))]
        self.times = [t for t in times if self._get_filename(t)[0] in self.filenames]
        self.n_times = len(self.times)
        
        self.params = [{'name': 'air_temperature', 'stash': 'm01s03i236'},
                       {'name': 'surface_air_pressure', 'stash': 'm01s00i409'},
                       {'name': 'x_wind', 'stash': 'm01s03i225'},
                       {'name': 'y_wind', 'stash': 'm01s03i226'},
                       {'name': 'specific_humidity', 'stash': 'm01s03i237'},
                       {'name': 'stratiform_rainfall_amount', 'stash': 'm01s04i201'}]
        
        c = iris.load(root+filenames[0])[0]
        self.n_lats = c.coord('grid_latitude').shape[0]
        self.n_lons = c.coord('grid_longitude').shape[0]
        
        filt = np.random.choice([0, 1], size=self._total_length(), p=[1-frac, frac])
        self.inds = np.where(filt==1)[0]
        
        self.alt_cube = iris.load(self.root + self.altitude_file)[0]
        
        self.cubes_hr = {}
        self.cubes_lr = {}
        for f in filenames:
            self._load_cubes(f)
            
    def set_filter(self, filt):
        self.inds = np.where(filt==1)[0]

    def _expand_date(self, d):
        hs = [i for i in range(1, 4)]
        if d.hour == 3:
            hs.append(0)
        return [d - timedelta(hours=h) for h in hs]

    def _extract_date(self, filename):
        t = datetime.strptime(filename[:31], 'prods_op_mogreps-uk_%Y%m%d_%H')
        lead_time = timedelta(hours=int(filename[-6:-3]))
        return t + lead_time
    
    def _reduce_dim(self, cube, dim):
        new_dim = np.linspace(cube.coord(dim).points[0], 
                              cube.coord(dim).points[-1],
                              num = cube.coord(dim).points.shape[0] // self.scale_factor)
        return new_dim
    
    def _upscale(self, cube):
        new = cube.copy()
        new_lat = self._reduce_dim(cube, 'grid_latitude')
        new_lon = self._reduce_dim(cube, 'grid_longitude')
        return new.interpolate(sample_points=[('grid_latitude', new_lat), ('grid_longitude', new_lon)],
                               scheme=iris.analysis.Linear())
    
    def _bilinear_downscale(self, upscaled, target):
        return upscaled.regrid(target, iris.analysis.Linear()) # defaults to n-linear

    def _get_filename(self, time):
        run = time
        while run.hour not in [3, 9, 15, 21]:
            run -= timedelta(hours=1)
            
        lead = time - run
        leadh = int(lead.total_seconds() / 3600)
        d_string = datetime.strftime(run, 'prods_op_mogreps-uk_%Y%m%d')
        fname = d_string + "_{:02d}_00_{:03d}.nc".format(run.hour, ((leadh // 3) + 1) * 3)
        
        return (fname, leadh % 3)
    
    def _nearest_dry_cell(self, rain, lat, lon):
        ospts = ccrs.OSGB().transform_points(rain.coord_system().as_cartopy_crs(),
                                             rain.coord('grid_longitude').points[np.where(rain.data == 0)[1]],
                                             rain.coord('grid_latitude').points[np.where(rain.data == 0)[0]])[:,:2]
        
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(ospts)
        
        x = ccrs.OSGB().transform_point(rain.coord('grid_longitude').points[lon],
                                rain.coord('grid_latitude').points[lat],
                                rain.coord_system().as_cartopy_crs())
        
        return nbrs.kneighbors(np.array([x]))[0].flatten()[0] / 1000
    
    def _select(self, c, **kwargs):
        s = [slice(None, None, None) for _ in range(c.ndim)]
        coords = [dc.standard_name for dc in c.dim_coords]
        for key, value in kwargs.items():
            s[c.coord_dims(key)[0]] = value
        return c[tuple(s)]
    
    def _load_cubes(self, filename):
        cubes = iris.load(self.root + filename, 
                          iris.AttributeConstraint(STASH=[p['stash'] for p in self.params]))
        hr_dict = {}
        lr_dict = {}
        
        for p in self.params:
            c_hr = cubes.extract(iris.AttributeConstraint(STASH=p['stash']))[0]
            c_lr = self._bilinear_downscale(self._upscale(c_hr), target=c_hr)
            hr_dict[p['name']] = c_hr
            lr_dict[p['name']] = c_lr
        
        self.cubes_hr[filename] = hr_dict
        self.cubes_lr[filename] = lr_dict
        
    def _load_cell(self, time, lat, lon):
        result = {}
        filename, leadtime = self._get_filename(self.times[time])
        cubes_hr = self.cubes_hr[filename]
        cubes_lr = self.cubes_lr[filename]

        crs = cubes_hr['x_wind'].coords('grid_latitude')[0].coord_system.as_cartopy_crs()
        p_lat = cubes_hr['x_wind'].coord('grid_latitude')[lat].points
        p_lon = cubes_hr['x_wind'].coord('grid_longitude')[lon].points
        r_lon, r_lat = ccrs.PlateCarree().transform_point(p_lon, p_lat, crs)
        result['longitude'] = r_lon; result['latitude'] = r_lat;
        
        result['DOY'] = self.times[time].timetuple().tm_yday
        result['surface_altitude'] = self._select(self.alt_cube, grid_latitude=lat, grid_longitude=lon).data.item()
        
        for p in self.params:
            result[p['name']] = self._select(cubes_lr[p['name']], grid_latitude=lat, 
                                             grid_longitude=lon, time=leadtime).data.item()
        
        rain = 'stratiform_rainfall_amount'
        result[rain + '_up'] = self._select(cubes_lr[rain], grid_latitude=lat+1, 
                                                 grid_longitude=lon, time=leadtime).data.item()
        result[rain + '_down'] = self._select(cubes_lr[rain], grid_latitude=lat-1, 
                                                   grid_longitude=lon, time=leadtime).data.item()
        result[rain + '_left'] = self._select(cubes_lr[rain], grid_latitude=lat, 
                                                   grid_longitude=lon-1, time=leadtime).data.item()
        result[rain + '_right'] = self._select(cubes_lr[rain], grid_latitude=lat, 
                                                    grid_longitude=lon+1, time=leadtime).data.item()
        
        empty = slice(None, None, None)
        precip = self._select(cubes_lr[rain], grid_latitude=empty, grid_longitude=empty, time=leadtime)
        result['distance'] = self._nearest_dry_cell(precip, lat, lon)
        
        target = self._select(cubes_hr[rain], 
                              grid_latitude=lat, grid_longitude=lon, time=leadtime).data.item()
    
        result['target'] = target
        return result
        
    def _convert_id(self, idx):
        time = idx // ((self.n_lats - 2) * (self.n_lons - 2))
        r = idx % ((self.n_lats - 2) * (self.n_lons - 2))
        lat = r // (self.n_lons - 2)
        lon = r % (self.n_lons - 2)
        return (time, lat + 1, lon + 1)

    def _total_length(self):
        return (self.n_times * (self.n_lats - 2) * (self.n_lons - 2)) - 1
    
    def __len__(self):
        return self.inds.shape[0]
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return np.array([self.__getitem__(x) for x in range(*idx.indices(self.__len__()))])
        if isinstance(idx, list):
            return np.array([self.__getitem__(x) for x in idx])
        return self._load_cell(*self._convert_id(self.inds[idx]))
    