#!/usr/bin/env python3
# Copyright (C) 2023, Daniel Stojcsics

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import rvebt.air_env.binvox_rw as binvox_rw
import numpy as np
import matplotlib.pyplot as plt

class ConvertBinvox():
    def load_binvox(self, file):
        '''
        Returns with opened binvox file as 3D array
        '''
        with open(file, 'rb') as f:
            return binvox_rw.read_as_3d_array(f)
        return None        
    
    def to_map(self, binvox, debug=False):
        '''
        Converts 3D array (from binvox file) and converts to terrain map.
        Each (x,y) cell contains the highest obstacle altitude
        Returns with map array, width & height (x,y dims) of map, and scale in meters
        '''
        obstacle_array=binvox.data.astype(int)
        np.swapaxes(obstacle_array,1,2)
        map = np.zeros((binvox.dims[0],binvox.dims[2]))
        for x in range(binvox.dims[0]):    
            for y in range(binvox.dims[2]):
                map[x,y]=max(0, np.where(obstacle_array[x,y,:])[0].max() - binvox.dims[1]//2)     
        
        # Testing:
        if debug:
            img = map/binvox.dims[1]//2 * 255            
            plt.imsave("/output/map.png", img, cmap='Greys') 

        # map, width, height, scale
        return [map, binvox.dims[0], binvox.dims[2], binvox.scale]

    def get_slice(self, map, altitude=5.0, tolerance=2.0, debug=False):
        '''
        Returns with a binary obstacle map (2D array) with obstacles present at a given altitued plus tolerance
        Default altitude: 5m with 2m tolerance
        '''
        slice = np.zeros_like(map)
        slice[map>(altitude-tolerance)] = 1

        # Testing        
        if debug:
            img = slice/25 * 255            
            plt.imsave("/output/slice.png", img, cmap='Greys') 

        return slice
        
        
       