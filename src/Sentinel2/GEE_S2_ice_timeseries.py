#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created from GEE_S2_timeseries.py
Modified to classify ice instead of water using methods from arcticIce.py

@author: ericlevenson
"""
import ee
import logging
import multiprocessing
from retry import retry
import sys

ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

start = '2023-3-1'
finish = '2025-8-26'

# Image scale
pixScale = 10
# Cloud probability threshold
CLD_PRB_THRESH = 50

# ***EARTH ENGINE-IFY***
# Ice season months: November, December, January, February, March, April, May
startDoy = ee.Date(start).getRelative('day', 'year')
endDoy = ee.Date(finish).getRelative('day', 'year')
eestart = ee.Date(start)
eefinish = ee.Date(finish)

@retry(tries=10, delay=1, backoff=2)
def getResult(index, lake_id):
    """Process a single lake ID and return ice time series"""
    
    # Import pre-processed lake polygons
    reaches = ee.FeatureCollection("projects/ee-eric-levenson/assets/ice_polygons")
    
    ###############################################################################
    ## ***IMAGE PRE-PROCESSING METHODS***

    def add_cloud_bands(img):
        # Get s2cloudless image, subset the probability band.
        cld_prb = ee.Image(img.get('s2cloudless')).select('probability')
        # Condition s2cloudless by the probability threshold value.
        clouds = cld_prb.gte(CLD_PRB_THRESH).rename('cloud_mask')
        clear = cld_prb.lt(CLD_PRB_THRESH).rename('clear_mask')
        # Add the cloud probability layer and cloud mask as image bands.
        return img.addBands(ee.Image([cld_prb, clouds, clear]))

    def clip_image(image):
        '''Clips to the roi defined at the beginning of the script'''
        return image.clip(roi)

    def getCover(image):
        '''calculates percentage of the roi covered by the clear mask'''
        actPixels = ee.Number(image.updateMask(image.select('clear_mask')).reduceRegion(
            reducer = ee.Reducer.count(),
            scale = 1000,
            geometry = image.geometry(),
            maxPixels=1e12,
            ).values().get(0))
        percCover = actPixels.divide(totPixels).multiply(100).round()
        return image.set('percCover', percCover,'actPixels',actPixels)

    def mosaicBy(imcol):
        '''Takes an image collection and creates a mosaic for each day'''
        imlist = imcol.toList(imcol.size())
        
        def imdate(im):
            date = ee.Image(im).date().format("YYYY-MM-dd")
            return date
        all_dates = imlist.map(imdate)
        
        def orbitId(im):
            orb = ee.Image(im).get('SENSING_ORBIT_NUMBER')
            return orb
        all_orbits = imlist.map(orbitId)
        
        def spacecraft(im):
            return ee.Image(im).get('SPACECRAFT_NAME')
        all_spNames = imlist.map(spacecraft)
        
        concat_all = all_dates.zip(all_orbits).zip(all_spNames)
        
        def concat(el):
            return ee.List(el).flatten().join(" ")
        concat_all = concat_all.map(concat)
        concat_unique = concat_all.distinct()
        
        def mosaicIms(d):
            d1 = ee.String(d).split(" ")
            date1 = ee.Date(d1.get(0))
            orbit = ee.Number.parse(d1.get(1)).toInt()
            spName = ee.String(d1.get(2))
            im = imcol.filterDate(date1, date1.advance(1, "day")).filterMetadata('SPACECRAFT_NAME', 'equals', spName).filterMetadata('SENSING_ORBIT_NUMBER','equals', orbit).mosaic()
            return im.set(
                "system:time_start", date1.millis(),
                "system:date", date1.format("YYYY-MM-dd"),
                "system:id", d1)
        
        mosaic_imlist = concat_unique.map(mosaicIms)
        return ee.ImageCollection(mosaic_imlist)

    ###########################################################################
    ## ***ICE CLASSIFICATION METHODS*** (from arcticIce.py)

    def ice_classify(image):
        '''Classify ice using B4 band threshold'''
        clear_mask = image.select('clear_mask')
        ice = image.select('B4').gte(975).rename('ice')  # Addy's threshold
        water = image.select('B4').lt(975).rename('water')
        ice = ice.updateMask(clear_mask)
        water = water.updateMask(clear_mask)  # Apply clear mask to ice classification
        return image.addBands([ice, water])

    def getIceFrac(image):
        '''calculates percentage of ice coverage in the ROI'''
        waterPixels = ee.Number(image.updateMask(image.select('water')).reduceRegion(
            reducer = ee.Reducer.count(),
            scale = 100, # keep same as totPixels
            geometry = image.geometry(),
            maxPixels=1e12,
            ).values().get(0))  
        icePixels = ee.Number(image.updateMask(image.select('ice')).reduceRegion(
           reducer = ee.Reducer.count(),
           scale = 100, # keep same as totPixels
           geometry = image.geometry(),
           maxPixels=1e12,
           ).values().get(0))
        # calculate the perc of cover OF CLEAR PIXELS
        percIce = icePixels.divide(waterPixels.add(icePixels)).multiply(100).round()
        # number as output
        return image.set('percIce', percIce, 'waterPixels', waterPixels, 'icePixels', icePixels)

    def dayProps(image):
        '''Extract ice properties for export'''
        date = image.date().format('yyyy-MM-dd')
        icePerc = image.get('percIce')
        coverage = image.get('percCover')
        return image.set('output', [date, lake_id, icePerc, coverage])

    #############################################################################
    ## *** Main Processing Script ***
    
    try:
        # Define roi
        reach = ee.Feature(reaches.filter(ee.Filter.eq('lake_id', lake_id)).first())
        try:
            roi = ee.Geometry.MultiPolygon(reach.geometry().getInfo()['coordinates'])
        except KeyError:
            subset = reach.geometry().getInfo()['geometries'][1:]
            roi = ee.Geometry.MultiPolygon([geom['coordinates'] for geom in subset if geom['type'] == 'Polygon'])
        
        # Get images - filter for ice season months (Nov, Dec, Jan, Feb, Mar, Apr, May)
        images = ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterBounds(roi).filterDate(start,finish).filter(ee.Filter.calendarRange(11, 5, 'month')).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',50))
        s2Cloudless = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY').filterBounds(roi).filterDate(start,finish).filter(ee.Filter.calendarRange(11, 5, 'month'))
        
        # Merge surface reflectance and cloud probability collections
        images = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
                'primary': images,
                'secondary': s2Cloudless,
                'condition': ee.Filter.equals(**{
                    'leftField': 'system:index',
                    'rightField': 'system:index'
                })
            }))
        
        images = images.map(add_cloud_bands)
        images_all = mosaicBy(images)
        riverimages = images_all.map(clip_image)
        
        # Filter by percentile cover
        image_mask = riverimages.select('B2').mean().gte(0)
        totPixels = ee.Number(image_mask.reduceRegion(
            reducer = ee.Reducer.count(),
            scale = 1000,
            geometry = roi,
            maxPixels = 1e12
            ).values().get(0))
        riverimages = riverimages.map(getCover)
        riverimages = riverimages.filterMetadata('percCover','greater_than',0)

        # Ice classification (instead of water classification)
        riverimages = riverimages.map(ice_classify)
        riverimages = riverimages.map(getIceFrac)
        rivers = riverimages.map(dayProps)
        
        # Get results
        result = rivers.aggregate_array('output').getInfo()
        
        # Write file
        filename = f'{lake_id}_ice.csv'
        directory = '/Users/ericlevenson/University of Oregon Dropbox/Eric Levenson/SWOT/production/data/timeseries/GEE_S2_ice_timeseries/'
        with open(directory + filename, 'w') as out_file:
            # Write header
            header = 'date,id,ice_percentage,coverage'
            print(header, file=out_file)
            # Write data
            for items in result:
                line = ','.join([str(item) for item in items])
                print(line, file=out_file)
        
        print(f"Done: {index} - {lake_id}", flush=True)
        return True
        
    except Exception as e:
        print(f"Error processing {lake_id}: {str(e)}", flush=True)
        return False

if __name__ == '__main__':
    
    # Set up flush for logging  
    class FlushStreamHandler(logging.StreamHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,  
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[FlushStreamHandler(sys.stderr)] 
    )
    
    # Lake IDs to process - using same IDs as original GEE_S2_timeseries.py
    ids= [7421087812, 7720003253, 7720025003, 7740023192]
    # Create output directory if it doesn't exist
    import os
    output_dir = '/Users/ericlevenson/University of Oregon Dropbox/Eric Levenson/SWOT/production/data/timeseries/GEE_S2_ice_timeseries/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process with multiprocessing
    pool = multiprocessing.Pool(10)  # Adjust number of processes as needed
    pool.starmap(getResult, enumerate(ids))
    pool.close()
    pool.join()