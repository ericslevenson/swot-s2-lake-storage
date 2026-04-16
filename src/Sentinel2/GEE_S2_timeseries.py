#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created from GEE_S2_timeseries.ipynb
Converted to local processing script

@author: ericlevenson
"""
import ee
import logging
import multiprocessing
from retry import retry
import sys

ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

start = '2023-7-21'
finish = '2025-10-6'

# Image scale
pixScale = 10
# Cloud probability threshold
CLD_PRB_THRESH = 50

# Export Properties
exportSelectors = ['date','id', 'waterArea_25', 'waterArea_50', 'cloudArea', 'coverage']

# ***EARTH ENGINE-IFY***
startDoy = ee.Date(start).getRelative('day', 'year')
endDoy = ee.Date(finish).getRelative('day', 'year')
eestart = ee.Date(start)
eefinish = ee.Date(finish)

@retry(tries=10, delay=1, backoff=2)
def getResult(index, lake_id):
    """Process a single lake ID and return water area time series"""
    
    # Import pre-processed lake polygons
    reaches = ee.FeatureCollection("projects/ee-eric-levenson/assets/benchmark_lake_polygons_wgs84")
    
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
    ## ***WATER CLASSIFICATION METHODS***

    def ndwi(image):
        '''Adds an NDWI band to the input image'''
        return image.normalizedDifference(['B3', 'B8']).rename('NDWI').multiply(1000)

    def ndwiMean(image):
        '''calculate NDWI histogram and add mean as a property to the image'''
        NDWI = ndwi(image).select('NDWI').updateMask(image.select('clear_mask'))
        ndwimean = ee.Dictionary(NDWI.reduceRegion(
            geometry = roi,
            reducer = ee.Reducer.histogram(255, 2).combine('mean', None, True).combine('variance', None, True),
            scale = 100,
            maxPixels = 1e12
            )).get('NDWI_mean')
        return image.set('ndwiMean', ndwimean)

    def otsu(histogram):
        '''Returns the NDWI threshold for binary water classification'''
        counts = ee.Array(ee.Dictionary(histogram).get('histogram'))
        means = ee.Array(ee.Dictionary(histogram).get('bucketMeans'))
        size = means.length().get([0])
        total = counts.reduce(ee.Reducer.sum(), [0]).get([0])
        sum = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0])
        mean = sum.divide(total)
        indices = ee.List.sequence(1, size)
        
        def func_xxx(i):
            '''Compute between sum of squares, where each mean partitions the data.'''
            aCounts = counts.slice(0, 0, i)
            aCount = aCounts.reduce(ee.Reducer.sum(), [0]).get([0])
            aMeans = means.slice(0, 0, i)
            aMean = aMeans.multiply(aCounts).reduce(ee.Reducer.sum(), [0]).get([0]).divide(aCount)
            bCount = total.subtract(aCount)
            bMean = sum.subtract(aCount.multiply(aMean)).divide(bCount)
            return aCount.multiply(aMean.subtract(mean).pow(2)).add(
                   bCount.multiply(bMean.subtract(mean).pow(2)))
        
        bss = indices.map(func_xxx)
        return means.sort(bss).get([-1])

    def otsu_thresh(water_image):
        '''Calculate NDWI and create histogram. Return the OTSU threshold.'''
        NDWI = ndwi(water_image).select('NDWI').updateMask(water_image.select('clear_mask'))
        histogram = ee.Dictionary(NDWI.reduceRegion(
            geometry = roi,
            reducer = ee.Reducer.histogram(255, 2).combine('mean', None, True).combine('variance', None, True),
            scale = pixScale,
            maxPixels = 1e12
        ))
        return otsu(histogram.get('NDWI_histogram'))

    def adaptive_thresholding(water_image):
        '''Takes an image clipped to lakes and returns the water mask'''
        NDWI = ndwi(water_image).select('NDWI')
        threshold = ee.Number(otsu_thresh(water_image))
        threshold = threshold.divide(10).round().multiply(10)
        
        histo = NDWI.reduceRegion(
            geometry = roi,
            reducer = ee.Reducer.fixedHistogram(-1000, 1000, 200),
            scale = pixScale,
            maxPixels = 1e12
        )
        hist = ee.Array(histo.get('NDWI'))
        counts = hist.cut([-1,1])
        buckets = hist.cut([-1,0])
        
        threshold = ee.Array([threshold]).toList()
        buckets_list = buckets.toList()
        split = buckets_list.indexOf(threshold)
        
        land_slice = counts.slice(0,0,split)
        water_slice = counts.slice(0,split.add(1),-1)
        
        land_max = land_slice.reduce(ee.Reducer.max(),[0])
        water_max = water_slice.reduce(ee.Reducer.max(),[0])
        land_max = land_max.toList().get(0)
        water_max = water_max.toList().get(0)
        land_max = ee.List(land_max).getNumber(0)
        water_max = ee.List(water_max).getNumber(0)
        
        counts_list = counts.toList()
        otsu_val = ee.Number(counts_list.get(split))
        otsu_val = ee.List(otsu_val).getNumber(0)
        land_prom = ee.Number(land_max).subtract(otsu_val)
        water_prom = ee.Number(water_max).subtract(otsu_val)
        
        land_thresh = ee.Number(land_max).subtract((land_prom).multiply(ee.Number(0.9)))
        water_thresh = ee.Number(water_max).subtract((water_prom).multiply(ee.Number(0.9)))
        land_max_ind = land_slice.argmax().get(0)
        water_max_ind = water_slice.argmax().get(0)
        li = ee.Number(land_max_ind).subtract(1)
        li = li.max(ee.Number(1))
        wi = ee.Number(water_max_ind).add(1)
        wi = wi.min(ee.Number(199))
        land_slice2 = land_slice.slice(0,li,-1).subtract(land_thresh)
        water_slice2 = water_slice.slice(0,0,wi).subtract(water_thresh)
        land_slice2  = land_slice2.abs().multiply(-1)
        water_slice2 = water_slice2.abs().multiply(-1)
        land_index = ee.Number(land_slice2.argmax().get(0)).add(land_max_ind)
        water_index = ee.Number(water_slice2.argmax().get(0)).add(split)
        land_level = ee.Number(buckets_list.get(land_index))
        water_level = ee.Number(buckets_list.get(water_index))
        land_level = ee.Number(ee.List(land_level).get(0)).add(5)
        water_level = ee.Number(ee.List(water_level).get(0)).add(5)
        
        water_fraction = (NDWI.subtract(land_level)).divide(water_level.subtract(land_level)).multiply(100).rename('water_fraction')
        water_25 = water_fraction.gte(25).rename('water_25')
        water_50 = water_fraction.gte(50).rename('water_50')
        water_60 = water_fraction.gte(60).rename('water_60')
        water_75 = water_fraction.gte(75).rename('water_75')
        all_mask = water_image.select('B2').gt(5).rename('all_mask')
        cloud_mask_ed = water_image.select('clear_mask').neq(1).rename('cloud_mask_ed')
        return water_image.addBands([water_fraction,water_25,water_50,water_60,water_75,NDWI,cloud_mask_ed])
    
    def riverProps(image):
        # Area images masked for everything
        areaIm = image.pixelArea()
        waterArea25Im = areaIm.updateMask(image.select('water_25'))
        waterArea50Im = areaIm.updateMask(image.select('water_50'))
        #waterArea60Im = areaIm.updateMask(image.select('water_60'))
        #waterArea75Im = areaIm.updateMask(image.select('water_75'))
        cloudAreaIm = areaIm.updateMask(image.select('cloud_mask'))
        clearAreaIm = areaIm.updateMask(image.select('clear_mask'))
        
        watersum25 = ee.Number(waterArea25Im.select('area').reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry = roi,
            scale = 10,
            maxPixels=1e9
        ).get('area', -999)).format('%2f')

        watersum50 = ee.Number(waterArea50Im.select('area').reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry = roi,
            scale = 10,
            maxPixels=1e9
        ).get('area', -999)).format('%2f')
        '''
        watersum60 = ee.Number(ee.Dictionary(waterArea60Im.select('area').reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry = roi,
            scale = 10,
            maxPixels=1e9
        )).get('area', -999)).format('%2f')

        watersum75 = ee.Number(waterArea75Im.select('area').reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry = roi,
            scale = 10,
            maxPixels=1e9
        ).get('area', -999)).format('%2f')
        '''
        cloudArea = ee.Number(cloudAreaIm.select('area').reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry = roi,
            scale = 10,
            maxPixels=1e9
        ).get('area', -999)).format('%2f')

        date = image.date().format('yyyy-MM-dd')
        coverage = image.get('percCover')

        return image.set('output', [date, lake_id, watersum25, watersum50, cloudArea, coverage])
        #return image.set('output', [date, lake_id, 'test_w25', watersum50, 'test_w60', 'test_w75', 'cloudArea', coverage])


    #############################################################################
    ## *** Main Processing Script ***
    
    try:
        # Define roi
        reach = ee.Feature(reaches.filter(ee.Filter.eq('swot_lake_', lake_id)).first())
        try:
            roi = ee.Geometry.MultiPolygon(reach.geometry().getInfo()['coordinates'])
        except KeyError:
            subset = reach.geometry().getInfo()['geometries'][1:]
            roi = ee.Geometry.MultiPolygon([geom['coordinates'] for geom in subset if geom['type'] == 'Polygon'])
        
        # Get images
        images = ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterBounds(roi).filterDate(start,finish).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',50))
        s2Cloudless = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY').filterBounds(roi).filterDate(start,finish)
        
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
        riverimages = riverimages.filter(ee.Filter.notNull(['percCover']))
        riverimages = riverimages.filterMetadata('percCover','greater_than',5)
        #riverimages = riverimages.filter(ee.Filter.notNull(['percCover']))
        #riverimages = riverimages.map(ndwiMean) # map ndwiMean value for filtering
        #riverimages = riverimages.filter(ee.Filter.notNull(['ndwiMean'])) # apply filter
        riverimages = riverimages.filter(ee.Filter.notNull(['system:time_start']))
        
        # Water classification
        riverimages = riverimages.map(adaptive_thresholding)      
        rivers = riverimages.map(riverProps)
        #print(rivers.size().getInfo())
        #return True
        
        # Get results
        result = rivers.aggregate_array('output').getInfo()
        
        # Write file
        filename = f'{lake_id}_S2.csv'
        directory = 'data/timeseries/GEE_S2_timeseries/'
        with open(directory + filename, 'w') as out_file:
            # Write header
            header = ','.join(exportSelectors)
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
    
    # Lake IDs to process
    #ids=[7720009473, 7420024963, 8121282692, 7720009733, 7410042502, 7720014343, 7420464263, 7730013832, 7510140043, 7740010893, 7510154643, 7420077843, 7740024723, 7320195353, 7420310553, 7730001053, 7820014623, 7420832032, 7410010233, 7120116133, 7250051493, 7740048553, 7720025003, 7420536883, 7430059572, 7420192693, 7421019193, 7820016633, 7510103363, 7430056772, 7820047173, 7820015173, 7720023243, 7520024523, 7520005453, 7830198863, 7420081743, 7420108243, 7720009683, 7120754902, 7120838103, 7420552793, 7420861403, 7410005852, 7740037982, 7720014303, 7420310883, 7510138853, 7720003433, 7510103403, 7420125293, 7420348653, 7740042733, 7430014452, 7420418293, 7720003573, 7420115193, 7510103423]
    ids=[7120003053]
    # Create output directory if it doesn't exist
    import os
    output_dir = 'data/timeseries/GEE_S2_timeseries/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process with multiprocessing
    pool = multiprocessing.Pool(10)  # Adjust number of processes as needed
    pool.starmap(getResult, enumerate(ids))
    pool.close()
    pool.join()