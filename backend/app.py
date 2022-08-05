from flask import Flask, render_template, request
from flask_cors import CORS
import cv2
import math
import pandas as pd
from binascii import a2b_base64
import numpy as np
import base64
from sklearn.mixture import GaussianMixture
import json
from shapely import geometry, ops
import keras
import os, os.path
from PIL import Image
import matplotlib.pyplot as plt
import mplstereonet



app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return 'hello'

@app.route('/result', methods=['GET', 'POST'])
def result():
    curr_dir = os.getcwd()
    req_dir = os.path.join(curr_dir, "input_images")
    string_data = request.get_data().decode('utf-8')
    string_data = string_data[23:]
    #print(string_data)
    binary_data = a2b_base64(string_data)
    fd = open('input_images/image.png', 'wb')
    fd.write(binary_data)
    fd.close()
    number_files = len(os.listdir(req_dir))
    file_path = os.path.join(req_dir, str(number_files+1)+'.png')
    im = Image.open('input_images/image.png')
    im.save(file_path, 'png')

    image=cv2.imread("input_images/image.png")
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]

    white_img = np.full((height,width, 3),
                        0, dtype = np.uint8)

    blur = cv2.GaussianBlur(gray,(5,5),0)
    bilateral = cv2.bilateralFilter(blur, 7, 50, 50)

    edges = cv2.Canny(bilateral,50,150,apertureSize=3)

    lines = cv2.HoughLinesP(
			edges, # Input edge image
			1, # Distance resolution in pixels
			np.pi/180, # Angle resolution in radians
			threshold=25, # Min number of votes for valid line
			minLineLength=10, # Min allowed length of line
			maxLineGap=15 # Max allowed gap between line for joining them
			)

    sumOfLengthsOfLines = 0
    # Iterate over points
    lines_list = []

    for points in lines:
        x1,y1,x2,y2=points[0]   
        sumOfLengthsOfLines =  sumOfLengthsOfLines + ((((x2 - x1 )**2) + ((y2-y1)**2))**0.5)

    avg = sumOfLengthsOfLines/len(lines)

    for points in lines:
        x1,y1,x2,y2=points[0]
        if ((((x2 - x1 )**2) + ((y2-y1)**2) )**0.5) >= width*0.08:
            cv2.line(image,(x1,y1),(x2,y2),(0,0,0),3)
            cv2.line(white_img,(x1,y1),(x2,y2),(255,255,255),6)

    image2=cv2.imread("input_images/image.png")

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]

    blur = cv2.GaussianBlur(gray,(5,5),0)
    bilateral = cv2.bilateralFilter(blur, 9, 50, 50)
    edges = cv2.Canny(bilateral,50,150,apertureSize=3)

    lines = cv2.HoughLinesP(
			edges, # Input edge image
			1, # Distance resolution in pixels
			np.pi/180, # Angle resolution in radians
			threshold=25, # Min number of votes for valid line
			minLineLength=10, # Min allowed length of line
			maxLineGap=10 # Max allowed gap between line for joining them
			)

    sumOfLengthsOfLines = 0
    # Iterate over points
    lines_list = []
    for points in lines:
        # Extracted points nested in the list
        x1,y1,x2,y2=points[0]
        #  Draw the lines joing the points
        sumOfLengthsOfLines =  sumOfLengthsOfLines + ((((x2 - x1 )**2) + ((y2-y1)**2))**0.5)
        # Maintain a simples lookup list for points

    avg = sumOfLengthsOfLines/len(lines)
    theta=[]
    linescsv=[]
    for points in lines:
        x1,y1,x2,y2=points[0]
        if ((((x2 - x1 )**2) + ((y2-y1)**2) )**0.5) >=width*0.08:
            cv2.line(image2,(x1,y1),(x2,y2),(0,255,0),1)
            linescsv.append(points[0])
            if x1==x2:
                theta.append(90)
            else:
                if(y1==y2):
                    temptheta=0
                else:
                    temptheta=math.atan((y2-y1)/(x2-x1))*(180/math.pi)
                if temptheta<0:
                    theta.append(temptheta+180)
                else:
                    theta.append(temptheta)

    _, im_arr = cv2.imencode('.jpg', image2)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)

    data=pd.DataFrame()
    data['theta']=theta
    clust=10
    gmm = GaussianMixture(n_components=clust)
    gmm.fit(data)
    labels = gmm.predict(data)
    frame = pd.DataFrame(data)
    frame['cluster'] = labels

    x1=[]
    x2=[]
    y1=[]
    y2=[]
    c=[]
    for i in linescsv:
        x1.append(i[0])
        x2.append(i[2])
        y1.append(i[1])
        y2.append(i[3])
    for i in range(len(theta)):
        if theta[i]==90:
            c.append(0);
        elif theta[i]==0:
            c.append(y1[i])
        else:
            tan1=math.tan(theta[i]*math.pi/180)
            c.append(y1[i]-tan1*x1[i])

    
    df=pd.DataFrame()
    df['x1']=x1
    df['y1']=y1
    df['x2']=x2
    df['y2']=y2
    df['c']=c

    df['label']=labels
    df['theta']=theta
    xavg=[]
    yavg=[]
    k=len(labels)
    for i in range(k): 
        xavg.append((x1[i]+x2[i])/2)
        yavg.append((y1[i]+y2[i])/2)
    df['xavg']=xavg
    df['yavg']=yavg

    df.set_index(df.columns[5], inplace = True)
    df.sort_values(by=['label'],inplace=True)

    linespacing=[]
    for i in range(clust):
        if isinstance(df.xavg[i], np.floating)==False:
            close_points=[]
            point=[]
            len_point=len(df.xavg[i].values)
            for j in range(len_point):
                point.append([df.xavg[i].values[j],df.yavg[i].values[j],df.theta[i].values[j],df.c[i].values[j]])
            thetavg=sum(df.theta[i].values)/len(df.theta[i].values)
        
            if thetavg>=88 and thetavg<=92:
                point.sort(key=lambda x:x[0])
            else :
                point.sort(key=lambda x:x[3])
            
            for x in range(len_point-1):
                tan1=math.tan(point[x][2]*math.pi/180)
                tan2=math.tan(point[x+1][2]*math.pi/180)
                c1=(point[x][1]-tan1*point[x][0])
                c2=(point[x+1][1]-tan2*point[x+1][0])
                if 85<thetavg and thetavg<95:
                    dist=abs(point[x][0]-point[x+1][0])                                                           
                else:
                    dist=abs((c1-c2)/math.sqrt(((tan1+tan2)/2)**2+1))
                if dist<width/15:                                  #####  Set the value 
                
                    close_points.append([x,x+1])


            lst = [-1]*len_point
            lst_minpoint=[1]*len_point
#           print(i,close_points)
            for j in range(len(close_points)):                          ## list containing line which are far apart
#               print(close_points[j][0],close_points[j][1])
                if lst[close_points[j][0]]==-1 and lst[close_points[j][1]]==-1:
                    lst[close_points[j][0]]=1
                    lst[close_points[j][1]]=1
                    lst_minpoint[close_points[j][0]]=-1
                else:
                    lst[close_points[j][0]]=1  
                    lst[close_points[j][1]]=1
            spacing=0
            prevtan=-1
            prevc=-1
            prevx=-1
            prevy=-1
            size=0;
            for t in range(len_point):            
                if lst_minpoint[t]==-1 or lst[t]==-1:
                    size=size+1
                    if 85<thetavg and thetavg<95:
                        if(prevx!=-1):
                            spacing=spacing+point[t][0]-prevx 
                        prevx=point[t][0]                  
                    elif (thetavg>0 and thetavg<5) or (thetavg>175 and thetavg<180) :
                        if(prevy!=-1):
                            spacing=spacing+abs(point[t][1]-prevy)
                        prevy=point[t][1]
                    else:                  
                        tan1=math.tan(point[t][2]*math.pi/180)
                        c1=(point[t][1]-tan1*point[t][0])
                        if prevtan!=-1:
                            spacing=spacing+abs((c1-prevc)/math.sqrt(((tan1+prevtan)/2)**2+1))
                        
                        prevc=c1
                        prevtan=tan1
            linespacing.append(spacing)

    # RQD CALCULATION
    intersection=[]
    rqd=0
    nonzerodist=10
    for i in range(10):
        x_point=(width/10)*i
        Rqdlist=[]
        for j in range(len(x1)):
            if (x_point>=x1[j] and x_point<=x2[j]) or (x_point<=x1[j] and x_point>=x2[j]):
                if theta[i]!=90:
                    y_point=math.tan(theta[i]*math.pi/180)*x_point+c[i]
                    intersection.append([x_point,y_point])
        intersection.sort(key=lambda x:x[1])
        limit_dist=0
        dist=0
        for i in range(len(intersection)-1):
            tempdist=math.sqrt((intersection[i][1]-intersection[i+1][1])**2+(intersection[i][0]-intersection[i+1][0])**2)
            if(tempdist<height/2):
                limit_dist=limit_dist+tempdist
            dist=dist+tempdist
        if dist==0:
            nonzerodist=nonzerodist-1
        else:
            rqd=rqd+(dist-limit_dist)/dist
    RQD=rqd/nonzerodist

    vgg_best_model = keras.models.load_model('inceptionv3_-saved-model-35-loss-0.15.hdf5')
    img = cv2.resize(cv2.imread('input_images/image.png'),(150,150))
    img_normalized = img/255
    vgg16_image_prediction = np.argmax(vgg_best_model.predict(np.array([img_normalized])))
    str1 = str(vgg16_image_prediction*10)
    str2 = str(vgg16_image_prediction*10+10)
    ans = str1 + "-" + str2
    print(ans)

    #print(im_b64)
    res_uri = str(im_b64)
    res_uri = res_uri[2:]
    res_uri = res_uri[:-1]

    res = {'res_uri': res_uri, 'linespacing':linespacing, 'rqd': RQD, 'gsi':ans}

    return json.dumps(res)


@app.route('/failure', methods=['GET', 'POST'])
def failure():
    string_data = request.get_data().decode('utf-8')
    arr = string_data.split();
    #print(arr)
    d_strike = int(arr[0])
    s_strike = int(arr[1])
    d_dip = int(arr[2])
    s_dip = int(arr[3])
    f_angle = int(arr[4])
    #print(d_strike)
    #print(type(d_strike))

    def pole2plunge_bearing(strike, dip):
    
        strike, dip = np.atleast_1d(strike, dip)
        bearing = strike - 90
        plunge = 90 - dip
        bearing[bearing < 0] += 360
        return plunge, bearing

    

    

    def sph2cart(lon, lat):
        x = np.cos(lat)*np.cos(lon)
        y = np.cos(lat)*np.sin(lon)
        z = np.sin(lat)
        return x, y, z
    
    def cart2sph(x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        lat = np.arcsin(z/r)
        lon = np.arctan2(y, x)
        return lon, lat

    def _rotate(lon, lat, theta, axis='x'):
        # Convert input to numpy arrays in radians
        lon, lat = np.atleast_1d(lon, lat)
        lon, lat = map(np.radians, [lon, lat])
        theta = np.radians(theta)

        # Convert to cartesian coords for the rotation
        x, y, z = sph2cart(lon, lat)

        lookup = {'x':_rotate_x, 'y':_rotate_y, 'z':_rotate_z}
        X, Y, Z = lookup[axis](x, y, z, theta)

        # Now convert back to spherical coords (longitude and latitude, ignore R)
        lon, lat = cart2sph(X,Y,Z)
        return lon, lat # in radians!

    def _rotate_x(x, y, z, theta):
        X = x
        Y = y*np.cos(theta) + z*np.sin(theta)
        Z = -y*np.sin(theta) + z*np.cos(theta)
        return X, Y, Z

    def _rotate_y(x, y, z, theta):
        X = x*np.cos(theta) + -z*np.sin(theta)
        Y = y
        Z = x*np.sin(theta) + z*np.cos(theta)
        return X, Y, Z

    def _rotate_z(x, y, z, theta):
        X = x*np.cos(theta) + -y*np.sin(theta)
        Y = x*np.sin(theta) + y*np.cos(theta)
        Z = z
        return X, Y, Z

    def antipode(lon, lat):
        x, y, z = sph2cart(lon, lat)
        return cart2sph(-x, -y, -z)
    
    def line(plunge, bearing):
        plunge, bearing = np.atleast_1d(plunge, bearing)
        # Plot the approriate point for a bearing of 0 and rotate it
        lat = 90 - plunge
        lon = 0
        lon, lat = _rotate(lon, lat, bearing)
        return lon, lat

    

    

    def pole(strike, dip):
        strike, dip = np.atleast_1d(strike, dip)
        mask = dip > 90
        dip[mask] = 180 - dip[mask]
        strike[mask] += 180
        # Plot the approriate point for a strike of 0 and rotate it
        lon, lat = -dip, 0.0
        lon, lat = _rotate(lon, lat, strike)
        return lon, lat



    class PlanarSliding(object):

        def __init__(self, strike, dip, fric_angle=35, latlim=20):
        
            self.strike = strike
            self.dip = dip
            self.fric_angle = fric_angle
            self.latlim = latlim
        
            if latlim <= 0 or latlim >= 90:
                raise ValueError('latlim must be > 0 and < 90')
            
            if dip <= 0 or dip > 90:
                raise ValueError('dip must be > 0 and <= 90')
            
            if dip <= fric_angle:
                raise ValueError('No planar sliding zones generated as the input'
                             ' slope dip is shallower than the friction angle')
        def check_failure(self, strikes, dips, curved_lateral_limits=True):
            strikes = (strikes-self.strike)%360
            dipdirs = (strikes+90)%360
        
            if curved_lateral_limits:
#               lons, lats = stereonet_math.pole(strikes, dips)
                lons, lats =pole(strikes, dips)
                lats = np.degrees(lats)
                within_lat = ((lats >= -self.latlim-1e-8) &  # with tolerance
                            (lats <= self.latlim+1e-8))
            else:
                within_lat = ((dipdirs >= 90-self.latlim) &
                            (dipdirs <= 90+self.latlim))

#          llons, llats = stereonet_math.line(dips, dipdirs)
            llons, llats =line(plunges, bearings)
            llons = np.degrees(llons)
            daylight = llons >= 90-self.dip-1e-8  # with tolerance
        
            fric_slip = dips >= self.fric_angle
        
            main = within_lat & fric_slip & daylight
            secondary = ~within_lat & fric_slip & daylight
        
            return main, secondary
    plunges,bearings=pole2plunge_bearing(d_strike,d_dip)
    p2=PlanarSliding(s_strike,s_dip,f_angle)
    tup = p2.check_failure(d_strike,d_dip);

    a = tup[0][0]
    print(tup)
    b = tup[1][0]
    str1 = "error"
    str2 = "error"
    if a:
        str1 = "true"
    else:
        str1 = "false"

    return str1

@app.route('/failure2', methods=['GET', 'POST'])
def failure2():

    def sph2cart(lon, lat):
        """
        Converts a longitude and latitude (or sequence of lons and lats) given in
        _radians_ to cartesian coordinates, `x`, `y`, `z`, where x=0, y=0, z=0 is
        the center of the globe.
        Parameters
        ----------
        lon : array-like
            Longitude in radians
        lat : array-like
            Latitude in radians
        Returns
        -------
        `x`, `y`, `z` : Arrays of cartesian coordinates
        """
        x = np.cos(lat)*np.cos(lon)
        y = np.cos(lat)*np.sin(lon)
        z = np.sin(lat)
        return x, y, z

    def cart2sph(x, y, z):
        """
        Converts cartesian coordinates `x`, `y`, `z` into a longitude and latitude.
        x=0, y=0, z=0 is assumed to correspond to the center of the globe.
        Returns lon and lat in radians.
        Parameters
        ----------
        `x`, `y`, `z` : Arrays of cartesian coordinates
        Returns
        -------
        lon : Longitude in radians
        lat : Latitude in radians
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        lat = np.arcsin(z/r)
        lon = np.arctan2(y, x)
        return lon, lat

    def pole(strike, dip):
        """
        Calculates the longitude and latitude of the pole(s) to the plane(s)
        specified by `strike` and `dip`, given in degrees.
        Parameters
        ----------
        strike : number or sequence of numbers
            The strike of the plane(s) in degrees, with dip direction indicated by
            the azimuth (e.g. 315 vs. 135) specified following the "right hand
            rule".
        dip : number or sequence of numbers
            The dip of the plane(s) in degrees.
        Returns
        -------
        lon, lat : Arrays of longitude and latitude in radians.
        """
        strike, dip = np.atleast_1d(strike, dip)
        mask = dip > 90
        dip[mask] = 180 - dip[mask]
        strike[mask] += 180
        # Plot the approriate point for a strike of 0 and rotate it
        lon, lat = -dip, 0.0
        lon, lat = _rotate(lon, lat, strike)
        return lon, lat

    def _rotate(lon, lat, theta, axis='x'):
        """
        Rotate "lon", "lat" coords (in _degrees_) about the X-axis by "theta"
        degrees.  This effectively simulates rotating a physical stereonet.
        Returns rotated lon, lat coords in _radians_).
        """
        # Convert input to numpy arrays in radians
        lon, lat = np.atleast_1d(lon, lat)
        lon, lat = map(np.radians, [lon, lat])
        theta = np.radians(theta)

        # Convert to cartesian coords for the rotation
        x, y, z = sph2cart(lon, lat)

        lookup = {'x':_rotate_x, 'y':_rotate_y, 'z':_rotate_z}
        X, Y, Z = lookup[axis](x, y, z, theta)

        # Now convert back to spherical coords (longitude and latitude, ignore R)
        lon, lat = cart2sph(X,Y,Z)
        return lon, lat # in radians!

    def _rotate_x(x, y, z, theta):
        X = x
        Y = y*np.cos(theta) + z*np.sin(theta)
        Z = -y*np.sin(theta) + z*np.cos(theta)
        return X, Y, Z

    def _rotate_y(x, y, z, theta):
        X = x*np.cos(theta) + -z*np.sin(theta)
        Y = y
        Z = x*np.sin(theta) + z*np.cos(theta)
        return X, Y, Z

    def _rotate_z(x, y, z, theta):
        X = x*np.cos(theta) + -y*np.sin(theta)
        Y = x*np.sin(theta) + y*np.cos(theta)
        Z = z
        return X, Y, Z

    def geographic2plunge_bearing(lon, lat):
        """
        Converts longitude and latitude in stereonet coordinates into a
        plunge/bearing.
        Parameters
        ----------
        lon, lat : numbers or sequences of numbers
            Longitudes and latitudes in radians as measured from a
            lower-hemisphere stereonet
        Returns
        -------
        plunge : array
            The plunge of the vector in degrees downward from horizontal.
        bearing : array
            The bearing of the vector in degrees clockwise from north.
        """
        lon, lat = np.atleast_1d(lon, lat)
        x, y, z = sph2cart(lon, lat)

        # Bearing will be in the y-z plane...
        bearing = np.arctan2(z, y)

        # Plunge is the angle between the line and the y-z plane
        r = np.sqrt(x*x + y*y + z*z)
        r[r == 0] = 1e-15
        plunge = np.arcsin(x / r)

        # Convert back to azimuths in degrees..
        plunge, bearing = np.degrees(plunge), np.degrees(bearing)
        bearing = 90 - bearing
        bearing[bearing < 0] += 360

        # If the plunge angle is upwards, get the opposite end of the line
        upwards = plunge < 0
        plunge[upwards] *= -1
        bearing[upwards] -= 180
        bearing[upwards & (bearing < 0)] += 360

        return plunge, bearing

    def line(plunge, bearing):
        """
        Calculates the longitude and latitude of the linear feature(s) specified by
        `plunge` and `bearing`.
        Parameters
        ----------
        plunge : number or sequence of numbers
            The plunge of the line(s) in degrees. The plunge is measured in degrees
            downward from the end of the feature specified by the bearing.
        bearing : number or sequence of numbers
            The bearing (azimuth) of the line(s) in degrees.
        Returns
        -------
        lon, lat : Arrays of longitude and latitude in radians.
        """
        plunge, bearing = np.atleast_1d(plunge, bearing)
        # Plot the approriate point for a bearing of 0 and rotate it
        lat = 90 - plunge
        lon = 0
        lon, lat = _rotate(lon, lat, bearing)
        return lon, lat

    def _shape(shape_type, strike=0, dip=0, angle=0):
        """
        Prepare elements required to construct the kinematic analysis plots (e.g. 
        planes, cones) into Shapely geometries.
        """
        if shape_type=='plane':
            lon, lat = plane(strike, dip)
            return geometry.LineString(np.hstack((lon, lat)))
        
        elif shape_type=='curved_latlims':
            lon1, lat1, lon2, lat2 = _curved_latlims(angle)
            return [geometry.LineString(np.vstack((lon1, lat1)).T), 
                    geometry.LineString(np.vstack((lon2, lat2)).T)]
        
        elif shape_type=='cone':
            lon, lat = cone(90, 0, angle, segments=200)
            return geometry.Polygon(np.vstack((lon[0], lat[0])).T)
        
        elif shape_type=='daylight_envelope':
            lon, lat = daylight_envelope(strike, dip)
            return geometry.Polygon(np.hstack((lon[:-1], lat[:-1])))
        
        elif shape_type=='flexural_envelope':
            p_lon, p_lat = plane(0, 1e-9) # perimeter
            sl_lon, sl_lat = plane(strike, dip-angle)  # slip limit
            lon = np.vstack((p_lon, np.flip(sl_lon[1:-1])))
            lat = np.vstack((p_lat, np.flip(sl_lat[1:-1])))
            return geometry.Polygon(np.hstack((lon, lat)))
        
        elif shape_type=='wedge_envelope':
            sf_lon, sf_lat = plane(0, dip) # slope face
            sl_lon, sl_lat = plane(0, angle) # slip limit
            lon = np.vstack((sf_lon, np.flip(sl_lon[1:-1])))
            lat = np.vstack((sf_lat, np.flip(sl_lat[1:-1])))
            return geometry.Polygon(np.hstack((lon, lat)))

    def _set_kws(kws, polygon=False, color='None', edgecolor='None', alpha=None, label=None):
        """
        Set default kws for the kinematic analysis plot elements
        """

        kws = {} if kws is None else kws
        
        if 'lw' not in kws:
            kws.setdefault('linewidth', 1)

        if polygon:
            if 'color' not in kws:
                if 'fc' not in kws:
                    kws.setdefault('facecolor', color)
                if 'ec' not in kws:
                    kws.setdefault('edgecolor', edgecolor)
            kws.setdefault('alpha', alpha)
        else:
            if 'c' not in kws:
                kws.setdefault('color', color)
        
        kws.setdefault('label', label)
        
        return kws

    def plane(strike, dip, segments=100, center=(0, 0)):
        """
        Calculates the longitude and latitude of `segments` points along the
        stereonet projection of each plane with a given `strike` and `dip` in
        degrees.  Returns points for one hemisphere only.
        Parameters
        ----------
        strike : number or sequence of numbers
            The strike of the plane(s) in degrees, with dip direction indicated by
            the azimuth (e.g. 315 vs. 135) specified following the "right hand
            rule".
        dip : number or sequence of numbers
            The dip of the plane(s) in degrees.
        segments : number or sequence of numbers
            The number of points in the returned `lon` and `lat` arrays.  Defaults
            to 100 segments.
        center : sequence of two numbers (lon, lat)
            The longitude and latitude of the center of the hemisphere that the
            returned points will be in. Defaults to 0,0 (approriate for a typical
            stereonet).
        Returns
        -------
        lon, lat : arrays
            `num_segments` x `num_strikes` arrays of longitude and latitude in
            radians.
        """
        lon0, lat0 = center
        strikes, dips = np.atleast_1d(strike, dip)
        lons = np.zeros((segments, strikes.size), dtype=np.float)
        lats = lons.copy()
        for i, (strike, dip) in enumerate(zip(strikes, dips)):
            # We just plot a line of constant longitude and rotate it by the strike.
            dip = 90 - dip
            lon = dip * np.ones(segments)
            lat = np.linspace(-90, 90, segments)
            lon, lat = _rotate(lon, lat, strike)

            if lat0 != 0 or lon0 != 0:
                dist = angular_distance([lon, lat], [lon0, lat0], False)
                mask = dist > (np.pi / 2)
                lon[mask], lat[mask] = antipode(lon[mask], lat[mask])
                change = np.diff(mask.astype(int))
                ind = np.flatnonzero(change) + 1
                lat = np.hstack(np.split(lat, ind)[::-1])
                lon = np.hstack(np.split(lon, ind)[::-1])

            lons[:,i] = lon
            lats[:,i] = lat

        return lons, lats

    def _curved_latlims(angle, segments=100):
        """
        Calculates the longitude and latitude of `segments` points along the
        stereonet projection of the "curved" lateral limit bounds in both 
        direction, for strike=0. 
        """
        # Plot lines of constant latitude
        angle = np.radians(angle)
        lat1 = -angle * np.ones(segments)
        lon1 = np.linspace(-np.pi/2, np.pi/2, segments)
        lat2 = angle * np.ones(segments)
        lon2 = lon1.copy()
                
        return lon1, lat1, lon2, lat2

    def cone(plunge, bearing, angle, segments=100):
        """
        Calculates the longitude and latitude of the small circle (i.e. a cone)
        centered at the given *plunge* and *bearing* with an apical angle of
        *angle*, all in degrees.
        Parameters
        ----------
        plunge : number or sequence of numbers
            The plunge of the center of the cone(s) in degrees. The plunge is
            measured in degrees downward from the end of the feature specified by
            the bearing.
        bearing : number or sequence of numbers
            The bearing (azimuth) of the center of the cone(s) in degrees.
        angle : number or sequence of numbers
            The apical angle (i.e. radius) of the cone(s) in degrees.
        segments : int, optional
            The number of vertices in the small circle.
        Returns
        -------
        lon, lat : arrays
            `num_measurements` x `num_segments` arrays of longitude and latitude in
            radians.
        """
        plunges, bearings, angles = np.atleast_1d(plunge, bearing, angle)
        lons, lats = [], []
        for plunge, bearing, angle in zip(plunges, bearings, angles):
            lat = (90 - angle) * np.ones(segments, dtype=float)
            lon = np.linspace(-180, 180, segments)
            lon, lat = _rotate(lon, lat, -plunge, axis='y')
            lon, lat = _rotate(np.degrees(lon), np.degrees(lat), bearing, axis='x')
            lons.append(lon)
            lats.append(lat)
        return np.vstack(lons), np.vstack(lats)

    def daylight_envelope(strike, dip, segments=500):
        """
        Calculates the longitude and latitude of `segments` points along the
        stereonet projection of the daylight envelope of each slope face
        with a given `strike` and `dip` in degrees.
        Parameters
        ----------
        strike : number or sequence of numbers
            The strike of the plane(s) (slope face) in degrees, with dip direction
            indicated by the azimuth (e.g. 315 vs. 135) specified following the 
            "right hand rule".
        dip : number or sequence of numbers
            The dip of the plane(s) in degrees.
        segments : number or sequence of numbers
            The number of points in the returned `lon` and `lat` arrays.  Defaults
            to 500 segments.
        Returns
        -------
        lon, lat : arrays
            `num_segments` x `num_strikes` arrays of longitude and latitude in
            radians.
        """
        
        # Get apparent dips from -90 to +90 (azimuth difference) from slope dip 
        # direction, i.e. +0 to +180 from slope strike. This essentially generates 
        # points defining the great-circle plane that represents the slope face
        dl_bearings = np.linspace(0, 180, segments).reshape(segments, 1)
        dl_plunges = apparent_dip(dip, 90-dl_bearings)
        
        # More points needed for daylight envelope for steep slopes
        if dip > 89:
            # Crop original end sections at apparent dip = 0
            dl_bearings = dl_bearings[1:-1]
            # Create main section. End points cropped to avoid overlapping
            b2 = dl_bearings[1:-1]
            p2 = apparent_dip(dip, 90-b2)
            # Get the apparent dip of the cropped end points (new connection points)
            connect_dip = apparent_dip(dip, 90 - dl_bearings[0])
            # Create the two new end sections, by generating points from 
            # apparent dip = 0 to the apparent dip of the connection points 
            p1 = np.linspace(0, connect_dip, segments)
            b1 = 90 + azimuth_diff(dip, p1)
            p3 = p1[::-1]
            b3 = 90 - azimuth_diff(dip, p3)
            # Connect the 3 sections
            dl_bearings = np.vstack((b1, b2[::-1], b3))
            dl_plunges = np.vstack((p1, p2[::-1], p3))

        # Convert to lat,lon of poles
        lon, lat = pole(dl_bearings-90, dl_plunges)
        lon, lat = _rotate(np.degrees(lon), np.degrees(lat), strike)
        return lon, lat

    def angular_distance(first, second, bidirectional=True):
        """
        Calculate the angular distance between two linear features or elementwise
        angular distance between two sets of linear features. (Note: a linear
        feature in this context is a point on a stereonet represented
        by a single latitude and longitude.)
        Parameters
        ----------
        first : (lon, lat) 2xN array-like or sequence of two numbers
            The longitudes and latitudes of the first measurements in radians.
        second : (lon, lat) 2xN array-like or sequence of two numbers
            The longitudes and latitudes of the second measurements in radians.
        bidirectional : boolean
            If True, only "inner" angles will be returned. In other words, all
            angles returned by this function will be in the range [0, pi/2]
            (0 to 90 in degrees).  Otherwise, ``first`` and ``second``
            will be treated as vectors going from the origin outwards
            instead of bidirectional infinite lines.  Therefore, with
            ``bidirectional=False``, angles returned by this function
            will be in the range [0, pi] (zero to 180 degrees).
        Returns
        -------
        dist : array
            The elementwise angular distance between each pair of measurements in
            (lon1, lat1) and (lon2, lat2).
        Examples
        --------
        Calculate the angle between two lines specified as a plunge/bearing
            >>> angle = angular_distance(line(30, 270), line(40, 90))
            >>> np.degrees(angle)
            array([ 70.])
        Let's do the same, but change the "bidirectional" argument:
            >>> first, second = line(30, 270), line(40, 90)
            >>> angle = angular_distance(first, second, bidirectional=False)
            >>> np.degrees(angle)
            array([ 110.])
        Calculate the angle between two planes.
            >>> angle = angular_distance(pole(0, 10), pole(180, 10))
            >>> np.degrees(angle)
            array([ 20.])
        """
        lon1, lat1 = first
        lon2, lat2 = second
        lon1, lat1, lon2, lat2 = np.atleast_1d(lon1, lat1, lon2, lat2)
        xyz1 = sph2cart(lon1, lat1)
        xyz2 = sph2cart(lon2, lat2)
        # This is just a dot product, but we need to work with multiple measurements
        # at once, so einsum is quicker than apply_along_axis.
        dot = np.einsum('ij,ij->j', xyz1, xyz2)
        angle = np.arccos(dot)

        # There are numerical sensitivity issues around 180 and 0 degrees...
        # Sometimes a result will have an absolute value slighly over 1.
        if np.any(np.isnan(angle)):
            rtol = 1e-4
            angle[np.isclose(dot, -1, rtol)] = np.pi
            angle[np.isclose(dot, 1, rtol)] = 0

        if bidirectional:
            mask = angle > np.pi / 2
            angle[mask] = np.pi - angle[mask]

        return angle

    def antipode(lon, lat):
        """
        Calculates the antipode (opposite point on the globe) of the given point or
        points. Input and output is expected to be in radians.
        Parameters
        ----------
        lon : number or sequence of numbers
            Longitude in radians
        lat : number or sequence of numbers
            Latitude in radians
        Returns
        -------
        lon, lat : arrays
            Sequences (regardless of whether or not the input was a single value or
            a sequence) of longitude and latitude in radians.
        """
        x, y, z = sph2cart(lon, lat)
        return cart2sph(-x, -y, -z)

    def apparent_dip(true_dip, azimuth_diff):
        """
        Calculate the apparent dip angle(s), given the true dip angle(s) and the 
        azimuth (i.e. bearing) difference between the apparent dip direction(s) and
        true dip direction(s). All in degrees.
        Parameters
        ----------
        true_dip : number or sequence of numbers
            true dip angle(s) in degrees
        azimuth_diff : number or sequence of numbers
            azimuth difference between the apparent dip direction(s) and true dip 
            direction(s), in degrees
        Returns
        -------
        apparent_dip : array
            apparent dip angle(s) in degrees
        """    
        true_dip, azimuth_diff = np.atleast_1d(true_dip, azimuth_diff)
        azimuth_diff = np.radians(np.abs(azimuth_diff))
        true_dip = np.radians(true_dip)
        apparent_dip = np.arctan(np.cos(azimuth_diff)*np.tan(true_dip))
        return np.degrees(apparent_dip)

    def azimuth_diff(true_dip, apparent_dip):
        """
        Calculate the absolute difference in azimuth (i.e. bearing) between true 
        dip direction(s) and apparent dip direction(s), given the true dip angle(s)
        and the apparent dip angle(s). All in degrees.
        Parameters
        ----------
        true_dip : number or sequence of numbers
            true dip angle(s) in degrees
        apparent_dip : number or sequence of numbers
            apparent dip angle(s) in degrees
        Returns
        -------
        azimuth_diff : array
            azimuth difference between the apparent dip direction(s) and true dip 
            direction(s)
        """    
        true_dip, apparent_dip = np.atleast_1d(true_dip, apparent_dip)
        true_dip = np.radians(true_dip)
        apparent_dip = np.radians(apparent_dip)
        azimuth_diff = np.arccos(np.tan(apparent_dip)/np.tan(true_dip))
        return np.degrees(azimuth_diff)

    def subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, hemisphere='lower', projection='equal_area', **fig_kw):
        """
        Identical to matplotlib.pyplot.subplots, except that this will default to
        producing equal-area stereonet axes.
        This prevents constantly doing:
            >>> fig, ax = plt.subplot(subplot_kw=dict(projection='stereonet'))
        or
            >>> fig = plt.figure()
            >>> ax = fig.add_subplot(111, projection='stereonet')
        Using this function also avoids having ``mplstereonet`` continually appear
        to be an unused import when one of the above methods are used.
        Parameters
        -----------
        nrows : int
        Number of rows of the subplot grid.  Defaults to 1.
        ncols : int
        Number of columns of the subplot grid.  Defaults to 1.
        hemisphere : string
            Currently this has no effect. When upper hemisphere and dual
            hemisphere plots are implemented, this will control which hemisphere
            is displayed.
        projection : string
            The projection for the axes. Defaults to 'equal_area'--an equal-area
            (a.k.a. "Schmidtt") stereonet. May also be 'equal_angle' for an
            equal-angle (a.k.a. "Wulff") stereonet or any other valid matplotlib
            projection (e.g. 'polar' or 'rectilinear' for a "normal" axes).
        The following parameters are identical to matplotlib.pyplot.subplots:
        sharex : string or bool
        If *True*, the X axis will be shared amongst all subplots.  If
        *True* and you have multiple rows, the x tick labels on all but
        the last row of plots will have visible set to *False*
        If a string must be one of "row", "col", "all", or "none".
        "all" has the same effect as *True*, "none" has the same effect
        as *False*.
        If "row", each subplot row will share a X axis.
        If "col", each subplot column will share a X axis and the x tick
        labels on all but the last row will have visible set to *False*.
        sharey : string or bool
            If *True*, the Y axis will be shared amongst all subplots. If
            *True* and you have multiple columns, the y tick labels on all but
            the first column of plots will have visible set to *False*
            If a string must be one of "row", "col", "all", or "none".
            "all" has the same effect as *True*, "none" has the same effect
            as *False*.
            If "row", each subplot row will share a Y axis.
            If "col", each subplot column will share a Y axis and the y tick
            labels on all but the last row will have visible set to *False*.
        *squeeze* : bool
            If *True*, extra dimensions are squeezed out from the
            returned axis object:
            - if only one subplot is constructed (nrows=ncols=1), the
            resulting single Axis object is returned as a scalar.
            - for Nx1 or 1xN subplots, the returned object is a 1-d numpy
            object array of Axis objects are returned as numpy 1-d
            arrays.
            - for NxM subplots with N>1 and M>1 are returned as a 2d
            array.
        If *False*, no squeezing at all is done: the returned axis
            object is always a 2-d array contaning Axis instances, even if it
            ends up being 1x1.
        *subplot_kw* : dict
            Dict with keywords passed to the
            :meth:`~matplotlib.figure.Figure.add_subplot` call used to
            create each subplots.
        *fig_kw* : dict
            Dict with keywords passed to the :func:`figure` call.  Note that all
            keywords not recognized above will be automatically included here.
        Returns
        --------
        fig, ax : tuple
            - *fig* is the :class:`matplotlib.figure.Figure` object
            - *ax* can be either a single axis object or an array of axis
            objects if more than one supblot was created.  The dimensions
            of the resulting array can be controlled with the squeeze
            keyword, see above.
        """
        if projection in ['equal_area', 'equal_angle']:
            projection += '_stereonet'
        if subplot_kw == None:
            subplot_kw = {}
        subplot_kw['projection'] = projection
        return plt.subplots(nrows, ncols, sharex=sharex, sharey=sharey,
                            squeeze=squeeze, subplot_kw=subplot_kw, **fig_kw)

    def _rotate_shape(shape, strike):
        """
        Rotate the Shapely geometries according to a certain strike and return the 
        latitude, longitude arrays.
        """
        if shape.geom_type == 'LineString':
            lon, lat = shape.xy
        elif shape.geom_type == 'Polygon':
            lon, lat = shape.exterior.xy    
        lon = np.degrees(lon)
        lat = np.degrees(lat)
        lon, lat = _rotate(lon, lat, strike)

        return np.array([lon, lat])

    class PlanarSliding(object):
        """ 
        Kinematic analysis for planar sliding failures
    
        Parameters
        ----------
        strike : number
            The strike of the slope face in degrees, with dip direction indicated 
            by the azimuth (e.g. 315 vs. 135) specified following the "right hand
            rule".
        dip : number (> 0 and <90)
            The dip of the slope face in degrees.
        fric_angle : number, default=35
            The friction angle along the sliding discontinuities, in degrees. 
            Note that the slope dip should be steeper than the friction angle, or 
            else no planar sliding zones can be generated.
        latlim : number (> 0 and <90), default=20
            The lateral limits for typical planar sliding failures, counted from 
            the dip direction of the slope face in degrees. Daylighting 
            discontinuities dipping steeper than friction angle but outside the 
            lateral limits are considered to be less probable (i.e. secdondary 
            failure zones).
        """    

        def __init__(self, strike, dip, fric_angle=35, latlim=20):
        
            self.strike = strike
            self.dip = dip
            self.fric_angle = fric_angle
            self.latlim = latlim
        
            if latlim <= 0 or latlim >= 90:
                raise ValueError('latlim must be > 0 and < 90')
            
            if dip <= 0 or dip > 90:
                raise ValueError('dip must be > 0 and <= 90')
            
            if dip <= fric_angle:
                raise ValueError('No planar sliding zones generated as the input'
                                ' slope dip is shallower than the friction angle')

        def check_failure(self, strikes, dips, curved_lateral_limits=True):
            """ 
            Check whether planar sliding failures are kinematically feasible on a 
            sequence of discontinuities
        
            Parameters
            ----------
            strikes : numbers
                The strikes of the discontinuities in degrees, with dip direction 
                indicated by the azimuth (e.g. 315 vs. 135) specified following the
                "right hand rule".
            dips : numbers
                The dip angles of the discontinuities in degrees.
            curved_lateral_limits : boolean
                Consider lateral limits as curved lines (align with small circles) 
                if set to 'True'. Straight lines through the stereonet center are 
                used if set to 'False'. Defaults to 'True'
        
            Returns
            ----------
            main: squence of booleans
                True if the discontinuity is in the main planar sliding zone
            secondary: squence of booleans
                True if the discontinuity is in the secondary planar sliding zone
            """    
            strikes = (strikes-self.strike)%360
            dipdirs = (strikes+90)%360
        
            if curved_lateral_limits:
                lons, lats = pole(strikes, dips)
                lats = np.degrees(lats)
                within_lat = ((lats >= -self.latlim-1e-8) &  # with tolerance
                            (lats <= self.latlim+1e-8))
            else:
                within_lat = ((dipdirs >= 90-self.latlim) &
                            (dipdirs <= 90+self.latlim))

            llons, llats = line(dips, dipdirs)
            llons = np.degrees(llons)
            daylight = llons >= 90-self.dip-1e-8  # with tolerance
        
            fric_slip = dips >= self.fric_angle
        
            main = within_lat & fric_slip & daylight
            secondary = ~within_lat & fric_slip & daylight
        
            return main, secondary
    
        def plot_kinematic(self, secondary_zone=True, construction_lines=True, 
                        slopeface=True, curved_lateral_limits=True,
                        main_kws=None, secondary_kws=None, lateral_kws=None,
                        friction_kws=None, daylight_kws=None, slope_kws=None, 
                        ax=None):
                   
            """
            Generate the planar sliding kinematic analysis plot for pole vectors. 
            (Note: The discontinuity data to be used in conjunction with this plot 
            should be displayed as POLES)
        
        This function plots the following elements on a StereonetAxes: 
            (1) main planar sliding zone
            (2) secondary planar sliding zones
            (3) construction lines, i.e. friction cone, lateral limits and 
                daylight envelope
            (4) slope face
        
            (2)-(4) are optioanl. The style of the elements above can be specified 
            with their kwargs, or on the artists returned by this function later.
        
            Parameters
            ----------
            secondary_zone : boolean
                Plot the secondary zones if set to True. Defaults to 'True'.
            construction_lines : boolean
                Plot the construction lines if set to True. Defaults to 'True'.
            slopeface : boolean
                Plot the slope face as a great-circle plane on stereonet. Defaults
                to 'True'.
            curved_lateral_limits : boolean
                Plot curved lateral limits (align with small circles) if set to 
                True, or else will be plotted as straight lines through the 
                stereonet center. Defaults to 'True'
            main_kws : dictionary
                kwargs for the main planar sliding zone 
                (matplotlib.patches.Polygon)
            secondary_kws : dictionary
                kwargs for the secondary planar sliding zones 
                (matplotlib.patches.Polygon)
            lateral_kws : dictionary
                kwargs for the lateral limits (matplotlib.lines.Line2D)
            friction_kws : dictionary
                kwargs for the friction cone (matplotlib.patches.Polygon)
            daylight_kws : dictionary
                kwargs for the daylight envelope (matplotlib.patches.Polygon)
            slope_kws : dictionary
                kwargs for the slope face (matplotlib.lines.Line2D)
            ax : StereonetAxes
                The StereonetAxes to plot on. A new StereonetAxes will be generated
                if set to 'None'. Defaults to 'None'.
        
            Returns
            -------
            result : dictionary
                A dictionary mapping each element of the kinematic analysis plot to
                a list of the artists created. The dictionary has the following 
                keys:
                - `main` : the main planar sliding zone
                - `secondary` : the two secondary planar sliding zones
                - `slope` : the slope face
                - `daylight` : the daylight envelope 
                - `friction` : the friction cone
                - `lateral` : the two lateral limits
            """
        
            # Convert the construction lines into shapely linestrings / polygons    
            daylight_envelope = _shape('daylight_envelope', strike=0, dip=self.dip)
            friction_cone = _shape('cone', angle=self.fric_angle)
            if curved_lateral_limits:
                lat_lim1, lat_lim2 = _shape('curved_latlims', angle=self.latlim)
            else:
                lat_lim1 = _shape('plane', strike=90-self.latlim, dip=90)
                lat_lim2 = _shape('plane', strike=90+self.latlim, dip=90)
        
            # Get the failure zones (as shapely polygons) from geometry interaction
            sliding_zone = daylight_envelope.difference(friction_cone)
            split_polys = ops.split(sliding_zone,lat_lim1)
            sec_zone_present = len(split_polys)==2
            if sec_zone_present:
                if split_polys[0].intersects(lat_lim2):
                    sliding_zone, sec_zone1 = split_polys
                else:
                    sec_zone1, sliding_zone = split_polys
                
                split_polys = ops.split(sliding_zone,lat_lim2)
                if split_polys[0].touches(sec_zone1):
                    sliding_zone, sec_zone2 = split_polys
                else:
                    sec_zone2, sliding_zone = split_polys
                
            # Start plotting
            if ax==None:
                figure, axes = subplots(figsize=(8, 8))
            else:
                axes = ax
            
            # List of artists to be output
            main = []
            secondary = []
            slope = []
            daylight = []
            friction = []
            lateral = []
        
            # Plot the main planar sliding zone
            main_kws = _set_kws(main_kws, polygon=True,
                                color='r', alpha=0.3,
                                label='Potential Planar Sliding Zone')
            main.extend(axes.fill(
                *_rotate_shape(sliding_zone, self.strike), **main_kws))
            
            # Plot the secondary planar sliding zones
            if secondary_zone and sec_zone_present:
                secondary_kws = _set_kws(secondary_kws, polygon=True, 
                                        color='yellow', alpha=0.3,
                                        label='Secondary Planar Sliding Zone')            
                secondary_kws2 = secondary_kws.copy()
                secondary_kws2.pop('label')
                secondary.extend(axes.fill(
                    *_rotate_shape(sec_zone1, self.strike), **secondary_kws))
                secondary.extend(axes.fill(
                    *_rotate_shape(sec_zone2, self.strike),**secondary_kws2))

            # Plot the slope face
            if slopeface:
                slope_kws = _set_kws(slope_kws, color='k', label='Slope Face')
                slope.extend(axes.plane(self.strike, self.dip, **slope_kws))

            # Plot the construction lines (daylight envelope, friction cone 
            # and lateral limits)
            if construction_lines:
                daylight_kws = _set_kws(daylight_kws, polygon=True, edgecolor='r')
                friction_kws = _set_kws(friction_kws, polygon=True, edgecolor='r')
                lateral_kws = _set_kws(lateral_kws, color='r')
                lateral_kws2 = lateral_kws.copy()
                lateral_kws2.pop('label')
                daylight.extend(axes.fill(
                    *_rotate_shape(daylight_envelope, self.strike),**daylight_kws))
                friction.extend(axes.fill(
                    *friction_cone.exterior.xy, **friction_kws))
                lateral.extend(axes.plot(
                    *_rotate_shape(lat_lim1, self.strike), **lateral_kws))
                lateral.extend(axes.plot(
                    *_rotate_shape(lat_lim2, self.strike), **lateral_kws2))
            
            return dict(main=main, secondary=secondary, slope=slope,
                        daylight=daylight, friction=friction, lateral=lateral)

    class FlexuralToppling(object):
        """ 
        Kinematic analysis for flexural toppling failures
        
        Parameters
        ----------
        strike : number
            The strike of the slope face in degrees, with dip direction indicated 
            by the azimuth (e.g. 315 vs. 135) specified following the "right hand
            rule".
        dip : number (> 0 and <90)
            The dip of the slope face in degrees.
        fric_angle : number, default=35
            The friction angle along the toppling discontinuities, in degrees. 
            Note that the slope dip should be steeper than the friction angle, or 
            else no toppling zones can be generated.
        latlim : number (> 0 and <90), default=20
            The lateral limits for typical flexural toppling failures, counted from 
            the dip direction of the slope face in degrees. Discontinuities dipping 
            steeper than the slip limit for flexural toppling but outside the 
            lateral limits are considered to be less probable (i.e. secdondary 
            failure zones).
        """    
        
        def __init__(self, strike, dip, fric_angle=35, latlim=20):

            self.strike = strike
            self.dip = dip
            self.fric_angle = fric_angle
            self.latlim = latlim
            
            if latlim <= 0 :
                raise ValueError('latlim must be greater than 0 degree.')
                
            if latlim >= 90 :
                raise ValueError('latlim must be smaller than 90 degrees.'
                                ' Try 90-1e-9 if you really need to use 90.')

            if self.dip <= self.fric_angle:
                raise ValueError('No flexural toppling zones generated as the input'
                                ' slope dip is shallower than the friction angle')
                
        def check_failure(self, strikes, dips, curved_lateral_limits=True):
            """ 
            Check whether flexural toppling failures are kinematically feasible on 
            a sequence of discontinuities
            
            Parameters
            ----------
            strikes : numbers
                The strikes of the discontinuities in degrees, with dip direction 
                indicated by the azimuth (e.g. 315 vs. 135) specified following the
                "right hand rule".
            dips : numbers
                The dip angles of the discontinuities in degrees.
            curved_lateral_limits : boolean
                Consider lateral limits as curved lines (align with small circles) 
                if set to 'True'. Straight lines through the stereonet center are 
                used if set to 'False'. Defaults to 'True'
            
            Returns
            ----------
            main: squence of booleans
                True if the discontinuity is in the main flexural toppling zone
            secondary: squence of booleans
                True if the discontinuity is in the secondary flexural toppling zone
                Note: This is not normally considered
            """    

            strikes = (strikes-self.strike)%360
            dipdirs = (strikes+90)%360
            
            lons, lats = pole(strikes, dips)
            lats = np.degrees(lats)
            lons = np.degrees(lons)

            if curved_lateral_limits:
                within_lat = ((lats >= -self.latlim-1e-8) & # with tolerance
                            (lats <= self.latlim+1e-8))
            else:
                within_lat = ((dipdirs >= 270-self.latlim) &
                            (dipdirs <= 270+self.latlim))
        
            fric_slip = lons >= 90-self.dip+self.fric_angle-1e-8 # with tolerance
            
            main = within_lat & fric_slip
            secondary = ~within_lat & fric_slip
            
            return main, secondary
        
        def plot_kinematic(self, secondary_zone=False, construction_lines=True, 
                        slopeface=True, curved_lateral_limits=True,
                        main_kws=None, secondary_kws=None, lateral_kws=None,
                        slip_kws=None, slope_kws=None, 
                        ax=None):
            
            """
            Generate the flexural toppling kinematic analysis plot for pole vectors. 
            (Note: The discontinuity data to be used in conjunction with this plot 
            should be displayed as POLES)
            
            This function plots the following elements on a StereonetAxes: 
            (1) main flexural toppling zone
            (2) secondary flexural toppling zones (not normally considered)
            (3) construction lines, i.e. slip limit and lateral limits
            (4) slope face
            
            (2)-(4) are optioanl. The style of the elements above can be specified 
            with their kwargs, or on the artists returned by this function later.
            
            Parameters
            ----------
            secondary_zone : boolean
                Plot the secondary zones if set to True. This is not normally 
                considered. I just leave this option in case some users find it 
                useful. Defaults to 'False'.
            construction_lines : boolean
                Plot the construction lines if set to True. Defaults to 'True'.
            slopeface : boolean
                Plot the slope face as a great-circle plane on stereonet. Defaults
                to 'True'.
            curved_lateral_limits : boolean
                Plot curved lateral limits (align with small circles) if set to 
                True, or else will be plotted as straight lines through the 
                stereonet center. Defaults to 'True'
            main_kws : dictionary
                kwargs for the main flexural toppling zone 
                (matplotlib.patches.Polygon)
            secondary_kws : dictionary
                kwargs for the secondary flexural toppling zones 
                (matplotlib.patches.Polygon)
            lateral_kws : dictionary
                kwargs for the lateral limits (matplotlib.lines.Line2D)
            slip_kws : dictionary
                kwargs for the slip limit (matplotlib.lines.Line2D)
            slope_kws : dictionary
                kwargs for the slope face (matplotlib.lines.Line2D)
            ax : StereonetAxes
                The StereonetAxes to plot on. A new StereonetAxes will be generated
                if set to 'None'. Defaults to 'None'.
            
            Returns
            -------
            result : dictionary
                A dictionary mapping each element of the kinematic analysis plot to
                a list of the artists created. The dictionary has the following 
                keys:
                - `main` : the main flexural toppling zone
                - `secondary` : the two secondary flexural toppling zones
                - `slope` : the slope face
                - `slip` : the slip limit
                - `lateral` : the two lateral limits
            """

            # Convert the construction lines into shapely linestrings / polygons    
            envelope = _shape('flexural_envelope', strike=0, dip=self.dip, 
                                    angle=self.fric_angle)
            if curved_lateral_limits:
                lat_lim1, lat_lim2 = _shape('curved_latlims', angle=self.latlim)
            else:
                lat_lim1 = _shape('plane', strike=90+self.latlim, dip=90)
                lat_lim2 = _shape('plane', strike=90-self.latlim, dip=90)
            
            # Get the failure zones (as shapely polygons) from geometry interaction
            sec_zone1, toppling_zone = ops.split(envelope, lat_lim1)
            toppling_zone, sec_zone2 = ops.split(toppling_zone, lat_lim2)
            
            # Plotting
            if ax==None:
                figure, axes = subplots(figsize=(8, 8))
            else:
                axes = ax
            
            # List of artists to be output
            main = []
            secondary = []
            slope = []
            slip = []
            lateral = []

            # Plot the main flexural toppling sliding zone
            main_kws = _set_kws(main_kws, polygon=True,
                                color='r', alpha=0.3,
                                label='Potential Flexural Toppling Zone')
            main.extend(axes.fill(
                *_rotate_shape(toppling_zone, self.strike), **main_kws))
            
            # Plot the secondary flexural toppling zones
            if secondary_zone:
                secondary_kws = _set_kws(secondary_kws, polygon=True,
                                        color='yellow', alpha=0.3,
                                        label='Secondary Flexural Toppling Zone')
                secondary_kws2 = secondary_kws.copy()
                secondary_kws2.pop('label')
                secondary.extend(axes.fill(
                    *_rotate_shape(sec_zone1, self.strike), **secondary_kws))
                secondary.extend(axes.fill(
                    *_rotate_shape(sec_zone2, self.strike), **secondary_kws2))
            
            # Plot the slope face
            if slopeface:
                slope_kws = _set_kws(slope_kws, color='k', label='Slope Face')
                slope.extend(axes.plane(self.strike, self.dip, **slope_kws))

            # Plot the construction lines (friction cone and slip limit)
            if construction_lines:
                slip_kws = _set_kws(slip_kws, color='r')
                lateral_kws = _set_kws(lateral_kws, color='r')
                lateral_kws2 = lateral_kws.copy()
                lateral_kws2.pop('label')
                slip.extend(axes.plane(
                    self.strike, self.dip-self.fric_angle, **slip_kws))
                lateral.extend(axes.plot(
                    *_rotate_shape(lat_lim1, self.strike), **lateral_kws))
                lateral.extend(axes.plot(
                    *_rotate_shape(lat_lim2, self.strike), **lateral_kws2))
            
            return dict(main=main, secondary=secondary, slope=slope,
                        slip=slip, lateral=lateral)

    class WedgeSliding(object):
        """ 
        Kinematic analysis for wedge sliding failures
        
        Parameters
        ----------
        strike : number
            The strike of the slope face in degrees, with dip direction indicated 
            by the azimuth (e.g. 315 vs. 135) specified following the "right hand
            rule".
        dip : number (> 0 and <90)
            The dip of the slope face in degrees.
        fric_angle : number, default=35
            The friction angle along the discontinuity intersections, in degrees. 
            Note that the slope dip should be steeper than the friction angle, or 
            else no wedge sliding zones can be generated.
        """    
        
        def __init__(self, strike, dip, fric_angle=35):
            self.strike = strike
            self.dip = dip
            self.fric_angle = fric_angle
            
            if self.dip <= self.fric_angle:
                raise ValueError('No wedge sliding zones generated as the input'
                                ' slope dip is shallower than the friction angle.')
                
        def check_failure(self, bearings, plunges):
            """ 
            Check whether wedge sliding failures are kinematically feasible for a
            sequence of discontinuity intersection lines
            
            Parameters
            ----------
            bearing : number or sequence of numbers
                The bearing (azimuth) of the instersection line(s) in degrees.
            plunge : number or sequence of numbers
                The plunge of the line(s) in degrees. The plunge is measured in 
                degrees downward from the end of the feature specified by the 
                bearing.
            Returns
            ----------
            main: squence of booleans
                True if the discontinuity is in the main wedge sliding zone
            secondary: squence of booleans
                True if the discontinuity is in the secondary wedge sliding zone
            """    

            bearings = (bearings-self.strike)%360
            
            llons, llats = line(plunges, bearings)
            llons = np.degrees(llons)
            daylight = llons >= 90-self.dip-1e-8 # with tolerance
            
            slip = plunges >= self.fric_angle
            
            planar = llons <= 90-self.fric_angle+1e-8 # with tolerance
            
            main = slip & daylight
            secondary = ~slip & daylight & planar
            
            return main, secondary
        
        def plot_kinematic(self, secondary_zone=True, construction_lines=True, 
                        slopeface=True, main_kws=None, secondary_kws=None, 
                        friction_kws=None, fplane_kws=None, slope_kws=None, 
                        ax=None):
            
            """
            Generate the wedge sliding kinematic analysis plot for dip vectors. 
            (Note: This plot is used to analyze intersection lines between planes
            of discontinuities, displayed as "line" features instead of poles)
            
            This function plots the following elements on a StereonetAxes: 
            (1) main wedge sliding zone
            (2) secondary wedge sliding zones
            (3) construction lines, i.e. friction cone and friction plane
            (4) slope face
            
            (2)-(4) are optioanl. The style of the elements above can be specified 
            with their kwargs, or on the artists returned by this function later.
            
            Parameters
            ----------
            secondary_zone : boolean
                Plot the secondary zones if set to True. Defaults to 'True'.
            construction_lines : boolean
                Plot the construction lines if set to True. Defaults to 'True'.
            slopeface : boolean
                Plot the slope face as a great-circle plane on stereonet. Defaults
                to 'True'.
            main_kws : dictionary
                kwargs for the main wedge sliding zone 
                (matplotlib.patches.Polygon)
            secondary_kws : dictionary
                kwargs for the secondary wedge sliding zones 
                (matplotlib.patches.Polygon)
            fplane_kws : dictionary
                kwargs for the friction plane (matplotlib.lines.Line2D)
            slope_kws : dictionary
                kwargs for the slope face (matplotlib.lines.Line2D)
            ax : StereonetAxes
                The StereonetAxes to plot on. A new StereonetAxes will be generated
                if set to 'None'. Defaults to 'None'.
            
            Returns
            -------
            result : dictionary
                A dictionary mapping each element of the kinematic analysis plot to
                a list of the artists created. The dictionary has the following 
                keys:
                - `main` : the main wedge sliding zone
                - `secondary` : the secondary wedge sliding zones (it's one polygon)
                - `slope` : the slope face
                - `friction` : the friction cone
                - `fplane` : the friction plane
            """

            # Convert the construction lines into shapely linestrings / polygons
            # -1e-2 to prevent secondary zone splitting into two polygons
            friction_cone = _shape('cone', angle=90-self.fric_angle-1e-2)  
            envelope = _shape('wedge_envelope', strike=0, 
                            dip=self.dip, angle=self.fric_angle)
            
            # Get the failure zones (as shapely polygons) from geometry interaction
            wedge_zone = envelope.intersection(friction_cone)
            sec_zone = envelope.difference(friction_cone)
            
            # Plotting
            if ax==None:
                figure, axes = subplots(figsize=(8, 8))
            else:
                axes = ax
            
            # List of artists to be output
            main = []
            secondary = []
            slope = []
            friction = []
            fplane = []

            # Plot the main wedge sliding zone
            main_kws = _set_kws(main_kws, polygon=True,
                                color='r', alpha=0.3,
                                label='Potential Wedge Sliding Zone')
            main.extend(axes.fill(
                *_rotate_shape(wedge_zone, self.strike), **main_kws))
            
            # Plot the secondary main wedge sliding zones
            if secondary_zone:
                secondary_kws = _set_kws(secondary_kws, polygon=True,
                                        color='yellow', alpha=0.3,
                                        label='Secondary Wedge Sliding Zone')
                secondary.extend(axes.fill(
                    *_rotate_shape(sec_zone, self.strike), **secondary_kws))
                
            # Plot the slope face
            if slopeface:
                slope_kws = _set_kws(slope_kws, color='k', label='Slope Face')
                slope.extend(axes.plane(self.strike, self.dip, **slope_kws))

            # Plot the construction lines (friction cone and friction plane)
            if construction_lines:
                friction_kws = _set_kws(friction_kws, polygon=True, edgecolor='r')
                fplane_kws = _set_kws(fplane_kws, color='r')
                friction.extend(axes.fill(
                    *friction_cone.exterior.xy, **friction_kws))
                fplane.extend(axes.plane(
                    self.strike, self.fric_angle, **fplane_kws))

            return dict(main=main, secondary=secondary, slope=slope,
                        friction=friction, fplane=fplane)

    def plane_intersection(strike1, dip1, strike2, dip2):
        """
        Finds the intersection of two planes. Returns a plunge/bearing of the linear
        intersection of the two planes.
        Also accepts sequences of strike1s, dip1s, strike2s, dip2s.
        Parameters
        ----------
        strike1, dip1 : numbers or sequences of numbers
            The strike and dip (in degrees, following the right-hand-rule) of the
            first plane(s).
        strike2, dip2 : numbers or sequences of numbers
            The strike and dip (in degrees, following the right-hand-rule) of the
            second plane(s).
        Returns
        -------
        plunge, bearing : arrays
            The plunge and bearing(s) (in degrees) of the line representing the
            intersection of the two planes.
        """
        norm1 = sph2cart(*pole(strike1, dip1))
        norm2 = sph2cart(*pole(strike2, dip2))
        norm1, norm2 = np.array(norm1), np.array(norm2)
        lon, lat = cart2sph(*np.cross(norm1, norm2, axis=0))
        return geographic2plunge_bearing(lon, lat)


    discontinuity = np.loadtxt('kinematic_data1.txt', delimiter=',')
    intersections = np.loadtxt('kinematic_data2.txt', delimiter=',')
    jstrikes = discontinuity[:,1] - 90
    jdips = discontinuity[:,0]

    ibearings=[]
    iplunges=[]

    for i in range(len(jstrikes)):
        for j in range(i+1,len(jstrikes)):
            plunges, bearings= plane_intersection(jstrikes[i], jdips[i], jstrikes[j], jdips[j])
            ibearings.append(bearings[0])
            iplunges.append(plunges[0])

    ibearings=np.array(ibearings)
    iplunges=np.array(iplunges)

    # ibearings = intersections[:,1]
    # iplunges = intersections[:,0]

    # Set up kinematic analysis 
    strike, dip = 180, 75
    P3 = PlanarSliding(strike, dip)
    T3 = FlexuralToppling(strike, dip, latlim=15)
    #W3 = WedgeSliding(strike, dip)

    # Check data
    mainP, secP = P3.check_failure(jstrikes, jdips)
    mainT, _ = T3.check_failure(jstrikes, jdips)
    # mainW, secW = W3.check_failure(ibearings, iplunges)

    # Start plotting
    fig = plt.figure(figsize=(15, 6))

    ax1 = fig.add_subplot(1,3,1, projection='stereonet')
    ax1.set_title('Planar Sliding', loc='left')
    ax2 = fig.add_subplot(1,3,2, projection='stereonet')
    ax2.set_title('Flexural Toppling', loc='left')
    # ax3 = fig.add_subplot(1,3,3, projection='stereonet')
    # ax3.set_title('Wedge Sliding', loc='left')

    # Set up kinematic analysis plots
    P3.plot_kinematic(ax=ax1, slope_kws={'label':''}, main_kws={'label':''}, 
                    secondary_kws={'label':''})
    T3.plot_kinematic(ax=ax2, slope_kws={'label':''}, main_kws={'label':''}, 
                    secondary_kws={'label':''})
    # W3.plot_kinematic(ax=ax3, slope_kws={'label':''}, main_kws={'label':''}, 
    #                 secondary_kws={'label':''})

    # Plot planar sliding data
    ax1.pole(jstrikes, jdips, c='k', ms=1, 
            label='Discontinuities (Poles) [{}]'.format(len(jstrikes)))
    ax1.pole(jstrikes[mainP], jdips[mainP], c='r', ms=2, 
            label='Planar sliding possible [{}]'.format(sum(mainP)))
    ax1.pole(jstrikes[secP], jdips[secP], c='c', ms=2, 
            label='Planar sliding partially possible [{}]'.format(sum(secP)))

    # Plot flexural toppling data
    ax2.pole(jstrikes, jdips, c='k', ms=1, 
            label='Discontinuities (Poles) [{}]'.format(len(jstrikes)))
    ax2.pole(jstrikes[mainT], jdips[mainT], c='r',  ms=2, 
            label='Toppling possible [{}]'.format(sum(mainT)))

    # Plot wedge sliding data
    # ax3.plot(line(iplunges, ibearings), 'ok', ms=1, 
    #          label='Discontinuity intersections (Lines) [{}]'.format(len(iplunges)))
    # ax3.plot(line(iplunges[mainW], ibearings[mainW]), 'or', ms=2, 
    #          label='Wedge sliding possible [{}]'.format(sum(mainW)))
    # ax3.plot(line(iplunges[secW], ibearings[secW]), 'oc', ms=2, 
    #          label='Wedge sliding possible on single plane [{}]'.format(sum(secW)))

    plt.savefig("output.jpg")
    for ax in [ax1, ax2]:
        ax.set_azimuth_ticks([0,90,180,270], labels=['N', 'E', 'S', 'W'])
        ax.grid(linestyle=':')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))


    with open("output.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())

    ret_plot = str(encoded_string)
    ret_plot = ret_plot[2:]
    ret_plot = ret_plot[:-1]

    return ret_plot



if __name__ == "__main__":
    app.run()
