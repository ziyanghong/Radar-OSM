import os
# import folium
import gpxpy
import gpxpy.gpx

def overlayGPX(gpxData, zoom=19):
    gpx_file = open(gpxData, 'r')
    gpx = gpxpy.parse(gpx_file)
    points = []
    for track in gpx.tracks:
        for segment in track.segments:   
            print(len(segment))     
    #         for point in segment.points:
    #             points.append(tuple([point.latitude, point.longitude]))
    # latitude = sum(p[0] for p in points)/len(points)
    # longitude = sum(p[1] for p in points)/len(points)
    # myMap = folium.Map(location=[latitude,longitude],zoom_start=zoom)
    # folium.PolyLine(points, color="red", weight=2.5, opacity=1).add_to(myMap)
    # return (myMap)


def main():
    gpx_file_name = '/media/data/RadarOSM/data/oxford_gps.gpx'
    overlayGPX(gpx_file_name)


if __name__== "__main__":
    main()
