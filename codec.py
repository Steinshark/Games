import ffmpeg 
import os 

DB_DATA_SOURCE = r"//FILESERVER/S Drive/Media/Raw/EbayBuildJenny"
DB_DATA_DEST   = r"//FILESERVER/S Drive/Media/Raw/convert"


files   =    os.listdir(DB_DATA_SOURCE)
for f in files:
    stream  = ffmpeg.input(DB_DATA_SOURCE+"/"+f,f="mp4")
    #input(f"loading {DB_DATA_SOURCE+f}")
    stream   = ffmpeg.output(stream,DB_DATA_DEST+"/"+f,format="mp4",vcodec="h264",**{'threads': 6}) 
    ffmpeg.run(stream)
    