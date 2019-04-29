import cairo,argparse,random

# remake of SatoshiRobatoFujimoto/arucomarker.py
# https://gist.github.com/SatoshiRobatoFujimoto/982a5721ea8842ded202d8a27886d0ea

#TEST: https://jcmellado.github.io/js-aruco/getusermedia/getusermedia.html
#http://terpconnect.umd.edu/~jwelsh12/enes100/markergen.html
#http://terpconnect.umd.edu/~jwelsh12/enes100/markers.js

# more online tool with different dictionaries
# http://chev.me/arucogen/

# aruco_25.pdf
# python ./arucomarker_gen.py --first=0 --count=60 --markersize=25 --border --bordersize=5 --spacing=0 --cols=5 --rows=8 --pagemargin=10

import string
import cv2.aruco as aruco

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000) # aruco.DICT_6X6_250

def drawMarker(canvas,id,sw,sh,x,y):
    marker = aruco.drawMarker(aruco_dict, id, aruco_dict.markerSize+2)

    sw = sw/(aruco_dict.markerSize+2)
    sh = sh/(aruco_dict.markerSize+2)
    for h in range(0,aruco_dict.markerSize+2):
        for w in range(0,aruco_dict.markerSize+2):
            # draw rectangle at ... w*sw+pad h*sh+pad sized sw,sh
            # filled and stroken in white or black ... 
            if marker[h][w]==0:
                ctx.set_source_rgb(0,0,0)
            else:
                #continue
                ctx.set_source_rgb(1,1,1)
            #ctx.rectangle(w*sw + x,h*sh + y,sw,sh);
            #ctx.stroke();
            ctx.rectangle(w*sw + x,h*sh + y,sw,sh);
            ctx.fill();
            ctx.stroke();


if __name__ == '__main__':
    import argparse

    pages = dict(A4=(210,297),A3=(297,420))

    parser = argparse.ArgumentParser(description='Aruco Page Maker to PDF, Emanuele Ruffaldi 2015')
    parser.add_argument('--page', default="A4", help='page size: A4 or A3')
    parser.add_argument('--pages',default=1,type=int,help="number of pages")
    parser.add_argument('--landscape', dest='landscape', action='store_const',const=True,default=False,help="set landscape")
    parser.add_argument('--portrait', dest='landscape', action='store_const',const=False,default=False,help="set landscape")
    parser.add_argument('--markersize', type=float,default=35,help="marker size (mm)")
    parser.add_argument('--bordersize', type=float, default=10,help="bourder around marker (mm)")
    parser.add_argument('--spacing', type=float,default=2,help="marker spacing in vertical and horizontal (mm)")
    parser.add_argument('--pagemargin', type=float,default=15,help="spacing default around (mm)")
    parser.add_argument('--fill', action="store_true",help="fills the page")
    parser.add_argument('--rows', type=int,default=5,help="fill rows")
    parser.add_argument('--cols', type=int,default=3,help="fill cols")
    parser.add_argument('--first', type=int,default=100,help="first id")
    parser.add_argument('--last', type=int,default=110,help="last id")
    parser.add_argument('--repeat', type=bool,default=False,help="repeat mode (ends at last)")
    parser.add_argument('--count', type=int,default=0,help="count (alternative to last)")
    parser.add_argument('--border', action='store_true',help="draws black border around")
    parser.add_argument('--pageborder', action='store_true',help="draws black border around")
    parser.add_argument('--axis', action='store_true',help="highlights axis")
    parser.add_argument('--random', action='store_true',help="randomize markers for board (and produces the randomization)")
    parser.add_argument('--output',default="aruco.pdf",help="outputfilename")


    args = parser.parse_args()

    page = pages[args.page]
    if args.landscape:
        page = (page[1],page[0])
    if args.count != 0:
        args.last = args.first + args.count - 1
    else:
        args.count = args.last - args.first + 1

    mm2pts = 2.83464567
    lw = 0.5 # mm
    lwdef = 0.5
    bordercolor = (0.5,0.5,0.5)
    if args.fill:
        args.cols = (page[0]-args.pagemargin*2)/(args.markersize+args.bordersize*2+args.spacing)
        args.rows = (page[1]-args.pagemargin*2)/(args.markersize+args.bordersize*2+args.spacing)
        print "fill results in rows x cols",args.rows,args.cols

    bid = 0

    width_pts, height_pts = page[0]*mm2pts,page[1]*mm2pts
    surface = cairo.PDFSurface (args.output, width_pts, height_pts)
    ctx = cairo.Context (surface)
    ctx.scale(mm2pts,mm2pts)
    done = False

    n = args.pages*args.rows*args.cols
    if args.count < n:
        n = args.count    
    markers = [args.first+i for i in range(0,n)]
    if args.random:
        random.shuffle(markers)
        open(args.output+".txt","w").write(" ".join([str(x) for x in markers]))

    for p in range(0,args.pages):
        if done:
            break
        if args.pageborder:
            ctx.set_source_rgb(*bordercolor)
            ctx.set_line_width(lw)
            ctx.rectangle(args.pagemargin,args.pagemargin,page[0]-args.pagemargin*2,page[1]-args.pagemargin*2)
            ctx.set_line_width(lwdef) # default
            ctx.stroke()
        y = args.pagemargin
        for r in range(0,args.rows):
            x = args.pagemargin    
            if done:
                break
            for c in range(0,args.cols):
                id = markers[bid % len(markers)]
                if not args.repeat and bid >= len(markers):
                    done = True
                    break
                drawMarker(ctx,id,args.markersize,args.markersize,x + args.bordersize,y + args.bordersize)
                bid = bid + 1
                if args.border:
                    ctx.set_source_rgb(*bordercolor)
                    ctx.set_line_width(lw)
                    ctx.rectangle(x,y,args.markersize + args.bordersize*2,args.markersize + args.bordersize*2)
                    ctx.set_line_width(lwdef) # default
                    ctx.stroke()
                if args.axis:
                    ctx.set_source_rgb(0, 0, 0)
                    ctx.select_font_face("Arial", cairo.FONT_SLANT_NORMAL,         cairo.FONT_WEIGHT_NORMAL)
                    (ax, ay, awidth, aheight, adx, ady) = ctx.text_extents("y>")
                    ctx.move_to(x + args.markersize -ax/mm2pts+args.bordersize-awidth/mm2pts,y-aheight*0.4)
                    ctx.show_text("y>")
                    (ax, ay, awidth, aheight, adx, ady) = ctx.text_extents("y")
                    ry0 = y +aheight*0.6 + args.markersize
                    rx0 = x + -awidth*1.2
                    ctx.move_to(rx0, ry0)
                    #ctx.show_text("v")
                    ry0 += aheight
                    ctx.move_to(rx0, ry0)
                    ctx.show_text("x")
                    ry0 += aheight
                    ctx.move_to(rx0, ry0)
                    ctx.show_text("v")
                x = x + args.bordersize*2+args.markersize + args.spacing
            y = y + args.markersize + args.bordersize*2 + args.spacing
        ctx.show_page()