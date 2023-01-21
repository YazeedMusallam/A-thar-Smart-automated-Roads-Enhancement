from fastapi import FastAPI, File, Request, UploadFile, status, Depends, Request
from segmentation import get_yolov7, get_image_from_bytes
from starlette.responses import Response
import io
from PIL import Image
import json
from fastapi.middleware.cors import CORSMiddleware
from database import engine, get_db
from sqlalchemy.orm import Session
import  modelz
import cv2
import numpy as np
import  math
import scipy.ndimage as ndimage
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import base64
model = get_yolov7()
modelz.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Custom YOLOV5 Machine Learning API",
    description="""Obtain object value out of image
                    and return image and json result""",
    version="0.0.1",
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# db = get_db



# def create_note(payload: schemas.PredBaseSchema, db: Session = Depends(get_db)):
#     print(**payload.dict())
#     pred = modelz.Pred(**payload.dict())
#     db.add(pred)
#     db.commit()
#     db.refresh(pred)
#     return {"status": "success", "note": pred}


@app.get('/notify/v1/health')
def get_health():
    """
    Usage on K8S
    readinessProbe:
        httpGet:
            path: /notify/v1/health
            port: 80
    livenessProbe:
        httpGet:
            path: /notify/v1/health
            port: 80
    :return:
        dict(msg='OK')
    """
    return dict(msg='OK')

def pixels_to_cm(pixels):
    return pixels * (2.54 / 96)

def Prepare_Data_without_outliers(warped_b):
    depth = warped_b[:,:] # or warped
    depth3d = np.copy(depth) # save a copy of the depth map in a new variable


    width = depth.shape[1] # store the width of depth image
    height = depth.shape[0]# store the height of depth image

    max_d = 190 # maximum disparity of Oak-D camera already known
    min_d = (882.5 * 7.5) / 190 # Minimum disparity of Oak-D camera

    sum_dis = 0 # store the sum of all the depth values in a variable
    disp_no =0 # store the total number of depth values
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            if depth[i][j]>=min_d and depth[i][j]<=max_d :
                sum_dis= sum_dis + depth[i][j]
                disp_no = disp_no + 1
    avg_depth = sum_dis/disp_no
    d1= depth.flatten()
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            if depth[i][j]<min_d or depth[i][j]>max_d :
                depth3d[i][j] = int(avg_depth)

    #We seperate out the garbage values and do not consider it while calculating the surfface fit

    depth1 = np.zeros((depth.shape[0],depth.shape[1]))
    w2 = []
    h2 =[]
    d2 =[]
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
                if(depth3d[i][j] == int(avg_depth) and depth[i][j]!= int(avg_depth)):
                    depth1[i][j]= depth3d[i][j]
                else:
                    d2.append(depth3d[i][j])
                    w2.append(i)
                    h2.append(j)
                    
    depth_estimate = depth3d.max()-depth3d.min()
    depth_estimate = pixel_to_cm(depth_estimate)
    return depth_estimate


def Prepare_Data_without_outliers(warped_b):
    depth = warped_b[:,:] # or warped
    depth3d = np.copy(depth) # save a copy of the depth map in a new variable


    width = depth.shape[1] # store the width of depth image
    height = depth.shape[0]# store the height of depth image

    max_d = 190 # maximum disparity of Oak-D camera already known
    min_d = (882.5 * 7.5) / 190 # Minimum disparity of Oak-D camera

    sum_dis = 0 # store the sum of all the depth values in a variable
    disp_no =0 # store the total number of depth values
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            if depth[i][j]>=min_d and depth[i][j]<=max_d :
                sum_dis= sum_dis + depth[i][j]
                disp_no = disp_no + 1
    avg_depth = sum_dis/disp_no
    d1= depth.flatten()
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            if depth[i][j]<min_d or depth[i][j]>max_d :
                depth3d[i][j] = int(avg_depth)

    #We seperate out the garbage values and do not consider it while calculating the surfface fit

    depth1 = np.zeros((depth.shape[0],depth.shape[1]))
    w2 = []
    h2 =[]
    d2 =[]
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
                if(depth3d[i][j] == int(avg_depth) and depth[i][j]!= int(avg_depth)):
                    depth1[i][j]= depth3d[i][j]
                else:
                    d2.append(depth3d[i][j])
                    w2.append(i)
                    h2.append(j)
                    
    depth_estimate = depth3d.max()-depth3d.min()
    depth_estimate = pixel_to_cm(depth_estimate)
    return depth_estimate

def pixel_to_cm(value):
  return value * (2.54/96)
def get_area_depth(x1,x2,y1,y2,w,h,input_image):
    warped = np.copy(input_image)
    xmin = int(x1*w)
    ymin = int(y1*h)
    xmax = int(x2*w)
    ymax = int(y2*h)
    # thickness = max(2, int(w/275))
    # cut_offset_y = int((ymax-ymin)/2)
    # cut_offset_x = int((xmax-xmin)/2)
    # out = input_image[xmin-cut_offset_x:xmax+cut_offset_x, ymin-cut_offset_y:ymax+cut_offset_y, :]
    # All points are in format [cols, rows]
    # pt_A = [xmin-cut_offset_x, ymin-cut_offset_y]
    # pt_B = [xmin-cut_offset_x, ymax+cut_offset_y]
    # pt_C = [xmax+cut_offset_x, ymax+cut_offset_y]
    # pt_D = [xmax+cut_offset_x, ymin-cut_offset_y]
    # # Here, I have used L2 norm. You can use L1 also.
    # width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    # width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    # maxWidth = max(int(width_AD), int(width_BC))


    # height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    # height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    # maxHeight = max(int(height_AB), int(height_CD))
    # input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    # output_pts = np.float32([[0, 0],
    #                         [0, maxHeight - 1],
    #                         [maxWidth - 1, maxHeight - 1],
    #                         [maxWidth - 1, 0]])
    # # Compute the perspective transform M
    # M = cv2.getPerspectiveTransform(input_pts,output_pts)
    # out = cv2.warpPerspective(input_image,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)
    warped_b = ndimage.gaussian_filter(input_image[int(y1):int(y2), int(x1):int(x2), 1],(5, 5))
    inpt = input_image[int(y1):int(y2), int(x1):int(x2), 1]

    xmin = pixels_to_cm(x1)
    ymin = pixels_to_cm(y1)
    xmax = pixels_to_cm(x2)
    ymax = pixels_to_cm(y2)  
    try:
        dep_est = Prepare_Data_without_outliers(warped_b)
    except ZeroDivisionError:
        try:
            dep_est = Prepare_Data_without_outliers(warped_b)
        except Exception as e: 
            print('haha: ', e)
            dep_est = 'Unknown'
            pass
    Area = pixel_to_cm((inpt[0].max())*(inpt[1].max()))*100
    # print('ABC:    ', inpt[0].max() )
    # Area = pixel_to_cm((warped_b.shape[0])*(warped_b.shape[1]))
    # Area = pixels_to_cm(Area)

    return int(Area), dep_est
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request, db: Session = Depends(get_db)):
    ''' Returns html jinja2 template render for home page form
    '''
    result = db.query(modelz.Pred).order_by(modelz.Pred.id.desc()).limit(5).all()
    #'area', 'category', 'depth', 'id', 'metadata', 'no_of_potholes', 'registry']
    return templates.TemplateResponse('index.html', {
            "request": request,
            "result" : result
        })

@app.get("/forms")
def home(request: Request):
    ''' Returns html jinja2 template render for home page form
    '''

    return templates.TemplateResponse('forms.html', {
            "request": request
        })

# Function to convert bounding boxes in YOLO format to xmin, ymin, xmax, ymax.
def yolo2bbox(bboxes):
    xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1]-bboxes[3]/2
    xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1]+bboxes[3]/2
    return xmin, ymin, xmax, ymax

@app.post("/report")
async def create_upload_file(request: Request, img: UploadFile, db: Session = Depends(get_db)):
    # print()
    input_image = get_image_from_bytes(img.file.read())
    
    if input_image == 0:
        return templates.TemplateResponse("nothing.html", {"request": request, "msg": "The input image is invalid"})
    input_image = np.ascontiguousarray(input_image, dtype=np.uint8)
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    detect_res = json.loads(detect_res)

    if len(detect_res) == 0:
        return templates.TemplateResponse("nothing.html", {"request": request, "msg": "No potholes has been detected!"})

    else:
        input_image = np.array(input_image)
        h, w, _ = input_image.shape
        classs={
        0:'Low',
        1:'Medium',
        2:'High',
        }
        pred_list = []
        for k in range(len(detect_res)):
            x1 = int(detect_res[k]['xmin'])
            x2 = int(detect_res[k]['xmax'])
            y1 = int(detect_res[k]['ymin'])
            y2 = int(detect_res[k]['ymax'])
            #   x1, y1, x2, y2 = yolo2bbox([x_c, y_c, w, h])
            #   h, w, _ = image.shape
            # x1, y1, x2, y2 = yolo2bbox(box)
            # # Denormalize the coordinates.
            # xmin = int(x1*w)
            # ymin = int(y1*h)
            # xmax = int(x2*w)
            # ymax = int(y2*h)
            cat = detect_res[k]['class']

            seveirty = classs.get(cat, "nothing")

            Area, Depth = get_area_depth(x1,x2,y1,y2,w,h,input_image)
            Area = int(Area)

            if type(Depth) == str or Depth <= 0.8 or Area <= 30:
                Depth = 'Unknown'
                Area = 'Unknown'
            else:
                Depth = int(Depth)
                Area = int(Area)
            
            pred = modelz.Pred(area=str(Area), depth=str(Depth), category=str(seveirty), no_of_potholes=len(detect_res), status='Unresolved')
            db.add(pred)
            db.commit()
            db.refresh(pred)
            pred_list.append(pred)
            # print((x1, y1), (x2, y2))
            cv2.rectangle(
                input_image, 
                (x1, y1), (x2, y2),
                color=(0, 0, 255),
                thickness=2
            )
            # font
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            
            # org
            org = (x2+10, y2+10)
            # Blue color in BGR
            color = (255, 0, 0)
            
            # Line thickness of 2 px
            thickness = 1
            # fontScale
            fontScale = 1
            cv2.putText(input_image, str(seveirty), org, font, 
                          fontScale, color, thickness, cv2.LINE_AA)
        # results.render()  # updates results.imgs with boxes and labels
        for img in results.imgs:
            bytes_io = io.BytesIO()
            img_base64 = Image.fromarray(input_image)
            img_base64 = img_base64.convert('RGB')
            img_base64.save(bytes_io, format="jpeg")
        # return {"result": detect_res}
        base64_encoded_image = base64.b64encode(bytes_io.getvalue()).decode("utf-8")
        return templates.TemplateResponse("profile.html", {"request": request, "pred_list":pred_list,"myImage":base64_encoded_image})  # "Area": int(Area), "Depth" : int(Depth), "Severity": seveirty, "Number": len(detect_res), 

@app.post("/uplod/")
async def create_file(img: bytes = File(...)):
    input_image = get_image_from_bytes(img)
    print(input_image)
    # results = model(img)
    # detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    # detect_res = json.loads(detect_res)
    # return {"result": detect_res}


@app.post("/object-to-json")
async def detect_food_return_json_result(request: Request, file: UploadFile = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    detect_res = json.loads(detect_res)
    return {"result": detect_res}


@app.post("/object-to-img")
async def detect_food_return_base64_img(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    print(detect_res)
    detect_res = json.loads(detect_res)
    results.render()  # updates results.imgs with boxes and labels
    for img in results.imgs:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(bytes_io, format="jpeg")

    payload = {
        "mime" : "image/png",
        "image": encoded_image_string,
        "some_other_data": None
    }
    # return Response(headers  =  detect_res[0]   , content=bytes_io.getvalue(), media_type="image/jpeg")
