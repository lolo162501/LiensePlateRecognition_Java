package com.example.opencv_test;

import android.content.pm.ActivityInfo;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.os.Bundle;
import android.os.Looper;
import android.util.Log;
import android.view.Surface;
import android.view.View;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCamera2View;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends CameraActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private static final String TAG = "OpenCV_Camera";

    private JavaCamera2View javaCameraView;
    private View switchCameraBtn;
    private int cameraId = JavaCamera2View.CAMERA_ID_ANY;

    private Mat mRgba;
    private Mat mRgbaF;
    private Mat mRgbaT;
    private List<MatOfPoint> contours = new ArrayList<>();

    Scalar lowerColor = new Scalar(22, 93, 0);
    Scalar upperColor = new Scalar(45, 255, 255);

    Scalar borderColor = new Scalar(1, 127, 32);
    Scalar redColor = new Scalar(255, 0, 0);
    Scalar greenColor = new Scalar(104, 191, 50);

    private BaseLoaderCallback baseLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (status == LoaderCallbackInterface.SUCCESS) {
                if (javaCameraView != null) {
                    javaCameraView.setCvCameraViewListener(MainActivity.this);
                    javaCameraView.enableView();
                }
            } else {
                super.onManagerConnected(status);
            }
        }
    };

    //複寫父類的 getCameraViewList 方法，把 javaCameraView 送到父 Activity，一旦權限被授予之後，javaCameraView 的 setCameraPermissionGranted 就會自動被調用。
    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        List<CameraBridgeViewBase> list = new ArrayList<>();
        list.add(javaCameraView);
        return list;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        findView();
        setListener();
    }

    private void findView() {
        javaCameraView = findViewById(R.id.javaCameraView);
        switchCameraBtn = findViewById(R.id.switchCameraBtn);
    }

    private void setListener() {
        switchCameraBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                switch (cameraId) {
                    case JavaCamera2View.CAMERA_ID_ANY:
                    case JavaCamera2View.CAMERA_ID_BACK:
                        cameraId = JavaCamera2View.CAMERA_ID_FRONT;
                        break;
                    case JavaCamera2View.CAMERA_ID_FRONT:
                        cameraId = JavaCamera2View.CAMERA_ID_BACK;
                        break;
                }
                Log.i(TAG, "cameraId : " + cameraId);
                //切換前後攝像頭，要先禁用，設置完再啟用才會生效
                javaCameraView.disableView();
                javaCameraView.setCameraIndex(cameraId);
                javaCameraView.enableView();
            }
        });
    }

    @Override
    protected void onResume() {
        if (getRequestedOrientation() != ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE) {
            setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
            System.out.println("onResume -> setRequestedOrientation");
        }
        //横屏后才加载部件，否则会FC
        if (OpenCVLoader.initDebug()) {
            System.out.println("onResume -> OpenCVLoader.initDebug");
            baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        } else {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, baseLoaderCallback);
            System.out.println("onResume -> OpenCVLoader.initAsync " + OpenCVLoader.OPENCV_VERSION);
        }
        super.onResume();
    }

    @Override
    public void onPause() {
        super.onPause();
        if (javaCameraView != null) {
            javaCameraView.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (javaCameraView != null) {
            javaCameraView.disableView();
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        System.out.println("Thread->  onCameraViewStarted ---> " + (Looper.getMainLooper().getThread() == Thread.currentThread()));
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mRgbaF = new Mat(height, width, CvType.CV_8UC4);
        mRgbaT = new Mat(width, width, CvType.CV_8UC4);
    }

    @Override
    public void onCameraViewStopped() {
        System.out.println("Thread-> onCameraViewStopped ---> " + (Looper.getMainLooper().getThread() == Thread.currentThread()));
        mRgba.release();
        mRgbaF.release();
        mRgbaT.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        System.out.println("Thread-> onCameraFrame ---> " + (Looper.getMainLooper().getThread() == Thread.currentThread()));
        Mat src = inputFrame.rgba();
        Mat src1 = src.clone();
        Imgproc.cvtColor(src, src, Imgproc.COLOR_RGB2HSV);// convert to HSV
        Imgproc.medianBlur(src, src, 5);
        Core.inRange(src, lowerColor, upperColor, src);
        Mat hierarchy = Mat.zeros(new Size(5, 5), CvType.CV_8UC1);

        Imgproc.findContours(src, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_TC89_L1);
//        Imgproc.drawContours(src1, contours, -1, borderColor, 10, Imgproc.LINE_AA);
        MatOfPoint matOfPoint = max_MatOfPoint(contours);
        test(src1, matOfPoint);
        release(src, hierarchy, matOfPoint);
        return src1;

    }

    private void release(Mat src, Mat hierarchy, MatOfPoint matOfPoint) {
        this.contours.clear();
        hierarchy.release();
        src.release();
        if (matOfPoint != null)
            matOfPoint.release();
    }

    private void test(Mat mat, MatOfPoint matOfPoint) {
        if (mat == null || matOfPoint == null) {
            return;
        }
        MatOfPoint2f approxCurve = new MatOfPoint2f();
        MatOfPoint2f contour2f = new MatOfPoint2f(matOfPoint.toArray());
        //Processing on mMOP2f1 which is in type MatOfPoint2f
        double approxDistance = Imgproc.arcLength(contour2f, true) * 0.1;
        Imgproc.approxPolyDP(contour2f, approxCurve, approxDistance, true);
        //Convert back to MatOfPoint
        MatOfPoint points = new MatOfPoint(approxCurve.toArray());
        // Get bounding rect of contour
        Rect rect = Imgproc.boundingRect(points);
        // draw enclosing rectangle (all same color, but you could use variable i to make them unique)
        extractSubMat(mat, rect, rect.y, rect.height / 2, redColor, "/sdcard/Pictures/Top_SubMat.jpg");
        extractSubMat(mat, rect, rect.y + rect.height / 2, rect.height, greenColor, "/sdcard/Pictures/Bottom_SubMat.jpg");
    }

    private void extractSubMat(Mat mat, Rect rect, int startY, int endY, Scalar scalarColor, String filePath) {
        Point pt1 = new Point(rect.x, startY);
        Point pt2 = new Point(rect.x + rect.width, rect.y + endY);
        Rect rect1 = new Rect(pt1, pt2);
        Imgproc.rectangle(mat, rect1, scalarColor, 3);
        System.out.println("寫入是否成功 : " + Imgcodecs.imwrite(filePath, mat.submat(rect1)));
    }

    private void test2(Mat mat, MatOfPoint matOfPoint) {
        if (mat == null || matOfPoint == null) {
            return;
        }
        MatOfPoint2f contour2f = new MatOfPoint2f(matOfPoint.toArray());
        RotatedRect rotatedRect = Imgproc.minAreaRect(contour2f);
        Rect rect = rotatedRect.boundingRect();
//        Imgproc.rectangle(mat, rect.tl(), rect.br(), redColor, 6);
//        Imgproc.circle(mat, rotatedRect.center, 5, borderColor, 5);
        Imgproc.rectangle(mat, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + (rect.height / 2)), redColor, 3);
        Imgproc.rectangle(mat, new Point(rect.x, rect.y + (rect.height / 2)), new Point(rect.x + rect.width, rect.y + rect.height), greenColor, 3);
    }

    private MatOfPoint max_MatOfPoint(List<MatOfPoint> contours) {
        double area = 0;
        MatOfPoint matOfPoint = null;
        for (MatOfPoint mat : contours) {
            double tmp = Imgproc.contourArea(mat);
            if (area < tmp) {
                area = tmp;
                matOfPoint = mat;
            }
        }
        return matOfPoint;
    }

}
