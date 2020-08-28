package com.example.opencv_test;

import android.content.pm.ActivityInfo;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.Toast;

import com.googlecode.tesseract.android.TessBaseAPI;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCamera2View;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends CameraActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private static final String TAG = "OpenCV_Camera";
    public static final String TESS_DATA = "/tessdata";
    private JavaCamera2View javaCameraView;
    private int cameraId = JavaCamera2View.CAMERA_ID_ANY;
    private Mat mRgba;
    private Mat mRgbaF;
    private Mat mRgbaT;
    private List<MatOfPoint> contours = new ArrayList<>();
    private long lastTime;
    private Scalar lowerColor = new Scalar(22, 93, 0);
    private Scalar upperColor = new Scalar(45, 255, 255);
    private Scalar redColor = new Scalar(255, 0, 0);
    private Scalar greenColor = new Scalar(104, 191, 50);
    private TessBaseAPI tessBaseAPI = new TessBaseAPI();

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
        copyReadAssets();
        findView();
    }

    private void findView() {
        javaCameraView = findViewById(R.id.javaCameraView);
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
        Mat rgba = inputFrame.rgba();
        Mat src = null;
        long time = System.currentTimeMillis() / 3000;
        if (time > lastTime) {
            src = rgba.clone();
            System.out.println("Java onCameraFrame ---> " + time + " , " + lastTime);
            Imgproc.cvtColor(rgba, rgba, Imgproc.COLOR_RGB2HSV);// convert to HSV
            Imgproc.medianBlur(rgba, rgba, 5);
            Core.inRange(rgba, lowerColor, upperColor, rgba);
            Mat hierarchy = Mat.zeros(new Size(5, 5), CvType.CV_8UC1);
            Imgproc.findContours(rgba, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_TC89_L1);
            MatOfPoint matOfPoint = getMaxMatOfPoint(contours);
            extractMat(src, matOfPoint);
            release(rgba, hierarchy, matOfPoint);
        }
        lastTime = time;
        return src == null ? rgba : src;
    }

    private void extractMat(Mat mat, MatOfPoint matOfPoint) {
        if (mat == null || matOfPoint == null) {
            System.out.println("extractMat == null ");
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
//        prepareTessData();
        extractSubMat(mat, rect, rect.y, rect.height / 2, redColor, "/sdcard/pic/SubMat000001.jpg", false);
        extractSubMat(mat, rect, rect.y + rect.height / 2, rect.height, greenColor, "/sdcard/pic/SubMat000002.jpg", true);
    }

    private void extractSubMat(Mat mat, Rect rect, int startY, int endY, Scalar scalarColor, String filePath, boolean isRotate) {
        Point pt1 = new Point(rect.x, startY);
        Point pt2 = new Point(rect.x + rect.width, rect.y + endY);
        Rect rect1 = new Rect(pt1, pt2);
//        Imgproc.rectangle(mat, rect1, scalarColor, 3);
        Mat subMat = mat.submat(rect1);
        if (isRotate) {
            Core.flip(subMat, subMat, -1);
        }
        Bitmap bitmap = Bitmap.createBitmap(subMat.cols(), subMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(subMat, bitmap);
        ocr(bitmap);
//        System.out.println("寫入是否成功 : " + Imgcodecs.imwrite(filePath, submat));
    }

    private void release(Mat src, Mat hierarchy, MatOfPoint matOfPoint) {
        this.contours.clear();
        hierarchy.release();
        src.release();
        if (matOfPoint != null)
            matOfPoint.release();
    }

    private MatOfPoint getMaxMatOfPoint(List<MatOfPoint> contours) {
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

    private void ocr(Bitmap bitmap) {
        tessBaseAPI.setDebug(true);
        String dataPath = getExternalFilesDir(null).getPath() + File.separator;
        tessBaseAPI.init(dataPath, "eng");
        tessBaseAPI.setImage(bitmap);
        System.out.println("OCR 辨識結果 : " + tessBaseAPI.getUTF8Text());
    }

    private void copyReadAssets() {
        AssetManager assetManager = getAssets();
        InputStream inputStream;
        OutputStream outputStream;
        String strDir = getExternalFilesDir(TESS_DATA) + File.separator;
        File fileDir = new File(strDir);
        fileDir.mkdirs();
        File file = new File(fileDir, "eng.traineddata");
        try {
            inputStream = assetManager.open("eng.traineddata");
            outputStream = new BufferedOutputStream(new FileOutputStream(file));
            copyFile(inputStream, outputStream);
            inputStream.close();
            outputStream.flush();
            outputStream.close();
        } catch (Exception e) {
            System.out.println("Exception : " + e.getMessage());
        }
    }

    private void copyFile(InputStream inputStream, OutputStream outputStream) throws IOException {
        byte[] buffer = new byte[1024];
        int read;
        while ((read = inputStream.read(buffer)) != -1) {
            outputStream.write(buffer, 0, read);
        }
    }
}
