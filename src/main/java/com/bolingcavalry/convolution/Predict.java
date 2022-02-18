package main.java.com.bolingcavalry.convolution;

import org.bytedeco.javacpp.opencv_core;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.util.Arrays;

public class Predict {

    /**
     * FIXME: opencv lib not found: https://github.com/bytedeco/javacv/issues/48
     */
    public static void main(String[] args) throws Exception {
        String modelFilePath = "E:\\workspace\\DL4J_practice\\cnnTrainedModel-MultiClassWeatherDataset1645169436424.zip";
        String predictImgPath = "E:\\dataset\\MultiClassWeatherDataset\\Cloudy\\cloudy1.jpg";

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelFilePath);

        File file = new File(predictImgPath);
        opencv_core.Mat resizedImage = new opencv_core.Mat();
        opencv_core.Size sz = new opencv_core.Size(30, 30);
        opencv_core.Mat opencvImage = org.bytedeco.javacpp.opencv_imgcodecs.imread(file.getAbsolutePath());
        org.bytedeco.javacpp.opencv_imgproc.resize(opencvImage, resizedImage, sz);
        NativeImageLoader nativeImageLoader = new NativeImageLoader();
        INDArray image = nativeImageLoader.asMatrix(resizedImage);
        INDArray reshapedImage = image.reshape(1, 3, 30, 30);

        int[] result = model.predict(reshapedImage);
        System.out.print(Arrays.toString(result));
    }
}
