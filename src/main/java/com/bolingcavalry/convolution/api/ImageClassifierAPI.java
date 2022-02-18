package main.java.com.bolingcavalry.convolution.api;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

public class ImageClassifierAPI {
    public static INDArray generateOutput(File inputFile, String modelFileLocation) throws IOException, InterruptedException {
        //retrieve the saved model
        final File modelFile = new File(modelFileLocation);
        final MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
        final RecordReader imageRecordReader = generateReader(inputFile);
        final ImagePreProcessingScaler normalizerStandardize = ModelSerializer.restoreNormalizerFromFile(modelFile);
        final DataSetIterator dataSetIterator = new RecordReaderDataSetIterator.Builder(imageRecordReader, 1).build();
        normalizerStandardize.fit(dataSetIterator);
        dataSetIterator.setPreProcessor(normalizerStandardize);
        return model.output(dataSetIterator);
    }

    private static RecordReader generateReader(File file) throws IOException, InterruptedException {
        final RecordReader recordReader = new ImageRecordReader(30, 30, 3);
        final InputSplit inputSplit = new FileSplit(file);
        recordReader.initialize(inputSplit);
        return recordReader;
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        final List<String> results = new ArrayList<>();
        String modelFilePath = "E:\\workspace\\DL4J_practice\\cnnTrainedModel-MultiClassWeatherDataset1645169436424.zip";
//        String predictImgPath = "E:\\dataset\\MultiClassWeatherDataset\\Rain\\rain183.jpg";
        String predictImgPath = "E:\\dataset\\MultiClassWeatherDataset\\Shine\\shine30.jpg";

        final File file = new File(predictImgPath);
        final INDArray indArray = generateOutput(file, modelFilePath);
        System.out.println(indArray);

        DecimalFormat df2 = new DecimalFormat("#.####");
        for (int i = 0; i < indArray.rows(); i++) {
            StringBuilder result = new StringBuilder("Image " + i + "->>>>>");
            for (int j = 0; j < indArray.columns(); j++) {
                result.append("\n Category ").append(j).append(": ").append(df2.format(indArray.getDouble(i, j) * 100)).append("%,   ");
            }
            result.append("\n\n");
            results.add(result.toString());
        }

        System.out.println(results);
    }
}


