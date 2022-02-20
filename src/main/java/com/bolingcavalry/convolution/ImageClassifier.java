package main.java.com.bolingcavalry.convolution;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInitDistribution;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.*;

/**
 * ref.https://github.com/rahul-raj/Java-Deep-Learning-Cookbook
 */
public class ImageClassifier {

    private static final long seed = 12345;
    private static final Random randNumGen = new Random(seed);
    //    private static final Integer epoch = 100;
//    private static final Integer epoch = 10; // FIXME: large epoch will cause exception
    private static final Integer epoch = 3;
    private static final Integer batchSize = 10;
    private static final Integer height = 30;
    private static final Integer width = 30;
//    private static final Integer height = 32; // cifar10
//    private static final Integer width = 32;


    private static final String datasetName = "MultiClassWeatherDataset";
//    private static final String datasetName = "WeatherImageRecognition"; // FIXME: A fatal error has been detected by the Java Runtime Environment
//    private static final String datasetName = "cifar10_dl4j.v1";
//    private static final String datasetName = "stanford_cars_type";

    private static final Logger log = LoggerFactory.getLogger(ImageClassifier.class);

    public static void main(String[] args) throws Exception {

        //R,G,B channels
        int channels = 3;

        //load files and split
        File parentDir = new File("E:\\dataset\\" + datasetName);
//        File parentDir = new File("E:\\dataset\\StanfordCarBodyTypeData\\stanford_cars_type");
//        File parentDir = new File("E:\\dataset\\cifar10_dl4j.v1\\train");
        FileSplit fileSplit = new FileSplit(parentDir, NativeImageLoader.ALLOWED_FORMATS, new Random(42));
        int numLabels = Objects.requireNonNull(fileSplit.getRootDir().listFiles(File::isDirectory)).length;

        //identify labels in the path
        ParentPathLabelGenerator parentPathLabelGenerator = new ParentPathLabelGenerator();

        //file split to train/test using the weights.
        BalancedPathFilter balancedPathFilter = new BalancedPathFilter(new Random(42), NativeImageLoader.ALLOWED_FORMATS, parentPathLabelGenerator);
        InputSplit[] inputSplits = fileSplit.sample(balancedPathFilter, 80, 20);

        //get train/test data
        InputSplit trainData = inputSplits[0];
        InputSplit testData = inputSplits[1];

//        //Data augmentation
//        ImageTransform transform1 = new FlipImageTransform(new Random(42));
//        ImageTransform transform2 = new FlipImageTransform(new Random(123));
//        ImageTransform transform3 = new WarpImageTransform(new Random(42), 42);
//        ImageTransform transform4 = new RotateImageTransform(new Random(42), 40);
//        ImageTransform transform5 = new ResizeImageTransform(30, 30);
//
//        //pipelines to specify image transformation.
//        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
//                new Pair<>(transform1, 0.7),
//                new Pair<>(transform2, 0.6),
//                new Pair<>(transform3, 0.5),
//                new Pair<>(transform4, 0.4),
//                new Pair<>(transform5, 0.3)
//        );
//        ImageTransform transform = new PipelineImageTransform(pipeline);

        ImageTransform transform = new MultiImageTransform(
                randNumGen,
                new FlipImageTransform(new Random(42)),
                new FlipImageTransform(new Random(123)),
                new WarpImageTransform(new Random(42), 42),
                new RotateImageTransform(new Random(42), 40),
                new ResizeImageTransform(30, 30)
        );

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        MultiLayerConfiguration config = getModelConfig(channels, numLabels);

        //train without transformations
        ImageRecordReader imageRecordReader = new ImageRecordReader(height, width, channels, parentPathLabelGenerator);
        imageRecordReader.initialize(trainData, null);
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(imageRecordReader, batchSize, 1, numLabels);
        scaler.fit(dataSetIterator);
        dataSetIterator.setPreProcessor(scaler);
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(50), new EvaluativeListener(dataSetIterator, 1, InvocationType.EPOCH_END));

//        model.setListeners(new ScoreIterationListener(100)); //PerformanceListener for optimized training
        model.fit(dataSetIterator, epoch);

        //train with transformations
        imageRecordReader.initialize(trainData, transform);
        dataSetIterator = new RecordReaderDataSetIterator(imageRecordReader, batchSize, 1, numLabels);
        scaler.fit(dataSetIterator);
        dataSetIterator.setPreProcessor(scaler);
        model.fit(dataSetIterator, epoch);

        imageRecordReader.initialize(testData);
        dataSetIterator = new RecordReaderDataSetIterator(imageRecordReader, batchSize, 1, numLabels);
        scaler.fit(dataSetIterator);
        dataSetIterator.setPreProcessor(scaler);

        // Another way to perform evaluation.
        // ClassificationScoreCalculator accuracyCalculator = new ClassificationScoreCalculator(Evaluation.Metric.ACCURACY,dataSetIterator);
        // ClassificationScoreCalculator f1ScoreCalculator = new ClassificationScoreCalculator(Evaluation.Metric.F1,dataSetIterator);
        // ClassificationScoreCalculator precisionCalculator = new ClassificationScoreCalculator(Evaluation.Metric.PRECISION,dataSetIterator);
        // ClassificationScoreCalculator recallCalculator = new ClassificationScoreCalculator(Evaluation.Metric.RECALL,dataSetIterator);
        // Evaluation evaluation = model.evaluate(dataSetIterator);
        // System.out.println("Accuracy =" + accuracyCalculator.calculateScore(model) + "");
        // System.out.println("F1 Score =" + f1ScoreCalculator.calculateScore(model) + "");
        // System.out.println("Precision =" + precisionCalculator.calculateScore(model) + "");
        // System.out.println("Recall =" + recallCalculator.calculateScore(model) + "");

        Evaluation evaluation = model.evaluate(dataSetIterator);
        System.out.println("args = [" + evaluation.stats() + "]");

        File modelFile = new File("cnnTrainedModel-" + datasetName + new Date().getTime() + ".zip");
        log.info("saving model file to: " + modelFile.getAbsolutePath());
        ModelSerializer.writeModel(model, modelFile, true);
        ModelSerializer.addNormalizerToModel(modelFile, scaler);
    }

    public static MultiLayerConfiguration getModelConfig(int channels, int numLabels) {
        MultiLayerConfiguration config;
        config = new NeuralNetConfiguration.Builder()
                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE) // CUDNN GPU support optimize memory
//                .weightInit(WeightInit.DISTRIBUTION)
//                .dist(new NormalDistribution(0.0, 0.01))
                .activation(Activation.RELU)
                .updater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 1e-2, 0.1, 100000), 0.9))
                .biasUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 2e-2, 0.1, 100000), 0.9))
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .l2(5 * 1e-4)
                // .weightInit(WeightInit.XAVIER)
                // .updater(new Nesterovs(0.008D,0.9D))
                .list()
                .layer(new ConvolutionLayer.Builder(11, 11)
                        .nIn(channels)
                        .nOut(96)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(3, 3)
                        .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nOut(256)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new LocalResponseNormalization.Builder().name("lrn2").build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(3, 3)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(500)
//                        .dist(new NormalDistribution(0.001, 0.005))
                        .weightInit(new WeightInitDistribution(new NormalDistribution(0.001, 0.005)))
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(500)
//                        .dist(new NormalDistribution(0.001, 0.005))
                        .weightInit(new WeightInitDistribution(new NormalDistribution(0.001, 0.005)))
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, 3)).backpropType(BackpropType.Standard)
                .build();
        return config;
    }

}